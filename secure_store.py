import json
import os
import re
import secrets
import sqlite3
import tempfile
from pathlib import Path
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b'DALDB2\0'
SCHEMA = 2
HEADER = len(MAGIC) + 28
MAX_FILE = 512 * 1024 * 1024
MAX_PAYLOAD = 4 * 1024 * 1024
RETENTION = 200
MAX_KNOWLEDGE = 512 * 1024
KNOWLEDGE_LIMIT = 65
SOURCE = re.compile(r'^(?:local|pack:[0-9a-f]{16})$')


class StoreError(ValueError):
    cause = 'unknown'


class StoreFormatError(StoreError):
    cause = 'format'


class StoreAuthError(StoreError):
    cause = 'authentication'


class StoreVersionError(StoreError):
    cause = 'version'


class StoreKeyError(StoreError):
    cause = 'key'


class SecureStore:
    persistent = True

    def __init__(self, password, path, retention=RETENTION):
        if not isinstance(password, str) or len(password) < 12:
            raise StoreKeyError('Senha muito curta')
        self.path = Path(path)
        self.retention = max(1, int(retention))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = None
        if self.path.is_symlink():
            raise StoreFormatError('Caminho de armazenamento inválido')
        if self.path.exists():
            if not self.path.is_file() or self.path.stat().st_size > MAX_FILE:
                raise StoreFormatError('Formato de armazenamento incompatível')
            raw = self.path.read_bytes()
            if not raw.startswith(MAGIC) or len(raw) < HEADER + 16:
                raise StoreFormatError('Formato de armazenamento incompatível')
            salt = raw[len(MAGIC):len(MAGIC) + 16]
        else:
            salt = secrets.token_bytes(16)
        self.salt = salt
        self.key = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(password.encode('utf-8'))
        self.conn = sqlite3.connect(':memory:')
        self.closed = False
        try:
            if raw is not None:
                nonce = raw[len(MAGIC) + 16:HEADER]
                try:
                    plaintext = AESGCM(self.key).decrypt(nonce, raw[HEADER:], MAGIC)
                except InvalidTag:
                    raise StoreAuthError('Senha incorreta ou armazenamento danificado') from None
                try:
                    self.conn.deserialize(plaintext)
                except sqlite3.DatabaseError:
                    raise StoreFormatError('Armazenamento danificado') from None
            self._migrate()
        except StoreError:
            self.conn.close()
            raise
        except (sqlite3.Error, OSError):
            self.conn.close()
            raise StoreFormatError('Senha incorreta ou armazenamento danificado') from None

    def _migrate(self):
        version = self.conn.execute('PRAGMA user_version').fetchone()[0]
        if version > SCHEMA:
            raise StoreVersionError('Versão futura do banco')
        if version < 1:
            self.conn.execute('CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL)')
            self.conn.execute('PRAGMA user_version=1')
        if version < 2:
            self.conn.execute('CREATE INDEX IF NOT EXISTS episodes_created_idx ON episodes(created)')
            self.conn.execute('PRAGMA user_version=2')
        missing = not self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='knowledge'").fetchone()
        if missing:
            self.conn.execute('CREATE TABLE knowledge (source TEXT PRIMARY KEY, updated TEXT DEFAULT CURRENT_TIMESTAMP, payload TEXT NOT NULL)')
        self.conn.commit()
        if version < SCHEMA or missing or not self.path.exists():
            self.save()

    def save(self):
        nonce = secrets.token_bytes(12)
        encrypted = MAGIC + self.salt + nonce + AESGCM(self.key).encrypt(nonce, self.conn.serialize(), MAGIC)
        fd, temp = tempfile.mkstemp(prefix='.dal-', dir=self.path.parent)
        try:
            if os.name != 'nt':
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'wb') as handle:
                handle.write(encrypted)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def add(self, episode):
        if not isinstance(episode, dict):
            raise ValueError('Episódio inválido')
        data = json.dumps(episode, ensure_ascii=False, allow_nan=False)
        if len(data.encode('utf-8')) > MAX_PAYLOAD:
            raise ValueError('Episódio excede o limite')
        with self.conn:
            self.conn.execute('INSERT INTO episodes(payload) VALUES (?)', (data,))
            removed = self.conn.execute('DELETE FROM episodes WHERE id NOT IN (SELECT id FROM episodes ORDER BY id DESC LIMIT ?)', (self.retention,)).rowcount
        if removed:
            self.conn.execute('VACUUM')
        self.save()

    def recent(self, limit=20):
        limit = max(1, min(1000, int(limit)))
        rows = self.conn.execute('SELECT payload FROM episodes ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        result = []
        for (payload,) in rows:
            try:
                value = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    def count(self):
        return int(self.conn.execute('SELECT COUNT(*) FROM episodes').fetchone()[0])

    def put_knowledge(self, source, record):
        data = _knowledge_payload(source, record)
        exists = self.conn.execute('SELECT 1 FROM knowledge WHERE source = ?', (source,)).fetchone()
        if not exists and self.knowledge_count() >= KNOWLEDGE_LIMIT:
            raise ValueError('Limite de contribuições atingido')
        with self.conn:
            self.conn.execute('INSERT INTO knowledge(source, payload) VALUES (?, ?) ON CONFLICT(source) DO UPDATE SET payload = excluded.payload, updated = CURRENT_TIMESTAMP', (source, data))
        self.save()

    def knowledge_items(self):
        result = []
        for source, payload in self.conn.execute('SELECT source, payload FROM knowledge ORDER BY source'):
            try:
                value = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and SOURCE.match(source):
                result.append((source, value))
        return result

    def knowledge_count(self):
        return int(self.conn.execute('SELECT COUNT(*) FROM knowledge').fetchone()[0])

    def delete_knowledge(self, source):
        with self.conn:
            removed = self.conn.execute('DELETE FROM knowledge WHERE source = ?', (str(source),)).rowcount
        if removed:
            self.save()
        return bool(removed)

    def close(self):
        if not self.closed:
            self.closed = True
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _knowledge_payload(source, record):
    if not isinstance(source, str) or not SOURCE.match(source) or not isinstance(record, dict):
        raise ValueError('Contribuição inválida')
    data = json.dumps(record, ensure_ascii=False, allow_nan=False)
    if len(data.encode('utf-8')) > MAX_KNOWLEDGE:
        raise ValueError('Contribuição excede o limite')
    return data


class VolatileStore:
    persistent = False

    def __init__(self, retention=RETENTION):
        self.retention = max(1, int(retention))
        self.items = []
        self.knowledge = {}

    def put_knowledge(self, source, record):
        data = _knowledge_payload(source, record)
        if source not in self.knowledge and len(self.knowledge) >= KNOWLEDGE_LIMIT:
            raise ValueError('Limite de contribuições atingido')
        self.knowledge[source] = data

    def knowledge_items(self):
        return [(source, json.loads(data)) for source, data in sorted(self.knowledge.items())]

    def knowledge_count(self):
        return len(self.knowledge)

    def delete_knowledge(self, source):
        return self.knowledge.pop(source, None) is not None

    def add(self, episode):
        if not isinstance(episode, dict):
            raise ValueError('Episódio inválido')
        data = json.dumps(episode, ensure_ascii=False, allow_nan=False)
        if len(data.encode('utf-8')) > MAX_PAYLOAD:
            raise ValueError('Episódio excede o limite')
        self.items.append(data)
        del self.items[:-self.retention]

    def recent(self, limit=20):
        limit = max(1, min(1000, int(limit)))
        return [json.loads(item) for item in reversed(self.items[-limit:])]

    def count(self):
        return len(self.items)

    def close(self):
        self.items.clear()
        self.knowledge.clear()
