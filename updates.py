import base64
import binascii
import hashlib
import io
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

MAX_MANIFEST = 4096
MAX_SIGNATURE = 256
MAX_PACKAGE = 50 * 1024 * 1024
MAX_ENTRY = 5 * 1024 * 1024
MAX_TOTAL = 50 * 1024 * 1024
MAX_ENTRIES = 256
MAX_RATIO = 200
MAX_BOOT_ATTEMPTS = 2
QUARANTINE_KEEP = 20
SETTLE_SECONDS = 2.
REQUIRED = frozenset({'app.py', 'version.py', 'updates.py', 'simulation.py'})
VERSION_PATTERN = re.compile(r'^(\d{1,4})\.(\d{1,4})\.(\d{1,4})$')
NAME_PATTERN = re.compile(r'^[A-Za-z0-9_\-]{1,64}(?:/[A-Za-z0-9_\-]{1,64}){0,3}\.(?:py|md|txt|json|pub)$')
DECLARED_VERSION = re.compile(r"^__version__ = '(\d{1,4}\.\d{1,4}\.\d{1,4})'\s*$", re.MULTILINE)
HEX64 = re.compile(r'^[0-9a-f]{64}$')
INBOX_FILES = ('update.zip', 'update.manifest.json', 'update.sig')

logger = logging.getLogger('dal.updates')


class UpdateError(ValueError):
    def __init__(self, message, cause='invalid', severity='medium'):
        super().__init__(message)
        self.cause = cause
        self.severity = severity


def parse_version(value):
    match = VERSION_PATTERN.match(value) if isinstance(value, str) else None
    if not match:
        raise UpdateError('Versão inválida', 'version')
    return tuple(int(part) for part in match.groups())


def load_trusted_key(path):
    path = Path(path)
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 256:
            return None
        text = path.read_text(encoding='ascii').strip()
        key = bytes.fromhex(text)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return key if len(key) == 32 else None


def _read_limited(path, limit):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise UpdateError('Arquivo de atualização inválido', 'file_type', 'high')
    if path.stat().st_size > limit:
        raise UpdateError('Arquivo de atualização excede o limite', 'size', 'high')
    with path.open('rb') as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise UpdateError('Arquivo de atualização excede o limite', 'size', 'high')
    return data


class SignedUpdates:
    def __init__(self, public_key_bytes):
        if not isinstance(public_key_bytes, (bytes, bytearray)) or len(public_key_bytes) != 32:
            raise ValueError('Chave pública Ed25519 inválida')
        self.key = Ed25519PublicKey.from_public_bytes(bytes(public_key_bytes))

    def verify_manifest(self, manifest_bytes, signature_b64):
        if not isinstance(manifest_bytes, (bytes, bytearray)) or len(manifest_bytes) > MAX_MANIFEST:
            raise UpdateError('Manifesto inválido', 'manifest', 'high')
        try:
            if isinstance(signature_b64, str):
                signature_b64 = signature_b64.encode('ascii')
            signature = base64.b64decode(bytes(signature_b64).strip(), validate=True)
        except (binascii.Error, ValueError, UnicodeEncodeError, TypeError):
            raise UpdateError('Assinatura inválida', 'signature', 'high') from None
        if len(signature) != 64:
            raise UpdateError('Assinatura inválida', 'signature', 'high')
        try:
            self.key.verify(signature, bytes(manifest_bytes))
        except InvalidSignature:
            raise UpdateError('Assinatura inválida', 'signature', 'high') from None
        try:
            manifest = json.loads(bytes(manifest_bytes).decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            raise UpdateError('Manifesto inválido', 'manifest', 'high') from None
        if not isinstance(manifest, dict) or set(manifest) != {'version', 'sha256'}:
            raise UpdateError('Manifesto inválido', 'manifest', 'high')
        if not isinstance(manifest['version'], str) or not isinstance(manifest['sha256'], str) or not HEX64.match(manifest['sha256']):
            raise UpdateError('Manifesto inválido', 'manifest', 'high')
        return manifest

    def verify(self, manifest_bytes, signature_b64, package_path):
        manifest = self.verify_manifest(manifest_bytes, signature_b64)
        digest = hashlib.sha256(_read_limited(package_path, MAX_PACKAGE)).hexdigest()
        if digest != manifest['sha256']:
            raise UpdateError('Pacote não corresponde ao manifesto assinado', 'digest', 'high')
        return manifest


def inspect_package(data):
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, ValueError, OSError):
        raise UpdateError('Pacote corrompido', 'archive') from None
    files = {}
    total = 0
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ENTRIES:
            raise UpdateError('Pacote com número de arquivos inválido', 'entries', 'high')
        for info in infos:
            name = info.filename
            if name.endswith('/'):
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise UpdateError('Pacote contém link simbólico', 'symlink', 'high')
            if mode not in (0, stat.S_IFREG):
                raise UpdateError('Pacote contém tipo de arquivo inválido', 'file_type', 'high')
            path = PurePosixPath(name)
            if '\\' in name or path.is_absolute() or '..' in path.parts or not NAME_PATTERN.match(name):
                raise UpdateError('Pacote contém caminho inválido', 'traversal', 'high')
            if name in files or name.lower() in {key.lower() for key in files}:
                raise UpdateError('Pacote contém nomes duplicados', 'duplicate', 'high')
            if info.flag_bits & 0x1:
                raise UpdateError('Pacote criptografado não suportado', 'encrypted', 'high')
            if info.file_size > MAX_ENTRY or info.file_size > max(info.compress_size, 1) * MAX_RATIO:
                raise UpdateError('Pacote com compressão suspeita', 'bomb', 'high')
            with archive.open(info) as handle:
                content = handle.read(MAX_ENTRY + 1)
            if len(content) > MAX_ENTRY or len(content) != info.file_size:
                raise UpdateError('Pacote com compressão suspeita', 'bomb', 'high')
            total += len(content)
            if total > MAX_TOTAL:
                raise UpdateError('Pacote excede o limite', 'bomb', 'high')
            files[name] = content
    missing = REQUIRED - set(files)
    if missing:
        raise UpdateError('Pacote incompleto', 'incomplete')
    return files


def declared_version(files):
    try:
        text = files['version.py'].decode('utf-8')
    except (KeyError, UnicodeDecodeError):
        raise UpdateError('Versão ausente', 'version') from None
    match = DECLARED_VERSION.search(text)
    if not match:
        raise UpdateError('Versão ausente', 'version')
    return match.group(1)


def _atomic_write(path, data):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix='.dal-', dir=path.parent)
    try:
        if os.name != 'nt':
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _default_state():
    return {'current': None, 'previous': None, 'pending': None, 'attempts': 0, 'healthy': False, 'bad': []}


class UpdateManager:
    def __init__(self, base, bundled_version, public_key_bytes, journal=None, clock=time.time):
        self.root = Path(base) / 'updates'
        self.inbox = self.root / 'inbox'
        self.releases = self.root / 'releases'
        self.quarantine = self.root / 'quarantine'
        self.state_path = self.root / 'state.json'
        for directory in (self.root, self.inbox, self.releases, self.quarantine):
            directory.mkdir(parents=True, exist_ok=True)
            if os.name != 'nt':
                try:
                    os.chmod(directory, 0o700)
                except OSError:
                    pass
        self.bundled_version = bundled_version
        self.bundled = parse_version(bundled_version)
        self.verifier = SignedUpdates(public_key_bytes)
        self.journal = journal
        self.clock = clock
        self.lock = threading.RLock()

    def _event(self, category, error):
        if self.journal is not None:
            self.journal.record(category, getattr(error, 'severity', 'medium'), getattr(error, 'cause', 'unknown'))

    def load_state(self):
        state = _default_state()
        try:
            raw = _read_limited(self.state_path, 65536)
            loaded = json.loads(raw.decode('utf-8'))
        except FileNotFoundError:
            return state
        except (UpdateError, OSError, UnicodeDecodeError, ValueError):
            self._event('update_state', UpdateError('Estado inválido', 'state', 'low'))
            return state
        if not isinstance(loaded, dict):
            return state
        for key in ('current', 'previous', 'pending'):
            value = loaded.get(key)
            if isinstance(value, str) and VERSION_PATTERN.match(value):
                state[key] = value
        attempts = loaded.get('attempts')
        if isinstance(attempts, int) and not isinstance(attempts, bool) and 0 <= attempts <= 100:
            state['attempts'] = attempts
        state['healthy'] = loaded.get('healthy') is True
        bad = loaded.get('bad')
        if isinstance(bad, list):
            state['bad'] = [v for v in bad if isinstance(v, str) and VERSION_PATTERN.match(v)][-50:]
        return state

    def save_state(self, state):
        _atomic_write(self.state_path, json.dumps(state, sort_keys=True).encode('utf-8'))

    def _known_versions(self, state):
        known = [self.bundled]
        for key in ('current', 'pending'):
            if state[key]:
                known.append(parse_version(state[key]))
        return max(known)

    def _release_dir(self, version):
        parse_version(version)
        return self.releases / version

    def _quarantine_inbox(self):
        entries = sorted((p for p in self.quarantine.iterdir() if p.is_dir() and not p.is_symlink()), key=lambda p: p.name)
        for old in entries[:max(0, len(entries) - QUARANTINE_KEEP + 1)]:
            shutil.rmtree(old, ignore_errors=True)
        stamp = time.strftime('%Y%m%d-%H%M%S')
        target = Path(tempfile.mkdtemp(prefix=f'{stamp}-', dir=self.quarantine))
        for name in INBOX_FILES:
            source = self.inbox / name
            if source.exists() or source.is_symlink():
                try:
                    os.replace(source, target / name)
                except OSError:
                    try:
                        source.unlink()
                    except OSError:
                        pass

    def _clear_inbox(self):
        for name in INBOX_FILES:
            try:
                (self.inbox / name).unlink()
            except FileNotFoundError:
                pass

    def scan_inbox(self):
        with self.lock:
            return self._scan_inbox()

    def _scan_inbox(self):
        if self.journal is not None and self.journal.blocked():
            return {'status': 'locked'}
        present = [(self.inbox / name).exists() or (self.inbox / name).is_symlink() for name in INBOX_FILES]
        if not any(present):
            return {'status': 'none'}
        if not all(present):
            return {'status': 'incomplete'}
        try:
            newest = max((self.inbox / name).lstat().st_mtime for name in INBOX_FILES)
        except OSError:
            return {'status': 'incomplete'}
        if self.clock() - newest < SETTLE_SECONDS:
            return {'status': 'incomplete'}
        try:
            manifest_bytes = _read_limited(self.inbox / 'update.manifest.json', MAX_MANIFEST)
            signature = _read_limited(self.inbox / 'update.sig', MAX_SIGNATURE)
            package = _read_limited(self.inbox / 'update.zip', MAX_PACKAGE)
            manifest = self.verifier.verify_manifest(manifest_bytes, signature)
            if hashlib.sha256(package).hexdigest() != manifest['sha256']:
                raise UpdateError('Pacote não corresponde ao manifesto assinado', 'digest', 'high')
            version = manifest['version']
            candidate = parse_version(version)
            state = self.load_state()
            if version in state['bad']:
                raise UpdateError('Versão marcada como defeituosa', 'known_bad', 'low')
            if candidate <= self._known_versions(state):
                raise UpdateError('Versão não é mais recente', 'downgrade', 'medium')
            files = inspect_package(package)
            if declared_version(files) != version:
                raise UpdateError('Versão declarada diverge do manifesto', 'version', 'high')
            self._stage(version, files, package, manifest_bytes, signature)
        except UpdateError as error:
            self._event('update_rejected', error)
            self._quarantine_inbox()
            return {'status': 'rejected', 'cause': error.cause}
        except OSError:
            logger.exception('Falha de E/S ao preparar atualização')
            return {'status': 'error'}
        state = self.load_state()
        state['pending'] = version
        self.save_state(state)
        self._clear_inbox()
        self._prune(state)
        return {'status': 'staged', 'version': version}

    def _stage(self, version, files, package, manifest_bytes, signature):
        target = self._release_dir(version)
        if target.exists():
            shutil.rmtree(target)
        temp = Path(tempfile.mkdtemp(prefix=f'.{version}-', dir=self.releases))
        try:
            code = temp / 'code'
            code.mkdir()
            root = code.resolve()
            for name, content in files.items():
                destination = (code / name).resolve()
                if root not in destination.parents:
                    raise UpdateError('Pacote contém caminho inválido', 'traversal', 'high')
                destination.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(destination, content)
            _atomic_write(temp / 'package.zip', package)
            _atomic_write(temp / 'manifest.json', manifest_bytes)
            _atomic_write(temp / 'manifest.sig', signature)
            os.replace(temp, target)
        finally:
            if temp.exists():
                shutil.rmtree(temp, ignore_errors=True)

    def _prune(self, state):
        keep = {state[key] for key in ('current', 'previous', 'pending') if state[key]}
        for entry in self.releases.iterdir():
            if entry.name in keep:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                try:
                    entry.unlink()
                except OSError:
                    pass

    def verify_release(self, version):
        try:
            target = self._release_dir(version)
            manifest_bytes = _read_limited(target / 'manifest.json', MAX_MANIFEST)
            signature = _read_limited(target / 'manifest.sig', MAX_SIGNATURE)
            manifest = self.verifier.verify(manifest_bytes, signature, target / 'package.zip')
            if manifest['version'] != version:
                raise UpdateError('Versão divergente', 'version', 'high')
            files = inspect_package(_read_limited(target / 'package.zip', MAX_PACKAGE))
            code = target / 'code'
            if code.is_symlink() or not code.is_dir():
                raise UpdateError('Release inválido', 'file_type', 'high')
            present = set()
            for path in code.rglob('*'):
                if path.is_symlink():
                    raise UpdateError('Release contém link simbólico', 'symlink', 'high')
                if path.is_file():
                    present.add(path.relative_to(code).as_posix())
            if present != set(files):
                raise UpdateError('Release adulterado', 'tampered', 'high')
            for name, content in files.items():
                if (code / name).read_bytes() != content:
                    raise UpdateError('Release adulterado', 'tampered', 'high')
            return True
        except UpdateError as error:
            self._event('release_integrity', error)
            return False
        except OSError:
            self._event('release_integrity', UpdateError('Release ilegível', 'io', 'low'))
            return False

    def select_boot(self):
        with self.lock:
            return self._select_boot()

    def _select_boot(self):
        state = self.load_state()
        if state['pending'] and state['pending'] not in state['bad'] and self._release_dir(state['pending']).is_dir():
            if state['current'] != state['pending']:
                state['previous'] = state['current']
            state['current'] = state['pending']
            state['attempts'] = 0
            state['healthy'] = False
        state['pending'] = None
        while state['current']:
            current = state['current']
            failure = None
            if parse_version(current) <= self.bundled:
                failure = 'superseded'
            elif not state['healthy'] and state['attempts'] >= MAX_BOOT_ATTEMPTS:
                failure = 'boot_failure'
                self._event('release_boot', UpdateError('Falhas de inicialização', 'boot_failure', 'low'))
            elif not self.verify_release(current):
                failure = 'integrity'
            if failure is None:
                break
            if failure != 'superseded':
                state['bad'] = (state['bad'] + [current])[-50:]
            fallback = state['previous']
            usable = fallback and fallback != current and fallback not in state['bad'] and parse_version(fallback) > self.bundled
            state['current'] = fallback if usable else None
            state['previous'] = None
            state['attempts'] = 0
            state['healthy'] = False
        if state['current'] and not state['healthy']:
            state['attempts'] += 1
        self.save_state(state)
        self._prune(state)
        return (self._release_dir(state['current']) / 'code') if state['current'] else None

    def mark_healthy(self):
        with self.lock:
            state = self.load_state()
            if state['current'] and not state['healthy']:
                state['healthy'] = True
                state['attempts'] = 0
                self.save_state(state)
                return True
            return False

    def status(self):
        state = self.load_state()
        return {'bundled': self.bundled_version, 'current': state['current'], 'pending': state['pending']}
