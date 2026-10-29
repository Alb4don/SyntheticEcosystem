import os
import secrets
import stat
import string
from pathlib import Path
from secure_store import SecureStore, StoreFormatError, StoreKeyError

HEX = frozenset(string.hexdigits.lower())
KEY_LENGTH = 64


def _read_key(keyfile):
    if keyfile.is_symlink() or not keyfile.is_file():
        raise StoreKeyError('Invalid key path')
    info = keyfile.stat()
    if info.st_size != KEY_LENGTH:
        raise StoreKeyError('Invalid key')
    if os.name != 'nt' and stat.S_IMODE(info.st_mode) & 0o077:
        os.chmod(keyfile, 0o600)
    key = keyfile.read_bytes()
    try:
        text = key.decode('ascii')
    except UnicodeDecodeError:
        raise StoreKeyError('Invalid key') from None
    if len(text) != KEY_LENGTH or not set(text) <= HEX:
        raise StoreKeyError('Invalid key')
    return text


class LocalStore(SecureStore):
    def __init__(self, path, retention=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise StoreFormatError('Invalid database path')
        keyfile = path.with_suffix('.key')
        if keyfile.exists() or keyfile.is_symlink():
            key = _read_key(keyfile)
        else:
            if path.exists():
                raise StoreKeyError('Existing database has no key')
            key = secrets.token_hex(32)
            fd = os.open(str(keyfile), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(key.encode('ascii'))
                stream.flush()
                os.fsync(stream.fileno())
        if retention is None:
            super().__init__(key, path)
        else:
            super().__init__(key, path, retention)
