import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_PATTERNS = (
    (re.compile(r'(?<![A-Za-z0-9])[A-Fa-f0-9]{32,}(?![A-Za-z0-9])'), '[REDACTED-HEX]'),
    (re.compile(r'(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}'), '[REDACTED-B64]'),
    (re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+'), '[REDACTED-EMAIL]'),
)
_CONTROL = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')
MAX_LENGTH = 8000


def redact(text):
    text = str(text)
    home = str(Path.home())
    for variant in {home, home.replace('\\', '/'), home.replace('/', '\\')}:
        if variant and len(variant) > 1:
            text = re.sub(re.escape(variant), '~', text, flags=re.IGNORECASE)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    text = _CONTROL.sub('?', text)
    return text if len(text) <= MAX_LENGTH else text[:MAX_LENGTH] + '…[truncated]'


class RedactingFormatter(logging.Formatter):
    def format(self, record):
        return redact(super().format(record))


def configure(base, level=logging.WARNING):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    path = base / 'errors.log'
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, '_dal_handler', False):
            root.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding='utf-8')
    handler._dal_handler = True
    if os.name != 'nt':
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    handler.setFormatter(RedactingFormatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    root.addHandler(handler)
    root.setLevel(level)
    return handler
