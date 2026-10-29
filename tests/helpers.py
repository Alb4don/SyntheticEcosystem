import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

BASE_FILES = {'app.py': b'VALUE = 1\n', 'simulation.py': b'X = 1\n', 'updates.py': b'Y = 1\n'}


def keypair():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return private, public


def make_zip(files, attrs=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = (attrs or {}).get(name, 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return buffer.getvalue()


def release_files(version, extra=None):
    files = dict(BASE_FILES)
    files['version.py'] = f"__version__ = '{version}'\n".encode()
    files.update(extra or {})
    return files


def drop(inbox, private, version, files=None, package=None, manifest_version=None):
    package = package if package is not None else make_zip(files if files is not None else release_files(version))
    manifest = json.dumps({'version': manifest_version or version, 'sha256': hashlib.sha256(package).hexdigest()}).encode()
    inbox = Path(inbox)
    (inbox / 'update.zip').write_bytes(package)
    (inbox / 'update.manifest.json').write_bytes(manifest)
    (inbox / 'update.sig').write_bytes(base64.b64encode(private.sign(manifest)))
    return manifest
