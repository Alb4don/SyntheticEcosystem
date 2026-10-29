import argparse
import base64
import getpass
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from updates import DECLARED_VERSION, NAME_PATTERN, inspect_package, parse_version

HERE = Path(__file__).resolve().parent
EXCLUDED = re.compile(r'^(test_.*|conftest)\.py$')


def _passphrase(confirm=False):
    value = os.environ.get('DAL_SIGNING_PASSPHRASE') or getpass.getpass('Senha da chave de assinatura: ')
    if confirm and not os.environ.get('DAL_SIGNING_PASSPHRASE') and value != getpass.getpass('Confirme a senha: '):
        raise SystemExit('Senhas não conferem')
    if len(value) < 12:
        raise SystemExit('A senha deve ter pelo menos 12 caracteres')
    return value.encode('utf-8')


def keygen(args):
    private_path = Path(args.private)
    if private_path.exists():
        raise SystemExit('Arquivo de chave privada já existe')
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.BestAvailableEncryption(_passphrase(True)))
    fd = os.open(str(private_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(pem)
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    Path(args.public).write_text(public.hex() + '\n', encoding='ascii')
    print('Chaves geradas.')


def build(args):
    source = Path(args.source).resolve()
    text = (source / 'version.py').read_text(encoding='utf-8')
    match = DECLARED_VERSION.search(text)
    if not match:
        raise SystemExit('version.py inválido')
    version = match.group(1)
    parse_version(version)
    names = sorted(p.name for p in source.glob('*.py') if not EXCLUDED.match(p.name))
    names += [n for n in ('requirements.txt', 'README.md', 'update_key.pub', 'community_keys.txt') if (source / n).is_file()]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    package = output / 'update.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            if not NAME_PATTERN.match(name):
                raise SystemExit('Nome de arquivo não permitido no pacote')
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (source / name).read_bytes())
    data = package.read_bytes()
    inspect_package(data)
    manifest = json.dumps({'version': version, 'sha256': hashlib.sha256(data).hexdigest()}, sort_keys=True).encode('utf-8')
    key = serialization.load_pem_private_key(Path(args.private).read_bytes(), password=_passphrase())
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit('Chave privada não é Ed25519')
    (output / 'update.manifest.json').write_bytes(manifest)
    (output / 'update.sig').write_text(base64.b64encode(key.sign(manifest)).decode('ascii') + '\n', encoding='ascii')
    print(f'Pacote {version} assinado em {output}')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='release_tool')
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('keygen')
    generate.add_argument('--private', required=True)
    generate.add_argument('--public', default=str(HERE / 'update_key.pub'))
    generate.set_defaults(handler=keygen)
    package = commands.add_parser('build')
    package.add_argument('--source', default=str(HERE))
    package.add_argument('--private', required=True)
    package.add_argument('--output', required=True)
    package.set_defaults(handler=build)
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
