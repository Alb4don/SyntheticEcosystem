import argparse
import base64
import json
import os
import sys
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from knowledge import KnowledgeError, MAX_PACK_BYTES, load_pack_file, load_trusted_keys, pack_bytes, robust_merge
from version import __version__


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PACK_BYTES:
        raise KnowledgeError('Arquivo inválido', 'file')
    return path.read_bytes()


def validate(args):
    keys = load_trusted_keys(args.keys) if args.keys else frozenset()
    failures = 0
    seen = set()
    for name in args.files:
        try:
            item, digest = load_pack_file(_read(name), keys)
            if digest in seen:
                raise KnowledgeError('Pacote duplicado', 'duplicate')
            seen.add(digest)
            if args.require_signature and not item.verified:
                raise KnowledgeError('Assinatura confiável ausente', 'signature')
            print(f'OK {name} {digest[:16]} episódios={item.episodes} verificado={item.verified}')
        except (KnowledgeError, OSError) as error:
            failures += 1
            print(f'REJEITADO {name}: {error}')
    return 1 if failures else 0


def merge(args):
    keys = load_trusted_keys(args.keys) if args.keys else frozenset()
    items, seen = [], set()
    for name in args.files:
        item, digest = load_pack_file(_read(name), keys)
        if digest not in seen:
            seen.add(digest)
            items.append(item)
    merged = robust_merge(items)
    if merged is None:
        raise SystemExit('Nenhum pacote com evidência')
    Path(args.output).write_text(json.dumps({'pack': merged.to_pack(__version__)}, sort_keys=True, indent=1), encoding='utf-8')
    print(f'{len(items)} pacote(s) combinados em {args.output}')
    return 0


def sign(args):
    document = json.loads(_read(args.file).decode('utf-8'))
    passphrase = os.environ.get('DAL_SIGNING_PASSPHRASE')
    if passphrase is None:
        import getpass
        passphrase = getpass.getpass('Senha da chave de assinatura: ')
    key = serialization.load_pem_private_key(Path(args.private).read_bytes(), password=passphrase.encode('utf-8'))
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit('Chave privada não é Ed25519')
    pack = document['pack']
    load_pack_file(json.dumps({'pack': pack}).encode('utf-8'))
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    signed = {'pack': pack, 'key': public.hex(), 'signature': base64.b64encode(key.sign(pack_bytes(pack))).decode('ascii')}
    Path(args.output or args.file).write_text(json.dumps(signed, sort_keys=True, indent=1), encoding='utf-8')
    print('Pacote assinado.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog='knowledge_tool')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('validate')
    check.add_argument('files', nargs='+')
    check.add_argument('--keys')
    check.add_argument('--require-signature', action='store_true')
    check.set_defaults(handler=validate)
    combine = commands.add_parser('merge')
    combine.add_argument('files', nargs='+')
    combine.add_argument('--keys')
    combine.add_argument('--output', required=True)
    combine.set_defaults(handler=merge)
    signer = commands.add_parser('sign')
    signer.add_argument('file')
    signer.add_argument('--private', required=True)
    signer.add_argument('--output')
    signer.set_defaults(handler=sign)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (KnowledgeError, OSError, ValueError, KeyError) as error:
        print(f'Erro: {error}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
