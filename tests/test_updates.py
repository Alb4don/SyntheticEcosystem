import base64
import hashlib
import json
import time
import pytest
from hypothesis import given, settings, strategies as st
from helpers import drop, keypair, make_zip, release_files
from security import SecurityJournal
from updates import SignedUpdates, UpdateError, UpdateManager, inspect_package, load_trusted_key

LATER = lambda: time.time() + 60


@pytest.fixture
def env(tmp_path):
    private, public = keypair()
    journal = SecurityJournal()
    manager = UpdateManager(tmp_path, '9.0.0', public, journal, clock=LATER)
    return manager, private, public, journal


def test_signature_verification(tmp_path):
    private, public = keypair()
    package = tmp_path / 'update.zip'
    package.write_bytes(b'test payload')
    manifest = json.dumps({'version': '2.0', 'sha256': hashlib.sha256(package.read_bytes()).hexdigest()}).encode()
    signature = base64.b64encode(private.sign(manifest)).decode()
    verifier = SignedUpdates(public)
    assert verifier.verify(manifest, signature, package)['version'] == '2.0'
    package.write_bytes(b'tampered')
    with pytest.raises(ValueError):
        verifier.verify(manifest, signature, package)
    with pytest.raises(ValueError):
        verifier.verify(manifest, 'not-base64!!', package)
    with pytest.raises(ValueError):
        SignedUpdates(b'short')


def test_stage_boot_and_mark_healthy(env):
    manager, private, _, _ = env
    drop(manager.inbox, private, '9.1.0')
    assert manager.scan_inbox() == {'status': 'staged', 'version': '9.1.0'}
    assert not any(manager.inbox.iterdir())
    code = manager.select_boot()
    assert code is not None and (code / 'app.py').read_bytes() == b'VALUE = 1\n'
    assert manager.mark_healthy()
    assert manager.select_boot() == code
    assert manager.load_state()['attempts'] == 0


@pytest.mark.parametrize('name', ['../evil.py', '/abs.py', 'dir\\evil.py', 'a/../../x.py', 'evil.exe', 'C:/x.py', ''.join(['a'] * 70) + '.py'])
def test_rejects_hostile_paths(env, name):
    manager, private, _, journal = env
    drop(manager.inbox, private, '9.1.0', release_files('9.1.0', {name: b'x'}))
    result = manager.scan_inbox()
    assert result['status'] == 'rejected'
    assert not (manager.inbox / 'update.zip').exists()
    assert any(manager.quarantine.iterdir())
    assert journal.score() > 0


def test_rejects_symlink_entries(env):
    manager, private, _, _ = env
    files = release_files('9.1.0', {'link.py': b'/etc/passwd'})
    drop(manager.inbox, private, '9.1.0', package=make_zip(files, {'link.py': 0o120777}))
    assert manager.scan_inbox() == {'status': 'rejected', 'cause': 'symlink'}


def test_rejects_zip_bomb(env):
    manager, private, _, _ = env
    drop(manager.inbox, private, '9.1.0', release_files('9.1.0', {'bomb.txt': b'\0' * (6 * 1024 * 1024)}))
    assert manager.scan_inbox() == {'status': 'rejected', 'cause': 'bomb'}


def test_rejects_downgrade_and_version_mismatch(env):
    manager, private, _, _ = env
    drop(manager.inbox, private, '8.9.9')
    assert manager.scan_inbox()['cause'] == 'downgrade'
    drop(manager.inbox, private, '9.2.0', release_files('9.3.0'))
    assert manager.scan_inbox()['cause'] == 'version'


def test_rejects_foreign_signature(env):
    manager, _, _, journal = env
    attacker, _ = keypair()
    drop(manager.inbox, attacker, '9.1.0')
    assert manager.scan_inbox() == {'status': 'rejected', 'cause': 'signature'}
    assert manager.select_boot() is None


def test_incomplete_or_still_copying_is_not_an_attack(tmp_path):
    private, public = keypair()
    journal = SecurityJournal()
    manager = UpdateManager(tmp_path, '9.0.0', public, journal, clock=time.time)
    (manager.inbox / 'update.zip').write_bytes(b'partial')
    assert manager.scan_inbox() == {'status': 'incomplete'}
    drop(manager.inbox, private, '9.1.0')
    assert manager.scan_inbox() == {'status': 'incomplete'}
    assert journal.score() == 0


def test_failed_boots_roll_back(env):
    manager, private, _, _ = env
    drop(manager.inbox, private, '9.1.0')
    manager.scan_inbox()
    first = manager.select_boot()
    manager.mark_healthy()
    drop(manager.inbox, private, '9.2.0')
    manager.scan_inbox()
    assert manager.select_boot().parent.name == '9.2.0'
    assert manager.select_boot().parent.name == '9.2.0'
    assert manager.select_boot() == first
    state = manager.load_state()
    assert '9.2.0' in state['bad'] and state['current'] == '9.1.0'


def test_tampered_release_falls_back_to_bundled(env):
    manager, private, _, journal = env
    drop(manager.inbox, private, '9.1.0')
    manager.scan_inbox()
    code = manager.select_boot()
    (code / 'app.py').write_bytes(b'import os\n')
    assert manager.select_boot() is None
    assert '9.1.0' in manager.load_state()['bad']
    assert journal.score() >= 4


def test_planted_bytecode_is_detected(env):
    manager, private, _, _ = env
    drop(manager.inbox, private, '9.1.0')
    manager.scan_inbox()
    code = manager.select_boot()
    (code / '__pycache__').mkdir()
    (code / '__pycache__' / 'app.cpython-313.pyc').write_bytes(b'evil')
    assert manager.select_boot() is None


def test_corrupted_state_is_recovered(env):
    manager, _, _, _ = env
    manager.state_path.write_text('{not json')
    assert manager.load_state()['current'] is None
    manager.state_path.write_text(json.dumps({'current': '../../x', 'attempts': -5, 'bad': [1, '9.9.9']}))
    state = manager.load_state()
    assert state['current'] is None and state['attempts'] == 0 and state['bad'] == ['9.9.9']


def test_repeated_attacks_trigger_lockdown(env):
    manager, _, _, journal = env
    attacker, _ = keypair()
    for _ in range(2):
        drop(manager.inbox, attacker, '9.1.0')
        manager.scan_inbox()
    assert journal.blocked()
    drop(manager.inbox, attacker, '9.1.0')
    assert manager.scan_inbox() == {'status': 'locked'}


def test_load_trusted_key(tmp_path):
    _, public = keypair()
    path = tmp_path / 'k.pub'
    path.write_text(public.hex())
    assert load_trusted_key(path) == public
    path.write_text('zz')
    assert load_trusted_key(path) is None
    assert load_trusted_key(tmp_path / 'missing') is None


def test_launcher_selects_verified_release(tmp_path, monkeypatch):
    import launcher
    private, public = keypair()
    here = tmp_path / 'install'
    here.mkdir()
    (here / 'update_key.pub').write_text(public.hex())
    base = tmp_path / 'home'
    monkeypatch.setattr(launcher, 'HERE', here)
    monkeypatch.setattr(launcher, 'BASE', base)
    assert launcher.choose_release() is None
    manager = UpdateManager(base, '9.0.0', public, clock=LATER)
    drop(manager.inbox, private, '9.4.0')
    assert manager.scan_inbox()['status'] == 'staged'
    assert launcher.choose_release().parent.name == '9.4.0'


@given(st.binary(max_size=600), st.binary(max_size=120))
@settings(max_examples=200, deadline=None)
def test_fuzz_manifest_and_signature(manifest, signature):
    _, public = keypair()
    with pytest.raises(UpdateError):
        SignedUpdates(public).verify_manifest(manifest, signature)


@given(st.binary(max_size=2000))
@settings(max_examples=200, deadline=None)
def test_fuzz_package_parser(data):
    with pytest.raises(UpdateError):
        inspect_package(data)


@given(st.dictionaries(st.text(min_size=1, max_size=20), st.binary(max_size=50), max_size=6))
@settings(max_examples=150, deadline=None)
def test_fuzz_package_names(files):
    try:
        package = make_zip(files)
    except (ValueError, UnicodeEncodeError):
        return
    try:
        result = inspect_package(package)
    except UpdateError:
        return
    assert all('..' not in name and not name.startswith('/') for name in result)


def test_launcher_process_boots_staged_release(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    install = tmp_path / 'install'
    install.mkdir()
    for source in root.glob('*.py'):
        shutil.copy(source, install / source.name)
    private, public = keypair()
    (install / 'update_key.pub').write_text(public.hex())
    home = tmp_path / 'home'
    home.mkdir()
    manager = UpdateManager(home / '.digital_anthropology_lab', '9.0.0', public, clock=LATER)
    app_code = b"import updates\nimport version\ndef main(**kw):\n    print('RELEASE', version.__version__, kw['release_active'], updates.Y)\n    return 0\n"
    drop(manager.inbox, private, '9.5.0', release_files('9.5.0', {'app.py': app_code}))
    assert manager.scan_inbox()['status'] == 'staged'
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home), QT_QPA_PLATFORM='offscreen')
    result = subprocess.run([sys.executable, str(install / 'launcher.py')], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-500:]
    assert 'RELEASE 9.5.0 True 1' in result.stdout
    assert not list((manager.releases / '9.5.0' / 'code').rglob('__pycache__'))


def test_quarantine_is_bounded(env):
    manager, _, _, journal = env
    attacker, _ = keypair()
    journal.threshold = 10**6
    for _ in range(25):
        drop(manager.inbox, attacker, '9.1.0')
        manager.scan_inbox()
    assert len(list(manager.quarantine.iterdir())) <= 20
