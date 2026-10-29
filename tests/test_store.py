import os
import pytest
from hypothesis import given, settings, strategies as st
from learning import EpisodeLearner
from local_store import LocalStore
from secure_store import MAGIC, SecureStore, StoreAuthError, StoreError, StoreFormatError, StoreKeyError, VolatileStore

PASSWORD = 'correct horse battery'


def test_full_database_encryption(tmp_path):
    path = tmp_path / 'db.secure'
    store = SecureStore(PASSWORD, path)
    store.add({'secret': 'synthetic-secret-marker'})
    store.close()
    raw = path.read_bytes()
    assert raw.startswith(MAGIC)
    assert not raw.startswith(b'SQLite format 3')
    assert b'synthetic-secret-marker' not in raw
    with pytest.raises(ValueError):
        SecureStore('incorrect horse battery', path)
    store = SecureStore(PASSWORD, path)
    assert store.recent()[0]['secret'] == 'synthetic-secret-marker'
    assert store.conn.execute('PRAGMA user_version').fetchone()[0] == 2
    assert store.knowledge_items() == []
    store.close()


def test_single_bit_tamper_is_detected(tmp_path):
    path = tmp_path / 'db.secure'
    with SecureStore(PASSWORD, path) as store:
        store.add({'a': 1})
    raw = bytearray(path.read_bytes())
    raw[-5] ^= 0x01
    path.write_bytes(bytes(raw))
    with pytest.raises(StoreAuthError):
        SecureStore(PASSWORD, path)


def test_truncated_store_is_format_error(tmp_path):
    path = tmp_path / 'db.secure'
    path.write_bytes(MAGIC + b'x' * 10)
    with pytest.raises(StoreFormatError):
        SecureStore(PASSWORD, path)


@given(st.binary(max_size=400))
@settings(max_examples=120, deadline=None)
def test_fuzz_store_file_never_crashes(tmp_path_factory, data):
    path = tmp_path_factory.mktemp('fuzz') / 'db.secure'
    path.write_bytes(data)
    with pytest.raises(StoreError):
        SecureStore(PASSWORD, path)


@given(st.binary(min_size=40, max_size=300))
@settings(max_examples=60, deadline=None)
def test_fuzz_store_with_valid_magic(tmp_path_factory, data):
    path = tmp_path_factory.mktemp('fuzzmagic') / 'db.secure'
    path.write_bytes(MAGIC + data)
    with pytest.raises(StoreError):
        SecureStore(PASSWORD, path)


def test_retention_and_payload_limits(tmp_path):
    store = SecureStore(PASSWORD, tmp_path / 'db', retention=5)
    for i in range(12):
        store.add({'i': i})
    assert store.count() == 5
    assert [e['i'] for e in store.recent(10)] == [11, 10, 9, 8, 7]
    with pytest.raises(ValueError):
        store.add({'big': 'x' * (5 * 1024 * 1024)})
    with pytest.raises(ValueError):
        store.add(['not', 'a', 'dict'])
    with pytest.raises(ValueError):
        store.add({'nan': float('nan')})
    store.close()
    store.close()


def test_short_password_rejected(tmp_path):
    with pytest.raises(StoreKeyError):
        SecureStore('short', tmp_path / 'db')


def test_local_store_key_lifecycle(tmp_path):
    path = tmp_path / 'episodes.secure'
    store = LocalStore(path)
    store.add({'k': 1})
    store.close()
    key = path.with_suffix('.key')
    assert len(key.read_bytes()) == 64
    if os.name != 'nt':
        assert (key.stat().st_mode & 0o777) == 0o600
    with LocalStore(path) as store:
        assert store.recent()[0]['k'] == 1


def test_local_store_rejects_bad_keys(tmp_path):
    path = tmp_path / 'episodes.secure'
    LocalStore(path).close()
    key = path.with_suffix('.key')
    key.write_bytes(b'z' * 64)
    with pytest.raises(StoreKeyError):
        LocalStore(path)
    key.write_bytes(b'a' * 10)
    with pytest.raises(StoreKeyError):
        LocalStore(path)
    key.unlink()
    with pytest.raises(StoreKeyError):
        LocalStore(path)


@pytest.mark.skipif(os.name == 'nt', reason='symlinks need privileges on Windows')
def test_local_store_rejects_symlinked_key(tmp_path):
    path = tmp_path / 'episodes.secure'
    target = tmp_path / 'elsewhere.key'
    target.write_bytes(b'a' * 64)
    path.with_suffix('.key').symlink_to(target)
    with pytest.raises(StoreKeyError):
        LocalStore(path)


def test_learning(tmp_path):
    store = SecureStore(PASSWORD, tmp_path / 'db')
    assert EpisodeLearner(store).recommend() is None
    store.add({'config': {'population': 20, 'steps': 5, 'seed': 3}, 'final': {'polarization': .23}})
    result = EpisodeLearner(store).recommend(.25)
    assert result['config']['seed'] == 3
    assert result['validated'] is False
    store.close()


def test_learning_ignores_malformed_episodes():
    store = VolatileStore()
    for bad in ({'config': 'x'}, {'config': {'population': True, 'steps': 5, 'seed': 1}, 'final': {'polarization': .1}}, {'config': {'population': 10, 'steps': 5, 'seed': 1}, 'final': {'polarization': '0.2'}}, {'config': {'population': 10, 'steps': 5, 'seed': 1}, 'final': {'polarization': 7}}, {'config': {'population': 99999, 'steps': 5, 'seed': 1}, 'final': {'polarization': .2}}):
        store.add(bad)
    assert EpisodeLearner(store).recommend() is None
    assert EpisodeLearner(store).recommend(float('nan')) is None
    store.add({'config': {'population': 10, 'steps': 5, 'seed': 1}, 'final': {'polarization': .2}})
    assert EpisodeLearner(store).recommend()['samples'] == 1
