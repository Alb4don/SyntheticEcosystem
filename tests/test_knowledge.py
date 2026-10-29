import base64
import json
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import knowledge as K
from dialogue import ACTS, CONTEXTS
from knowledge import DialogueKnowledge, KnowledgeError, load_pack_file, pack_bytes, robust_merge, validate_pack
from secure_store import SecureStore, VolatileStore
from simulation import Config, Simulation


def _trained(seed=1, steps=200, population=40, inherited=None):
    simulation = Simulation(Config(population, steps, seed), inherited)
    simulation.run()
    return simulation.knowledge()


def _pack(item):
    return json.dumps({'pack': item.to_pack('9.0.0')}).encode()


def test_knowledge_from_simulation_is_bounded():
    item = _trained()
    assert item.means.shape == (CONTEXTS, len(ACTS))
    assert ((item.means >= 0) & (item.means <= 1)).all() and item.strength() > 0
    assert any(item.inherited(t) for t in range(4))


def test_inherited_prior_is_not_counted_as_new_evidence():
    teacher = _trained()
    fresh = Simulation(Config(30, 5, 3), teacher)
    assert fresh.inherited and fresh.knowledge().strength() == 0


def test_warm_start_beats_cold_start_on_unseen_seeds():
    teacher = None
    for generation in range(3):
        learned = _trained(100 + generation, 400, 60, teacher)
        teacher = learned if teacher is None else teacher.merge(learned)
    gains = []
    for seed in (7, 8, 9):
        cold, warm = Simulation(Config(60, 120, seed)), Simulation(Config(60, 120, seed), teacher)
        cold.run()
        warm.run()
        gains.append(np.mean([r['reward'] for r in warm.records[20:]]) - np.mean([r['reward'] for r in cold.records[20:]]))
        assert warm.mind.understanding_rate() > cold.mind.understanding_rate()
    assert np.mean(gains) > .05


def test_warm_runs_are_deterministic():
    teacher = _trained()
    a, b = Simulation(Config(30, 40, 5), teacher), Simulation(Config(30, 40, 5), teacher)
    a.run()
    b.run()
    assert a.mind.events == b.mind.events


def test_pack_roundtrip_and_caps():
    item = _trained()
    loaded, digest = load_pack_file(_pack(item))
    assert len(digest) == 64 and not loaded.verified
    assert loaded.evidence.max() <= K.UNVERIFIED_CAP
    assert np.allclose(loaded.means, item.means, atol=1e-3)


def test_signed_pack_from_trusted_key_is_verified():
    item = _trained()
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    pack = item.to_pack('9.0.0')
    document = {'pack': pack, 'key': public.hex(), 'signature': base64.b64encode(private.sign(pack_bytes(pack))).decode()}
    data = json.dumps(document).encode()
    assert load_pack_file(data, {public})[0].verified
    assert not load_pack_file(data, set())[0].verified
    pack['episodes'] += 1
    with pytest.raises(KnowledgeError):
        load_pack_file(json.dumps({'pack': pack, 'key': public.hex(), 'signature': document['signature']}).encode(), {public})


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(extra=1),
    lambda p: p.update(format='x'),
    lambda p: p.update(acts=list(reversed(ACTS))),
    lambda p: p['means'][0].__setitem__(0, 1.5),
    lambda p: p['means'][0].__setitem__(0, float('nan')),
    lambda p: p['means'].pop(),
    lambda p: p['evidence'][0].__setitem__(0, 0.5),
    lambda p: p['evidence'][0].__setitem__(0, True),
    lambda p: p['words'].__setitem__('0', [['DROP TABLE', .5]]),
    lambda p: p['words'].__setitem__('0', [['bada', .9], ['bada', .1]]),
    lambda p: p['words'].__setitem__('0', [['bada', .9], ['kelo', .9]]),
    lambda p: p.update(episodes=-1),
    lambda p: p.update(app_version='../../x'),
])
def test_malformed_packs_are_rejected(mutate):
    pack = _trained(steps=50).to_pack('9.0.0')
    mutate(pack)
    with pytest.raises(KnowledgeError):
        validate_pack(pack)


def test_oversized_and_unparseable_files_are_rejected():
    with pytest.raises(KnowledgeError):
        load_pack_file(b' ' * (K.MAX_PACK_BYTES + 1))
    with pytest.raises(KnowledgeError):
        load_pack_file(b'[' * 5000)
    with pytest.raises(KnowledgeError):
        load_pack_file(b'{"pack": {}, "evil": 1}')


@given(st.binary(max_size=3000))
@settings(max_examples=200, deadline=None)
def test_fuzz_pack_loader(data):
    with pytest.raises(KnowledgeError):
        load_pack_file(data)


@given(st.recursive(st.none() | st.booleans() | st.floats() | st.integers() | st.text(max_size=8), lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=8), children, max_size=5), max_leaves=30))
@settings(max_examples=200, deadline=None)
def test_fuzz_pack_structure(value):
    with pytest.raises(KnowledgeError):
        load_pack_file(json.dumps({'pack': value}).encode())


def test_robust_merge_resists_poisoned_minority():
    honest = []
    for seed in range(5):
        item = DialogueKnowledge(np.full((CONTEXTS, len(ACTS)), .6 + seed * .01), np.full((CONTEXTS, len(ACTS)), 150.), [{'bada': 1.}, {}, {}, {}], 1)
        honest.append(item)
    poison = [DialogueKnowledge(np.zeros((CONTEXTS, len(ACTS))), np.full((CONTEXTS, len(ACTS)), 200.), [{'zozo': 1.}, {}, {}, {}], 1) for _ in range(3)]
    merged = robust_merge(honest + poison)
    assert np.all(np.abs(merged.means - .62) < .05)
    assert merged.inherited(0) == 'bada'


def test_store_persists_knowledge_and_stays_rollback_compatible(tmp_path):
    path = tmp_path / 'db'
    learned = _trained(steps=60)
    with SecureStore('correct horse battery', path) as store:
        K.record_episode(store, learned)
        K.record_episode(store, learned)
    with SecureStore('correct horse battery', path) as store:
        assert store.conn.execute('PRAGMA user_version').fetchone()[0] == 2
        local, packs = K.stored_contributions(store)
        assert local.episodes == 2 and not packs
        assert K.effective_knowledge(store).episodes == 2


def test_import_limits_and_effective_merge():
    store = VolatileStore()
    with pytest.raises(KnowledgeError):
        K.export_document(store, '9.0.0')
    K.record_episode(store, _trained(steps=60))
    document = K.export_document(store, '9.0.0')
    source, item = K.import_pack(store, json.dumps(document).encode())
    assert source.startswith('pack:') and store.knowledge_count() == 2
    assert K.effective_knowledge(store) is not None
    for i in range(70):
        pack = json.loads(json.dumps(document))
        pack['pack']['episodes'] = i + 10
        try:
            K.import_pack(store, json.dumps(pack).encode())
        except ValueError:
            break
    assert store.knowledge_count() <= 65


def test_window_knowledge_menu_flows(tmp_path, make_window):
    store = VolatileStore()
    window = make_window(store)
    assert window.export_knowledge_file(tmp_path / 'p.json') is None
    K.record_episode(store, _trained(steps=60))
    path = window.export_knowledge_file(tmp_path / 'p')
    assert path.name == 'p.json'
    assert window.import_knowledge_file(path)
    (tmp_path / 'bad.json').write_text('{"pack": 1}')
    assert not window.import_knowledge_file(tmp_path / 'bad.json')
    assert window.clear_packs() == 1


def test_cli_validate_and_merge(tmp_path, capsys):
    import knowledge_tool
    good = tmp_path / 'good.json'
    good.write_bytes(_pack(_trained(steps=60)))
    bad = tmp_path / 'bad.json'
    bad.write_text('{"pack": {"format": "x"}}')
    assert knowledge_tool.main(['validate', str(good)]) == 0
    assert knowledge_tool.main(['validate', str(good), str(bad)]) == 1
    assert knowledge_tool.main(['validate', str(good), '--require-signature']) == 1
    out = tmp_path / 'aggregate.json'
    assert knowledge_tool.main(['merge', str(good), '--output', str(out)]) == 0
    assert load_pack_file(out.read_bytes())[0].strength() > 0
