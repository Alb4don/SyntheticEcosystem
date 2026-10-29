import time
import pytest
from hypothesis import given, settings, strategies as st
from app import Config, Simulation
from simulation import MAX_TIES


def test_determinism():
    c = Config(20, 12, 7)
    assert Simulation(c).run().equals(Simulation(c).run())


def test_groups():
    s = Simulation(Config(20, 5, 3))
    s.run()
    assert s.graph.number_of_nodes() == 20
    assert all(a in s.groups[g] for a, gs in s.memberships.items() for g in gs)


@pytest.mark.parametrize('args', [(1, 5, 4), (5, 0, 1), (5, 5, -1), (2001, 5, 1), (5, 10001, 1), (5, 5, 2**32), (True, 5, 1), (5.0, 5, 1), ('5', 5, 1), (None, 5, 1)])
def test_invalid_config(args):
    with pytest.raises(ValueError):
        Config(*args).validate()


@given(st.integers(min_value=5, max_value=60), st.integers(min_value=0, max_value=100000))
@settings(max_examples=35, deadline=None)
def test_properties(n, seed):
    s = Simulation(Config(n, 3, seed))
    frame = s.run()
    assert len(frame) == 3
    assert frame['polarization'].between(0, 1).all()
    assert frame['edges'].between(0, n * (n - 1) // 2).all()
    assert all(a in s.groups[g] for a, gs in s.memberships.items() for g in gs)


def test_degree_is_bounded_and_alliance_counter_is_exact():
    s = Simulation(Config(60, 150, 9))
    s.run()
    assert max(d for _, d in s.graph.degree()) <= MAX_TIES
    assert s.alliances == sum(1 for _, _, a in s.graph.edges(data='alliance') if a)


def test_opinions_stay_diverse_and_bounded():
    s = Simulation(Config(120, 300, 4))
    frame = s.run()
    assert (abs(s.opinions) <= 1).all()
    assert frame['polarization'].iloc[-1] > .05


def test_snapshot_is_detached_copy():
    s = Simulation(Config(30, 5, 2))
    s.step(1)
    snap = s.snapshot(selected=4)
    before = snap.positions.copy()
    s.step(2)
    assert (snap.positions == before).all()
    assert snap.detail['agente'] == 4
    assert s.snapshot(selected=999).detail is None
    assert len(snap.edges) <= 1600


def test_episode_is_json_serializable():
    import json
    s = Simulation(Config(25, 10, 1))
    s.run()
    data = json.dumps(s.episode(), allow_nan=False)
    assert '"config"' in data


def test_performance_smoke():
    started = time.perf_counter()
    Simulation(Config(100, 20, 4)).run()
    assert time.perf_counter() - started < 20


def test_large_population_step_latency():
    s = Simulation(Config(2000, 12, 4))
    s.step(1)
    started = time.perf_counter()
    for t in range(2, 12):
        s.step(t)
    assert (time.perf_counter() - started) / 10 < 1.5
