import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from observer import LIMIT, Observation, ObserverModel


def _positions(n, seed=0):
    return np.random.default_rng(seed).uniform(-400, 400, (n, 2))


@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_no_observer_means_almost_no_false_positives(seed):
    model = ObserverModel(200, seed)
    positions = _positions(200, seed)
    watched = 0
    for _ in range(300):
        watched += int(model.update(positions).sum())
    assert watched / (200 * 300) < .01


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_selected_agent_detects_observer_even_with_low_sensitivity(seed):
    model = ObserverModel(50, seed)
    agent = int(np.argmin(model.sensitivity))
    positions = _positions(50, seed)
    model.set_observation(Observation.sanitize(True, (-450, -450, 450, 450), tuple(positions[agent]), agent, 50))
    history = [bool(model.update(positions)[agent]) for _ in range(220)]
    assert sum(history[80:]) / len(history[80:]) > .8
    assert model.duration[agent] > 0 or not history[-1]


def test_cursor_proximity_raises_belief_locally():
    model = ObserverModel(300, 7)
    positions = _positions(300, 7)
    anchor = positions[0]
    model.set_observation(Observation.sanitize(True, (-450, -450, 450, 450), tuple(anchor), None, 300))
    for _ in range(120):
        model.update(positions)
    distance = np.linalg.norm(positions - anchor, axis=1)
    near, far = distance < 45, distance > 250
    assert near.sum() >= 1 and far.sum() >= 1
    assert model.belief[near].mean() > model.belief[far].mean() + .2


def test_hysteresis_limits_flapping():
    model = ObserverModel(100, 9)
    positions = _positions(100, 9)
    model.set_observation(Observation.sanitize(True, (-450, -450, 450, 450), (0., 0.), None, 100))
    previous = model.watched.copy()
    flips = 0
    for _ in range(300):
        current = model.update(positions).copy()
        flips += int((current != previous).sum())
        previous = current
    assert flips / 100 < 6


def test_shy_agents_move_away_from_cursor():
    model = ObserverModel(40, 3)
    positions = _positions(40, 3)
    model.set_observation(Observation.sanitize(True, (-450, -450, 450, 450), (0., 0.), None, 40))
    for _ in range(150):
        model.update(positions)
    flee = model.avoidance(positions)
    shy = np.flatnonzero(model.shy())
    for agent in shy:
        assert np.dot(flee[agent], positions[agent]) >= 0
    assert not flee[~model.shy()].any()


def test_observation_turning_off_decays_belief():
    model = ObserverModel(20, 4)
    positions = _positions(20, 4)
    model.set_observation(Observation.sanitize(True, (-450, -450, 450, 450), tuple(positions[0]), 0, 20))
    for _ in range(100):
        model.update(positions)
    model.set_observation(Observation())
    for _ in range(150):
        model.update(positions)
    assert model.belief[0] < .2 and not model.watched[0]


junk = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=5), st.lists(st.floats(allow_nan=True, allow_infinity=True), max_size=6), st.tuples(st.floats(allow_nan=True), st.floats(allow_infinity=True)))


@given(junk, junk, junk, junk, st.integers(min_value=0, max_value=3000))
@settings(max_examples=300, deadline=None)
def test_observation_sanitizer_fuzz(active, view, cursor, selected, population):
    obs = Observation.sanitize(active, view, cursor, selected, population)
    assert isinstance(obs, Observation)
    assert all(math.isfinite(v) and abs(v) <= LIMIT for v in obs.view)
    assert obs.view[0] <= obs.view[2] and obs.view[1] <= obs.view[3]
    if obs.cursor is not None:
        assert all(math.isfinite(v) and abs(v) <= LIMIT for v in obs.cursor)
    assert obs.selected is None or (isinstance(obs.selected, int) and 0 <= obs.selected < population)
    model = ObserverModel(10, 1)
    model.set_observation(obs)
    model.update(_positions(10))
    assert np.isfinite(model.posterior).all() and np.allclose(model.posterior.sum(axis=1), 1.)
