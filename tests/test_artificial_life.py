import numpy as np
from hypothesis import given, settings, strategies as st
from app import Config, Simulation
from artificial_life import ArtificialLife, WORLD
from social import SocialMind
from spatial import SpatialGrid, pairs_within


def test_reproducible_navigation():
    a, b = ArtificialLife(24, 123), ArtificialLife(24, 123)
    m, n = SocialMind(24, 123), SocialMind(24, 123)
    for t in range(1, 101):
        a.step(t, m)
        b.step(t, n)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.supplies, b.supplies)


def test_memory_is_bounded_and_observation_local():
    life = ArtificialLife(10, 42)
    for i in range(100):
        life.remember(0, i, 'test', 1, .5)
    assert len(life.memories[0]) == 48
    assert 0 not in life.observe(0)['agents']
    assert all(0 <= i < 10 for i in life.observe(0)['agents'])
    assert all(len(v) <= 16 for v in life.export_memories(16).values())


def test_resource_seeking_and_bounds():
    life = ArtificialLife(12, 5)
    mind = SocialMind(12, 5)
    life.hunger[:] = .9
    for t in range(1, 350):
        life.step(t, mind)
    assert np.isfinite(life.positions).all()
    assert (np.abs(life.positions) <= WORLD).all()
    assert (life.supplies >= 0).all()
    assert (life.hunger >= 0).all() and (life.hunger <= 1).all()
    assert any(life.visits.flatten())


def _nearest(points):
    d = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def test_agents_roam_instead_of_clumping():
    simulation = Simulation(Config(120, 400, 7))
    cells = set()
    for t in range(1, 401):
        simulation.step(t)
        cells.update(map(tuple, np.floor(simulation.life.positions / 90).astype(int).tolist()))
    nearest = _nearest(simulation.life.positions)
    assert (nearest < 20).mean() < .75
    assert np.median(nearest) > 12
    assert len(cells) >= 60
    assert simulation.records[-1]['dispersion'] > 180


def test_resources_do_not_collapse_under_load():
    simulation = Simulation(Config(200, 500, 2))
    simulation.run()
    assert simulation.life.hunger.mean() < .75
    assert (simulation.life.supplies > 0).any()


@given(st.lists(st.tuples(st.floats(-450, 450), st.floats(-450, 450)), min_size=0, max_size=80), st.floats(1, 150))
@settings(max_examples=80, deadline=None)
def test_spatial_index_matches_brute_force(points, radius):
    positions = np.array(points, dtype=float).reshape(-1, 2)
    rows, cols, dists = pairs_within(positions, radius)
    expected = {(i, j) for i in range(len(positions)) for j in range(i + 1, len(positions)) if np.linalg.norm(positions[i] - positions[j]) <= radius}
    assert {tuple(sorted((int(a), int(b)))) for a, b in zip(rows, cols)} == expected
    assert (dists <= radius + 1e-9).all()
    if len(positions):
        grid = SpatialGrid(positions, 120.)
        ids, _ = grid.query(positions[0], radius)
        brute = {i for i in range(len(positions)) if np.linalg.norm(positions[i] - positions[0]) <= radius}
        assert set(ids.tolist()) == brute
