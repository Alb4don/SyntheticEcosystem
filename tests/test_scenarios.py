import numpy as np
import pytest
from helpers import drop, keypair
from observer import Observation
from simulation import Config, Simulation
from updates import UpdateManager


class Scenario:
    def __init__(self):
        self.simulation = None

    def given_society(self, population, seed):
        self.simulation = Simulation(Config(population, 2000, seed))
        return self

    def when_user_watches(self, agent, cycles):
        s = self.simulation
        for _ in range(cycles):
            x, y = s.life.positions[agent]
            s.set_observation(Observation.sanitize(True, (x - 150, y - 150, x + 150, y + 150), (x, y), agent, s.n))
            s.step(s.tick + 1)
        return self

    def when_nobody_watches(self, cycles):
        s = self.simulation
        s.set_observation(Observation())
        self.watched_cycles = 0
        for _ in range(cycles):
            s.step(s.tick + 1)
            self.watched_cycles += int(s.observer.watched.sum())
        return self

    def alerts(self, since=0):
        return [e for e in self.simulation.mind.events if e.act == 'alertar_observador' and e.step > since]


def test_scenario_watched_agent_notices_observer_and_talks_about_it():
    scenario = Scenario().given_society(40, 11)
    scenario.when_user_watches(0, 250)
    s = scenario.simulation
    assert s.observer.watched[0]
    assert s.observer.belief[0] > .65
    assert scenario.alerts()
    assert all(s.observer.belief[e.speaker] >= 0 for e in scenario.alerts())


def test_scenario_unwatched_society_does_not_hallucinate_observer():
    scenario = Scenario().given_society(80, 5)
    scenario.when_nobody_watches(300)
    assert scenario.watched_cycles / (80 * 300) < .01
    assert len(scenario.alerts()) <= 2


def test_scenario_observation_ending_calms_society():
    scenario = Scenario().given_society(40, 2)
    scenario.when_user_watches(3, 150)
    peak = scenario.simulation.observer.belief.mean()
    scenario.when_nobody_watches(200)
    assert scenario.simulation.observer.belief.mean() < peak
    assert scenario.simulation.observer.watched.sum() <= 1


def test_scenario_agents_spread_out_over_time():
    scenario = Scenario().given_society(120, 8)
    start = np.linalg.norm(scenario.simulation.life.positions, axis=1).mean()
    scenario.when_nobody_watches(300)
    positions = scenario.simulation.life.positions
    assert np.linalg.norm(positions - positions.mean(axis=0), axis=1).mean() > .75 * start
    assert scenario.simulation.life.travelled.mean() > 150


@pytest.mark.parametrize('tamper', ['signature', 'package'])
def test_scenario_tampered_update_never_runs(tmp_path, tamper):
    private, public = keypair()
    manager = UpdateManager(tmp_path, '9.0.0', public, clock=lambda: 1e12)
    signer = keypair()[0] if tamper == 'signature' else private
    drop(manager.inbox, signer, '9.1.0')
    if tamper == 'package':
        (manager.inbox / 'update.zip').write_bytes((manager.inbox / 'update.zip').read_bytes() + b'x')
    assert manager.scan_inbox()['status'] == 'rejected'
    assert manager.select_boot() is None
