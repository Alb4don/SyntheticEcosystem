import numpy as np
from app import Config, Simulation
from dialogue import ACTS, ACT_INDEX, Lexicon, StrategyLearner, context
from social import TOPICS


def test_reproducible_social_dialogue():
    a = Simulation(Config(35, 50, 29))
    b = Simulation(Config(35, 50, 29))
    for tick in range(1, 51):
        a.step(tick)
        b.step(tick)
    assert a.mind.events == b.mind.events
    assert np.array_equal(a.mind.knowledge, b.mind.knowledge)


def test_social_invariants():
    simulation = Simulation(Config(55, 200, 5))
    for tick in range(1, 201):
        simulation.step(tick)
    mind = simulation.mind
    for array in (mind.energy, mind.social_need, mind.knowledge, mind.trust, mind.autonomy, mind.awareness):
        assert np.isfinite(array).all()
        assert ((array >= 0) & (array <= 1)).all()
    assert all(0 <= e.speaker < 55 and 0 <= e.listener < 55 and e.speaker != e.listener for e in mind.events)
    assert len(mind.events) <= 1500
    assert max(mind.autonomy) > .15 and int(mind.experience.sum()) > 0
    for event in mind.events:
        assert event.act in ACTS and event.topic in TOPICS
        assert event.text and event.reply
        assert '{' not in event.text and '}' not in event.reply


def test_dialogue_improves_while_running():
    simulation = Simulation(Config(40, 600, 3))
    simulation.run()
    understanding = [r['understanding'] for r in simulation.records]
    assert understanding[-1] > understanding[100]
    assert not np.allclose(simulation.mind.strategy.alpha, 1.)
    tier_two = ('meu domínio está em', 'nossas visões estão próximas', 'Bom te ver de novo', 'cerca de', 'Você parece entender', 'vemos isso de modo parecido', 'nossas posições divergem', 'tenho pensado em')
    assert any(marker in e.text for e in simulation.mind.events for marker in tier_two)


def test_strategy_learner_converges_to_rewarded_act():
    learner = StrategyLearner(1, np.random.default_rng(1))
    allowed = np.ones(len(ACTS), dtype=bool)
    for _ in range(300):
        act = learner.choose(0, 3, allowed)
        learner.update(0, 3, act, 1. if act == 2 else 0.)
    picks = [learner.choose(0, 3, allowed) for _ in range(200)]
    assert picks.count(2) / len(picks) > .85
    assert learner.dominant(0) == ACTS[2]


def test_strategy_respects_mask():
    learner = StrategyLearner(1, np.random.default_rng(2))
    allowed = np.zeros(len(ACTS), dtype=bool)
    allowed[ACT_INDEX['saudar']] = True
    assert all(learner.choose(0, 0, allowed) == ACT_INDEX['saudar'] for _ in range(50))


def test_context_encoding_is_unique():
    values = {context(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)}
    assert values == set(range(16))


def test_naming_game_adoption_and_alignment():
    lexicon = Lexicon(2, 1, np.random.default_rng(4))
    word, understood, heard = lexicon.interact(0, 1, 0)
    assert not understood and heard is None
    assert lexicon.top(1, 0) == word
    again, understood, heard = lexicon.interact(0, 1, 0)
    assert again == word and understood and heard == word
    assert lexicon.coherence() == 1.


def test_lexicon_capacity_is_bounded():
    lexicon = Lexicon(10, 1, np.random.default_rng(5), capacity=3)
    for speaker in range(1, 10):
        lexicon.interact(speaker, 0, 0)
    assert len(lexicon.entries[0][0]) <= 3


def test_resource_tip_redirects_hungry_listener():
    simulation = Simulation(Config(10, 5, 1))
    life = simulation.life
    life.remember(0, 1, 'resource', 2, 1.)
    life.hunger[1] = .9
    text, reply, reward = simulation.mind._realize('indicar_recurso', 0, 1, 0, 'bada', .0, .9, simulation, 3)
    assert 'recurso 2' in text
    assert reward >= .8 and life.plans[1][0] == ('move', 2)
    assert life.intent[1] == 'seguir indicação'


def test_affinity_table_is_bounded():
    simulation = Simulation(Config(20, 2, 1))
    mind = simulation.mind
    mind.affinity = {(i, -i - 1): .5 + (i % 7) / 100 for i in range(70000)}
    simulation.step(1)
    assert len(mind.affinity) <= 60000
