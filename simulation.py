from dataclasses import dataclass
import operator
import networkx as nx
import numpy as np
import pandas as pd
from artificial_life import ArtificialLife
from observer import ObserverModel
from social import SocialMind
from dialogue import ACTS
from knowledge import INHERIT_PROBABILITY, DialogueKnowledge

MAX_TIES = 24
DISPLAY_LIMIT = 400
EDGE_LIMIT = 1600
MAX_GROUPS = 50
CONFIDENCE = .4
OPINION_NOISE = .004


def _integer(value):
    if isinstance(value, bool):
        raise ValueError('Parâmetros fora dos limites permitidos')
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError('Parâmetros fora dos limites permitidos') from None


@dataclass(frozen=True)
class Config:
    population: int = 120
    steps: int = 100
    seed: int = 42

    def validate(self):
        population, steps, seed = _integer(self.population), _integer(self.steps), _integer(self.seed)
        if not 5 <= population <= 2000 or not 1 <= steps <= 10000 or not 0 <= seed < 2**32:
            raise ValueError('Parâmetros fora dos limites permitidos')
        return self

    def as_dict(self):
        return {'population': int(self.population), 'steps': int(self.steps), 'seed': int(self.seed)}


@dataclass(frozen=True)
class Snapshot:
    step: int
    total_steps: int
    population: int
    positions: np.ndarray
    speeds: np.ndarray
    colors: tuple
    intents: tuple
    watched: np.ndarray
    belief: np.ndarray
    edges: tuple
    conversations: tuple
    metrics: dict
    detail: dict | None
    resources: np.ndarray


class Simulation:
    def __init__(self, config, knowledge=None):
        config.validate()
        self.config = config
        population = int(config.population)
        self.n = population
        self.rng = np.random.default_rng(int(config.seed))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(population))
        self.groups = {i: set() for i in range(4)}
        self.memberships = {i: set() for i in range(population)}
        self.opinions = self.rng.uniform(-1, 1, population)
        self.records = []
        self.alliances = 0
        self.mind = SocialMind(population, int(config.seed))
        self.life = ArtificialLife(population, int(config.seed))
        self.observer = ObserverModel(population, int(config.seed))
        self.last_conversations = []
        self.tick = 0
        for agent in range(population):
            self.join(agent, int(self.rng.integers(4)))
        self.inherited = False
        if knowledge is not None:
            self._inherit(knowledge)

    def _inherit(self, knowledge):
        if not isinstance(knowledge, DialogueKnowledge):
            raise ValueError('Conhecimento inválido')
        alpha0, beta0 = knowledge.prior()
        self.mind.strategy.seed_prior(alpha0, beta0)
        rng = np.random.default_rng(int(self.config.seed) ^ 0x1A2B)
        for topic in range(len(knowledge.words)):
            word = knowledge.inherited(topic)
            heirs = np.flatnonzero(rng.random(self.n) < INHERIT_PROBABILITY)
            if word:
                for agent in heirs.tolist():
                    self.mind.lexicon.inherit(agent, topic, word)
        self.inherited = True

    def knowledge(self):
        return DialogueKnowledge.from_simulation(self)

    def join(self, agent, group):
        self.groups.setdefault(group, set()).add(agent)
        self.memberships[agent].add(group)

    def leave(self, agent, group):
        self.groups[group].discard(agent)
        self.memberships[agent].discard(group)

    def set_observation(self, observation):
        self.observer.set_observation(observation)

    def _remove_edge(self, a, b):
        if self.graph.has_edge(a, b):
            if self.graph.edges[a, b].get('alliance'):
                self.alliances -= 1
            self.graph.remove_edge(a, b)

    def _make_room(self, agent, keep):
        if self.graph.degree(agent) < MAX_TIES:
            return
        neighbors = [b for b in self.graph.adj[agent] if b != keep]
        if not neighbors:
            return
        weakest = max(neighbors, key=lambda b: (abs(float(self.opinions[agent] - self.opinions[b])), b))
        self._remove_edge(agent, weakest)

    def _tie(self, a, b, alliance):
        if a == b:
            return
        existing = self.graph.has_edge(a, b)
        if not existing:
            self._make_room(a, b)
            self._make_room(b, a)
        elif self.graph.edges[a, b].get('alliance'):
            self.alliances -= 1
        self.graph.add_edge(a, b, alliance=bool(alliance))
        if alliance:
            self.alliances += 1

    def step(self, t):
        n = self.n
        self.tick = int(t)
        first = self.rng.integers(0, n, size=n)
        second = (first + self.rng.integers(1, n, size=n)) % n
        rolls = self.rng.random((n, 6))
        positions = self.life.positions
        proximity = np.exp(-np.linalg.norm(positions[first] - positions[second], axis=1) / 150.).tolist()
        drift = self.rng.normal(0., OPINION_NOISE, n).tolist()
        opinions = self.opinions
        rolls_list = rolls.tolist()
        for k in range(n):
            a, b = int(first[k]), int(second[k])
            roll = rolls_list[k]
            shared = bool(self.memberships[a] & self.memberships[b])
            old_a, old_b = float(opinions[a]), float(opinions[b])
            similarity = 1 - abs(old_a - old_b) / 2
            probability = min(.95, .02 + .35 * similarity + .25 * shared + .30 * proximity[k])
            if roll[0] < probability:
                self._tie(a, b, shared and similarity > .7)
                if abs(old_a - old_b) < CONFIDENCE:
                    opinions[a] = min(1., max(-1., old_a + .08 * (old_b - old_a)))
                    opinions[b] = min(1., max(-1., old_b + .08 * (old_a - old_b)))
            opinions[a] = min(1., max(-1., float(opinions[a]) + drift[k]))
            if roll[1] < .025:
                if self.memberships[a] and roll[2] < .5:
                    options = sorted(self.memberships[a])
                    self.leave(a, options[int(roll[3] * len(options))])
                else:
                    options = sorted(self.groups)
                    self.join(a, options[int(roll[3] * len(options))])
            if roll[4] < .002 and len(self.groups) < MAX_GROUPS:
                self.join(a, max(self.groups) + 1)
            if roll[5] < .01:
                self._remove_edge(a, b)
        self.observer.update(self.life.positions)
        self.life.step(t, self.mind, self.observer)
        self.last_conversations = self.mind.step(self, t)
        for a, b, reward in self.mind.links:
            if reward > .3:
                shared = bool(self.memberships[a] & self.memberships[b])
                similarity = 1 - abs(float(self.opinions[a] - self.opinions[b])) / 2
                self._tie(a, b, shared and similarity > .7)
        active = sum(1 for members in self.groups.values() if members)
        dispersion = float(np.mean(np.linalg.norm(self.life.positions - self.life.positions.mean(axis=0), axis=1)))
        self.records.append({'step': int(t), 'edges': self.graph.number_of_edges(), 'groups': active, 'alliances': int(self.alliances), 'polarization': float(np.std(self.opinions)), 'components': nx.number_connected_components(self.graph), 'dispersion': round(dispersion, 3), 'observed_agents': int(self.observer.watched.sum()), 'mean_belief': round(float(self.observer.belief.mean()), 4), 'lexicon_coherence': round(float(self.mind.coherence), 4), 'understanding': round(float(self.mind.understanding_rate()), 4), 'reward': round(float(self.mind.recent_reward()), 4)})

    def run(self, progress=None):
        for t in range(1, self.config.steps + 1):
            self.step(t)
            if progress and (t % max(1, self.config.steps // 100) == 0 or t == self.config.steps):
                progress(t * 100 // self.config.steps)
        return pd.DataFrame.from_records(self.records)

    def frame(self):
        return pd.DataFrame.from_records(self.records)

    def group_of(self, agent):
        groups = self.memberships[agent]
        return min(groups) if groups else -1

    def detail(self, agent):
        if not isinstance(agent, int) or not 0 <= agent < self.n:
            return None
        return {'agente': agent, 'opinião': round(float(self.opinions[agent]), 4), 'grupos': sorted(self.memberships[agent]), 'conexões': int(self.graph.degree(agent)), 'mente': self.mind.describe(agent), 'vida': self.life.snapshot(agent), 'observação': self.observer.describe(agent)}

    def snapshot(self, selected=None):
        shown = min(self.n, DISPLAY_LIMIT)
        edges = []
        for a, b, alliance in self.graph.edges(data='alliance'):
            if a < shown and b < shown:
                edges.append((int(a), int(b), bool(alliance)))
                if len(edges) >= EDGE_LIMIT:
                    break
        metrics = dict(self.records[-1]) if self.records else {'step': 0}
        metrics['autonomy'] = round(float(np.mean(self.mind.autonomy)), 4)
        metrics['interactions'] = int(np.sum(self.mind.experience) // 2)
        return Snapshot(step=self.tick, total_steps=int(self.config.steps), population=self.n, positions=self.life.positions[:shown].copy(), speeds=np.linalg.norm(self.life.velocities[:shown], axis=1), colors=tuple(self.group_of(a) for a in range(shown)), intents=tuple(self.life.intent[:shown]), watched=self.observer.watched[:shown].copy(), belief=self.observer.belief[:shown].copy(), edges=tuple(edges), conversations=tuple(self.last_conversations[:8]), metrics=metrics, detail=self.detail(selected), resources=self.life.resources.copy())

    def episode(self):
        final = dict(self.records[-1]) if self.records else {}
        return {'config': self.config.as_dict(), 'final': final, 'agent_memories': self.life.export_memories(16), 'dialogue': {'acts': {name: int(count) for name, count in zip(ACTS, self.mind.act_counts)}, 'coherence': float(self.mind.coherence)}, 'inherited': bool(self.inherited)}
