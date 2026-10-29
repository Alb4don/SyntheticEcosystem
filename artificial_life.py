from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from spatial import SpatialGrid, pairs_within

WORLD = 450.
GRID_CELL = 120.
SEPARATION = 34.
CROWD_RADIUS = 60.
CROWD_LIMIT = 6
GOLDEN = 2.399963229728653
MAX_SPEED = 3.0
HUNGER_RATE = .0018
ARRIVAL = {'socialize': 30., 'collect': 40., 'move': 40.}
COMPASS = ('leste', 'nordeste', 'norte', 'noroeste', 'oeste', 'sudoeste', 'sul', 'sudeste')


@dataclass(frozen=True)
class MemoryEvent:
    tick: int
    kind: str
    partner: int
    reward: float


class ArtificialLife:
    def __init__(self, population, seed):
        self.n = int(population)
        n = self.n
        self.rng = np.random.default_rng(seed ^ 0xC0FFEE)
        rotation = float(self.rng.uniform(0, 2 * math.pi))
        index = np.arange(n)
        angles = index * GOLDEN + rotation
        radii = 400. * np.sqrt((index + .5) / max(n, 1))
        self.positions = np.column_stack((np.cos(angles) * radii, np.sin(angles) * radii))
        self.home = self.positions.copy()
        self.velocities = np.zeros((n, 2), dtype=float)
        self.resources = np.array([[-300., -250.], [300., -250.], [-300., 250.], [300., 250.], [0., 0.], [0., -330.], [0., 330.], [-360., 0.], [360., 0.]])
        self.capacity = 100. + .5 * n
        self.regeneration = max(.045, .016 * n * 5 / len(self.resources))
        self.supplies = np.full(len(self.resources), self.capacity, dtype=float)
        self.hunger = self.rng.uniform(.15, .55, n)
        self.fatigue = self.rng.uniform(.05, .35, n)
        self.independence = self.rng.uniform(.25, 1., n)
        self.perception = np.full(n, 115.)
        self.target = np.clip(self.positions + self.rng.normal(0, 60, (n, 2)), -420, 420)
        self.intent = ['explorar'] * n
        self.plans = [deque() for _ in range(n)]
        self.memories = [deque(maxlen=48) for _ in range(n)]
        self.visits = np.zeros((n, len(self.resources)), dtype=int)
        self.last_planned = np.full(n, -100, dtype=int)
        self.social_cooldown = np.full(n, -100, dtype=int)
        self.density = np.zeros(n, dtype=int)
        self.crowd_limit = CROWD_LIMIT
        self.travelled = np.zeros(n, dtype=float)
        self.grid = SpatialGrid(self.positions, GRID_CELL)

    def observe(self, agent):
        ids, _ = self.grid.query(self.positions[agent], self.perception[agent])
        nearby = [int(i) for i in ids if i != agent]
        resource_distances = np.linalg.norm(self.resources - self.positions[agent], axis=1)
        visible_resources = np.flatnonzero(resource_distances <= self.perception[agent]).tolist()
        return {'agents': nearby, 'resources': visible_resources}

    def remember(self, agent, tick, kind, partner=-1, reward=0.):
        self.memories[agent].append(MemoryEvent(int(tick), str(kind), int(partner), float(reward)))

    def last_resource(self, agent):
        for event in reversed(self.memories[agent]):
            if event.kind == 'resource' and 0 <= event.partner < len(self.resources):
                return event.partner
        return None

    def bearing(self, agent, resource):
        delta = self.resources[resource] - self.positions[agent]
        distance = float(np.linalg.norm(delta))
        angle = math.atan2(-float(delta[1]), float(delta[0])) % (2 * math.pi)
        return distance, COMPASS[int(round(angle / (math.pi / 4))) % 8]

    def direct_to(self, agent, resource, tick):
        if not 0 <= resource < len(self.resources) or self.supplies[resource] <= 1.:
            return False
        self.plans[agent].clear()
        self.plans[agent].extend([('move', int(resource)), ('collect', int(resource))])
        self.last_planned[agent] = tick
        self.intent[agent] = 'seguir indicação'
        return True

    def _levy_target(self, agent):
        length = min(520., 45. * float(self.rng.random()) ** (-1 / 1.6))
        angle = float(self.rng.uniform(0, 2 * math.pi))
        base = self.positions[agent]
        if self.rng.random() < (1 - self.independence[agent]) * .35:
            base = self.home[agent]
        target = base + length * np.array([math.cos(angle), math.sin(angle)])
        return np.clip(target, -420, 420)

    def plan(self, agent, tick, mind):
        if self.plans[agent] and tick - self.last_planned[agent] < 15:
            return
        self.plans[agent].clear()
        self.last_planned[agent] = tick
        if self.hunger[agent] > .55 and np.any(self.supplies > 1.):
            scores = np.linalg.norm(self.resources - self.positions[agent], axis=1) + 12 * self.visits[agent]
            scores = scores + np.where(self.supplies > 1., 0., 10000.)
            index = int(np.argmin(scores))
            self.plans[agent].extend([('move', index), ('collect', index)])
            self.intent[agent] = 'buscar recursos'
            return
        if self.hunger[agent] > .55:
            self.intent[agent] = 'procurar alimento'
            self.target[agent] = self._levy_target(agent)
            self.plans[agent].append(('explore', -1))
            return
        if self.fatigue[agent] > .72:
            self.plans[agent].append(('rest', -1))
            self.intent[agent] = 'descansar'
            return
        sociability = float(mind.social_need[agent]) * (1.25 - .6 * float(self.independence[agent]))
        if tick - self.social_cooldown[agent] > 20 and sociability > .42 and self.density[agent] <= self.crowd_limit:
            visible = self.observe(agent)['agents']
            if visible:
                partner = max(visible, key=lambda b: (mind.affinity.get((agent, b), .5), -b))
                self.plans[agent].append(('socialize', int(partner)))
                self.intent[agent] = 'procurar conversa'
                return
        self.intent[agent] = 'buscar espaço' if self.density[agent] > self.crowd_limit else 'explorar'
        self.target[agent] = self._levy_target(agent)
        self.plans[agent].append(('explore', -1))

    def _separation(self):
        rows, cols, dists = pairs_within(self.positions, CROWD_RADIUS)
        self.density = (np.bincount(rows, minlength=self.n) + np.bincount(cols, minlength=self.n)).astype(int)
        push = np.zeros((self.n, 2), dtype=float)
        close = dists < SEPARATION
        if not close.any():
            return push
        i, j, d = rows[close], cols[close], dists[close]
        diff = self.positions[i] - self.positions[j]
        coincident = d < 1e-6
        if coincident.any():
            angles = (i[coincident] * GOLDEN) % (2 * math.pi)
            diff[coincident] = np.column_stack((np.cos(angles), np.sin(angles)))
            d = np.where(coincident, 1., d)
        weight = ((SEPARATION - d) / SEPARATION) / d
        contribution = diff * weight[:, None]
        np.add.at(push, i, contribution)
        np.add.at(push, j, -contribution)
        return push

    def step(self, tick, mind, observer=None):
        self.supplies = np.minimum(self.capacity, self.supplies + self.regeneration)
        self.hunger = np.clip(self.hunger + HUNGER_RATE, 0, 1)
        self.fatigue = np.clip(self.fatigue + .001, 0, 1)
        n = self.n
        desired = np.zeros((n, 2), dtype=float)
        active = np.zeros(n, dtype=bool)
        for a in range(n):
            self.plan(a, tick, mind)
            if not self.plans[a]:
                continue
            action, value = self.plans[a][0]
            if action in ('move', 'collect'):
                destination = self.resources[value]
            elif action == 'socialize':
                if not 0 <= value < n or value == a or math.hypot(*(self.positions[value] - self.positions[a]).tolist()) > self.perception[a] * 1.6:
                    self.plans[a].popleft()
                    continue
                destination = self.positions[value]
            elif action == 'rest':
                self.velocities[a] *= .6
                self.fatigue[a] = max(0., self.fatigue[a] - .024)
                if self.fatigue[a] < .3:
                    self.plans[a].popleft()
                continue
            else:
                destination = self.target[a]
            direction = destination - self.positions[a]
            distance = math.hypot(float(direction[0]), float(direction[1]))
            if distance < ARRIVAL.get(action, 12.):
                if action == 'collect' and self.supplies[value] < 1.:
                    self.plans[a].clear()
                    self.last_planned[a] = -100
                    continue
                if action == 'collect':
                    amount = min(12., float(self.supplies[value]))
                    self.supplies[value] -= amount
                    self.hunger[a] = max(0., self.hunger[a] - amount / 20.)
                    self.visits[a, value] += 1
                    self.remember(a, tick, 'resource', value, amount / 12.)
                if action == 'socialize':
                    self.social_cooldown[a] = tick
                    self.plans[a].popleft()
                    self.target[a] = self._levy_target(a)
                    self.plans[a].append(('explore', -1))
                    self.intent[a] = 'explorar'
                elif action != 'explore':
                    self.plans[a].popleft()
                else:
                    self.target[a] = self._levy_target(a)
                continue
            speed = 1.8 * (1 - .35 * self.fatigue[a]) * (.85 + .3 * self.independence[a])
            desired[a] = direction / max(distance, 1e-9) * speed
            active[a] = True
        push = self._separation()
        flee = observer.avoidance(self.positions) if observer is not None else np.zeros((n, 2))
        force = desired + 1.6 * push + flee
        moving = active | np.any(push != 0, axis=1) | np.any(flee != 0, axis=1)
        self.velocities[moving] = .65 * self.velocities[moving] + .35 * force[moving]
        speeds = np.linalg.norm(self.velocities, axis=1)
        too_fast = speeds > MAX_SPEED
        self.velocities[too_fast] *= (MAX_SPEED / speeds[too_fast])[:, None]
        proposed = self.positions + self.velocities
        outside = np.abs(proposed) > WORLD
        self.velocities[outside] *= -.5
        proposed = np.clip(proposed, -WORLD, WORLD)
        self.travelled += np.linalg.norm(proposed - self.positions, axis=1)
        self.fatigue[active] = np.minimum(1., self.fatigue[active] + .0003)
        self.positions = proposed
        self.grid = SpatialGrid(self.positions, GRID_CELL)

    def snapshot(self, agent):
        return {'intenção': self.intent[agent], 'fome': round(float(self.hunger[agent]), 3), 'fadiga': round(float(self.fatigue[agent]), 3), 'independência': round(float(self.independence[agent]), 3), 'vizinhos_próximos': int(self.density[agent]), 'percepção': self.observe(agent), 'memórias_recentes': [vars(m) for m in list(self.memories[agent])[-5:]]}

    def export_memories(self, limit=16):
        limit = max(1, min(48, int(limit)))
        return {str(a): [vars(event) for event in list(events)[-limit:]] for a, events in enumerate(self.memories) if events}
