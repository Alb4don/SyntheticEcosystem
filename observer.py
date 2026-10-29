from dataclasses import dataclass
import math
import numpy as np

LIMIT = 5000.
WORLD_AREA = 900. * 900.
ON_LEVEL = .65
OFF_LEVEL = .35
STATES = ('ignorado', 'notado de relance', 'em foco')
PHRASES = ('sem ser notado', 'sendo notado de relance', 'sendo observado de perto')
MEANS = np.array([0., .35, .95])
SPREAD = .12
TRANSITION = np.array([[.995, .004, .001], [.04, .92, .04], [.01, .02, .97]])
PRIOR = np.array([.90, .08, .02])
FLOOR = 1e-6
GAZE_RADIUS = 60.
DWELL_SCALE = 40.


def _finite(value, default=0.):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return max(-LIMIT, min(LIMIT, number))


@dataclass(frozen=True)
class Observation:
    active: bool = False
    view: tuple = (0., 0., 0., 0.)
    cursor: tuple | None = None
    selected: int | None = None

    @classmethod
    def sanitize(cls, active, view, cursor, selected, population):
        if not isinstance(active, bool) or not active:
            return cls()
        try:
            x0, y0, x1, y1 = (_finite(v) for v in view)
        except (TypeError, ValueError):
            return cls()
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        point = None
        if cursor is not None:
            try:
                cx, cy = cursor
                point = (_finite(cx), _finite(cy))
            except (TypeError, ValueError):
                point = None
        chosen = None
        if isinstance(selected, (int, np.integer)) and not isinstance(selected, bool) and 0 <= int(selected) < int(population):
            chosen = int(selected)
        return cls(True, (x0, y0, x1, y1), point, chosen)


class ObserverModel:
    def __init__(self, population, seed):
        self.n = int(population)
        self.rng = np.random.default_rng(seed ^ 0x0B5E)
        self.sensitivity = self.rng.uniform(.6, 1.8, self.n)
        self.boldness = self.rng.uniform(0., 1., self.n)
        self.posterior = np.tile(PRIOR, (self.n, 1))
        self.watched = np.zeros(self.n, dtype=bool)
        self.duration = np.zeros(self.n, dtype=int)
        self.attention = np.zeros(self.n, dtype=float)
        self.arousal = np.zeros(self.n, dtype=float)
        self.habituation = np.zeros(self.n, dtype=float)
        self.surprise = np.zeros(self.n, dtype=float)
        self.observation = Observation()
        self.cursor_vectors = np.zeros((self.n, 2), dtype=float)
        self.previous_cursor = None
        self.cursor_speed = 0.
        self.detections = 0
        self.social_reports = 0

    @property
    def belief(self):
        return self.posterior[:, 2] + .5 * self.posterior[:, 1]

    @property
    def confidence(self):
        p = np.clip(self.posterior, FLOOR, 1.)
        return 1. - (-(p * np.log(p)).sum(axis=1) / math.log(3))

    def state(self, agent):
        return STATES[int(np.argmax(self.posterior[agent]))]

    def phrase(self, agent):
        return PHRASES[int(np.argmax(self.posterior[agent]))]

    def set_observation(self, observation):
        self.observation = observation if isinstance(observation, Observation) else Observation()

    def _attention(self, positions):
        obs = self.observation
        signal = np.zeros(self.n, dtype=float)
        self.cursor_vectors[:] = 0.
        if not obs.active:
            self.previous_cursor = None
            self.cursor_speed = 0.
            return signal
        x0, y0, x1, y1 = obs.view
        width, height = max(x1 - x0, 1.), max(y1 - y0, 1.)
        zoom = min(1., math.sqrt(WORLD_AREA / (width * height)) / 3.)
        margin_x = np.minimum(positions[:, 0] - x0, x1 - positions[:, 0])
        margin_y = np.minimum(positions[:, 1] - y0, y1 - positions[:, 1])
        inside = 1. / (1. + np.exp(-np.minimum(margin_x, margin_y) / 20.))
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        spread = max(width, height) / 3.
        foveal = np.exp(-np.sum((positions - center) ** 2, axis=1) / (2 * spread ** 2))
        signal += .35 * inside * (.4 + .6 * zoom) * (.5 + .5 * foveal)
        if obs.cursor is not None:
            cursor = np.asarray(obs.cursor, dtype=float)
            if self.previous_cursor is not None:
                step = float(np.linalg.norm(cursor - self.previous_cursor))
                self.cursor_speed = .6 * self.cursor_speed + .4 * step
            self.previous_cursor = cursor
            dwell = math.exp(-self.cursor_speed / DWELL_SCALE)
            delta = positions - cursor
            distance = np.linalg.norm(delta, axis=1)
            near = np.exp(-(distance ** 2) / (2 * GAZE_RADIUS ** 2))
            signal += .45 * near * (.45 + .55 * dwell)
            self.cursor_vectors = delta / np.maximum(distance, 1e-6)[:, None] * near[:, None]
        else:
            self.previous_cursor = None
            self.cursor_speed = 0.
        if obs.selected is not None and obs.selected < self.n:
            signal[obs.selected] += .9
        return np.clip(signal, 0., 1.)

    def update(self, positions):
        self.attention = self._attention(positions)
        sigma = 1. / self.sensitivity
        sample = self.attention + sigma * self.rng.standard_normal(self.n)
        predicted = self.posterior @ TRANSITION
        expected = predicted @ MEANS
        variance = (sigma ** 2 + SPREAD ** 2)[:, None]
        log_likelihood = -((sample[:, None] - MEANS[None, :]) ** 2) / (2 * variance)
        log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
        posterior = predicted * np.exp(log_likelihood)
        posterior = np.maximum(posterior / posterior.sum(axis=1, keepdims=True), FLOOR)
        self.posterior = posterior / posterior.sum(axis=1, keepdims=True)
        self.surprise = np.clip(np.abs(sample - expected) / (sigma + SPREAD), 0., 4.)
        belief = self.belief
        self.habituation = np.clip(.995 * self.habituation + .005 * belief, 0., 1.)
        novelty = np.clip(belief - self.habituation, 0., 1.)
        self.arousal = np.clip(.9 * self.arousal + .1 * np.clip(novelty * (1 + .15 * self.surprise), 0., 1.), 0., 1.)
        turned_on = (~self.watched) & (belief > ON_LEVEL)
        self.detections += int(turned_on.sum())
        self.watched = np.where(self.watched, belief > OFF_LEVEL, belief > ON_LEVEL)
        self.duration = np.where(self.watched, self.duration + 1, 0)
        return self.watched

    def share(self, listener, speaker, trust):
        if not (0 <= listener < self.n and 0 <= speaker < self.n) or listener == speaker:
            return False
        reliability = float(np.clip(trust, 0., 1.)) * .5
        likelihood = reliability * self.posterior[speaker] + (1. - reliability) / 3.
        updated = self.posterior[listener] * likelihood
        self.posterior[listener] = np.maximum(updated / updated.sum(), FLOOR)
        self.posterior[listener] /= self.posterior[listener].sum()
        self.social_reports += 1
        return True

    def shy(self):
        return self.watched & (self.boldness < .5)

    def performers(self):
        return self.watched & (self.boldness >= .5)

    def avoidance(self, positions):
        flee = np.zeros((self.n, 2), dtype=float)
        mask = self.shy()
        if mask.any():
            flee[mask] = (.6 + 1.4 * self.arousal[mask])[:, None] * self.cursor_vectors[mask]
        return flee

    def describe(self, agent):
        return {'crença_de_ser_observado': round(float(self.belief[agent]), 3), 'percebe_observador': bool(self.watched[agent]), 'estado_estimado': self.state(agent), 'confiança_metacognitiva': round(float(self.confidence[agent]), 3), 'excitação': round(float(self.arousal[agent]), 3), 'habituação': round(float(self.habituation[agent]), 3), 'perfil_sob_observação': 'exibido' if self.boldness[agent] >= .5 else 'reservado', 'sensibilidade': round(float(self.sensitivity[agent]), 3)}
