from collections import Counter
import numpy as np

ACTS = ('saudar', 'informar', 'perguntar', 'propor', 'concordar', 'discordar', 'indicar_recurso', 'alertar_observador')
ACT_INDEX = {name: index for index, name in enumerate(ACTS)}
CONTEXTS = 16
CONSONANTS = 'bdfgklmnprstvz'
VOWELS = 'aeiou'


def context(gap_high, similar, watched, partner_hungry):
    return int(bool(gap_high)) | int(bool(similar)) << 1 | int(bool(watched)) << 2 | int(bool(partner_hungry)) << 3


class StrategyLearner:
    def __init__(self, population, rng, discount=.99):
        self.rng = rng
        self.discount = float(discount)
        self.alpha = np.ones((population, CONTEXTS, len(ACTS)), dtype=float)
        self.beta = np.ones((population, CONTEXTS, len(ACTS)), dtype=float)
        self.alpha0 = np.ones((CONTEXTS, len(ACTS)), dtype=float)
        self.beta0 = np.ones((CONTEXTS, len(ACTS)), dtype=float)

    def seed_prior(self, alpha0, beta0):
        alpha0, beta0 = np.asarray(alpha0, dtype=float), np.asarray(beta0, dtype=float)
        if alpha0.shape != self.alpha0.shape or beta0.shape != self.beta0.shape or (alpha0 < 1).any() or (beta0 < 1).any() or not np.isfinite(alpha0).all() or not np.isfinite(beta0).all():
            raise ValueError('Prior inválido')
        self.alpha0, self.beta0 = alpha0.copy(), beta0.copy()
        self.alpha[:] = alpha0
        self.beta[:] = beta0

    def learned(self):
        return np.maximum(self.alpha - self.alpha0, 0.), np.maximum(self.beta - self.beta0, 0.)

    def choose(self, agent, ctx, allowed):
        samples = self.rng.beta(self.alpha[agent, ctx], self.beta[agent, ctx])
        samples = np.where(allowed, samples, -1.)
        return int(np.argmax(samples))

    def update(self, agent, ctx, act, reward):
        reward = float(np.clip(reward, 0., 1.))
        self.alpha[agent, ctx] = self.alpha0[ctx] + self.discount * (self.alpha[agent, ctx] - self.alpha0[ctx])
        self.beta[agent, ctx] = self.beta0[ctx] + self.discount * (self.beta[agent, ctx] - self.beta0[ctx])
        self.alpha[agent, ctx, act] += reward
        self.beta[agent, ctx, act] += 1. - reward

    def imitate(self, agent, ctx, act, reward, rate=.3):
        if reward > .55:
            self.alpha[agent, ctx, act] += rate * float(reward)

    def expected(self, agent):
        mean = self.alpha[agent] / (self.alpha[agent] + self.beta[agent])
        return mean.mean(axis=0)

    def dominant(self, agent):
        return ACTS[int(np.argmax(self.expected(agent)))]


class Lexicon:
    def __init__(self, population, topics, rng, capacity=5):
        self.rng = rng
        self.topics = int(topics)
        self.capacity = int(capacity)
        self.entries = [[{} for _ in range(self.topics)] for _ in range(population)]

    def _invent(self):
        syllables = int(self.rng.integers(2, 4))
        return ''.join(CONSONANTS[int(self.rng.integers(len(CONSONANTS)))] + VOWELS[int(self.rng.integers(len(VOWELS)))] for _ in range(syllables))

    def top(self, agent, topic):
        words = self.entries[agent][topic]
        if not words:
            return None
        return max(words.items(), key=lambda item: (item[1], item[0]))[0]

    def speak(self, agent, topic):
        word = self.top(agent, topic)
        if word is None:
            word = self._invent()
            self.entries[agent][topic][word] = .5
        return word

    def _trim(self, words):
        while len(words) > self.capacity:
            weakest = min(words.items(), key=lambda item: (item[1], item[0]))[0]
            del words[weakest]

    def interact(self, speaker, listener, topic):
        word = self.speak(speaker, topic)
        heard = self.top(listener, topic)
        mine, theirs = self.entries[speaker][topic], self.entries[listener][topic]
        if word in theirs:
            for words in (mine, theirs):
                words[word] = min(1., words[word] + .2)
                for other in [w for w in words if w != word]:
                    words[other] -= .15
                    if words[other] <= 0:
                        del words[other]
            return word, True, heard
        theirs[word] = .5 if not theirs else .45
        self._trim(theirs)
        mine[word] = max(.05, mine[word] - .05)
        return word, False, heard

    def inherit(self, agent, topic, word, score=.5):
        words = self.entries[agent][topic]
        if word not in words:
            words[word] = score
            self._trim(words)

    def coherence(self):
        population = len(self.entries)
        if not population:
            return 0.
        total = 0.
        for topic in range(self.topics):
            tops = Counter(word for word in (self.top(a, topic) for a in range(population)) if word is not None)
            speakers = sum(tops.values())
            total += (tops.most_common(1)[0][1] / speakers) if speakers else 0.
        return total / self.topics

    def vocabulary(self, agent):
        return {topic: self.top(agent, topic) for topic in range(self.topics)}
