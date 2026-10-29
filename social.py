from collections import deque
from dataclasses import dataclass
import numpy as np
from dialogue import ACTS, ACT_INDEX, Lexicon, StrategyLearner, context

TOPICS = ('recursos', 'território', 'cooperação', 'conhecimento')
AFFINITY_LIMIT = 60000
AFFINITY_KEEP = 40000


@dataclass(frozen=True)
class Utterance:
    step: int
    speaker: int
    listener: int
    text: str
    topic: str
    act: str = 'informar'
    reply: str = ''
    term: str = ''


class SocialMind:
    def __init__(self, population, seed):
        self.population = population
        self.rng = np.random.default_rng(seed ^ 0xA17C)
        self.energy = self.rng.uniform(.45, .95, population)
        self.curiosity = self.rng.uniform(.15, .95, population)
        self.trust = self.rng.uniform(.25, .8, population)
        self.social_need = self.rng.uniform(.2, .8, population)
        self.knowledge = self.rng.uniform(0, 1, (population, 4))
        self.goals = self.rng.integers(0, 4, population)
        self.last_spoken = np.full(population, -100, dtype=int)
        self.experience = np.zeros(population, dtype=int)
        self.autonomy = np.full(population, .15)
        self.awareness = np.full(population, .10)
        self.affinity = {}
        self.events = []
        self.current = []
        self.links = []
        self.max_events = 1500
        self.strategy = StrategyLearner(population, np.random.default_rng(seed ^ 0x5EED))
        self.lexicon = Lexicon(population, len(TOPICS), np.random.default_rng(seed ^ 0x1E81))
        self.act_counts = np.zeros(len(ACTS), dtype=int)
        self.success = 0
        self.attempts = 0
        self._coherence = 0.
        self.rewards = deque(maxlen=200)

    def _tier(self, agent):
        experience = int(self.experience[agent])
        return 0 if experience < 8 else (1 if experience < 30 else 2)

    def _pick(self, options):
        return options[int(self.rng.integers(len(options)))]

    def _learn(self, learner, teacher, topic, rate):
        delta = rate * float(self.trust[learner]) * (self.knowledge[teacher, topic] - self.knowledge[learner, topic])
        self.knowledge[learner, topic] = float(np.clip(self.knowledge[learner, topic] + delta, 0, 1))
        return abs(delta)

    def _realize(self, act, a, b, topic, term, gap, similarity, simulation, tick):
        life = simulation.life
        name = TOPICS[topic]
        tier = self._tier(a)
        ka, kb = float(self.knowledge[a, topic]), float(self.knowledge[b, topic])
        if act == 'saudar':
            if tier == 0:
                text = f'Olá, #{b}.'
            elif tier == 1:
                text = self._pick((f'Olá, #{b}. Tudo bem por aí?', f'Oi, #{b}. Por onde você tem andado?'))
            elif self.affinity.get((a, b), .5) > .55:
                text = f'Bom te ver de novo, #{b}. Como anda a busca por «{term}»?'
            else:
                text = f'Olá, #{b}. Sou #{a} e tenho pensado em «{term}».'
            reward = .2 + .3 * float(self.social_need[b])
            reply = 'Olá! Bom conversar.' if self.social_need[b] > .5 else 'Oi. Estou de passagem.'
            return text, reply, reward
        if act == 'informar':
            if gap > .12 and ka > kb:
                learned = self._learn(b, a, topic, .14)
                text = (f'Sei algo sobre «{term}».', f'Sobre «{term}» ({name}): encontrei padrões úteis e posso compartilhar.', f'Sobre «{term}» ({name}): meu domínio está em {ka:.0%} e o seu em {kb:.0%}; vou te passar o que aprendi.')[tier]
                reply = 'Obrigado, isso muda minha visão.' if learned > .02 else 'Interessante, vou pensar nisso.'
                return text, reply, min(1., learned * 8 + .1)
            if kb > ka + .12:
                learned = self._learn(a, b, topic, .07)
                text = f'Acho que entendo «{term}» ({name}).'
                return text, 'Na verdade, observei algo diferente; deixa eu corrigir.', min(1., learned * 6 + .15)
            return f'Sobre «{term}», nada de novo por enquanto.', 'Entendo.', .15
        if act == 'perguntar':
            text = (f'O que é «{term}»?', f'Como você entende «{term}»?', f'Você parece entender mais de «{term}» do que eu ({ka:.0%}). Pode me explicar?')[tier]
            if kb > ka + .12:
                learned = self._learn(a, b, topic, .14)
                return text, 'Claro, deixa eu explicar o que vi.', min(1., learned * 8 + .1)
            return text, 'Também sei pouco sobre isso.', .15
        if act == 'propor':
            accepted = similarity > .6 and float(self.trust[b]) > .35
            text = ('Vamos cooperar?', f'Vamos cooperar em «{term}»?', f'Proponho cooperarmos em «{term}»: nossas visões estão próximas ({similarity:.0%}).')[tier]
            if accepted:
                pa, pb = float(simulation.opinions[a]), float(simulation.opinions[b])
                simulation.opinions[a] = float(np.clip(pa + .05 * (pb - pa), -1, 1))
                simulation.opinions[b] = float(np.clip(pb + .05 * (pa - pb), -1, 1))
                return text, 'Combinado, vamos juntos.', .65
            return text, 'Agora não; prefiro seguir meu caminho.', .1
        if act == 'concordar':
            text = f'Concordo com você sobre «{term}».' if tier < 2 else f'Concordo sobre «{term}»: vemos isso de modo parecido ({similarity:.0%}).'
            if similarity > .6:
                return text, 'Bom saber que pensamos igual.', min(1., .2 + .8 * (similarity - .5))
            return text, 'Hum, não tenho tanta certeza.', .05
        if act == 'discordar':
            text = f'Vejo «{term}» de outro jeito.' if tier < 2 else f'Discordo sobre «{term}»: nossas posições divergem ({1 - similarity:.0%}).'
            if similarity < .5:
                return text, 'Respeito sua visão, mesmo diferente.', .45
            return text, 'Mas pensamos parecido!', .1
        if act == 'indicar_recurso':
            resource = life.last_resource(a)
            if resource is None:
                return f'Ainda procuro «{term}».', 'Boa sorte.', .05
            distance, direction = life.bearing(b, resource)
            text = (f'Há suprimentos no recurso {resource}.', f'Há suprimentos no recurso {resource}, ao {direction}.', f'Encontrei suprimentos no recurso {resource}: cerca de {distance:.0f} passos ao {direction} de você.')[tier]
            if life.hunger[b] > .55 and life.direct_to(b, resource, tick):
                life.remember(b, tick, 'tip', resource, .8)
                return text, 'Estou com fome; vou até lá agora!', .8
            return text, 'Bom saber, vou guardar isso.', .25
        observer = simulation.observer
        certainty = float(observer.confidence[a])
        state = observer.phrase(a)
        cycles = int(observer.duration[a])
        span = f'{cycles} ciclo' if cycles == 1 else f'{cycles} ciclos'
        habituated = float(observer.habituation[a]) > .45
        if observer.boldness[a] >= .5:
            if habituated:
                text = 'Continuam nos observando; já me acostumei e sigo normalmente.'
            else:
                text = ('Alguém está olhando para nós.', f'Sinto que estamos sendo observados... vamos mostrar o que sabemos sobre «{term}».', f'Estou {state} há {span} (certeza {certainty:.0%}). Vamos agir com naturalidade.')[tier]
        elif habituated:
            text = 'Ainda nos vigiam, mas já não me assusta tanto.'
        else:
            text = ('Fale baixo... alguém está nos vigiando.', 'Fale baixo... acho que alguém nos vigia. Vamos nos afastar.', f'Fale baixo: estou {state} há {span} (certeza {certainty:.0%}). Prefiro me afastar.')[tier]
        already = bool(observer.watched[b])
        observer.share(b, a, float(self.trust[b]))
        if already:
            return text, 'Também percebi esse olhar.', .15 + .7 * float(observer.belief[b])
        if observer.belief[b] > .5:
            return text, 'Não tinha notado, mas agora estou atento.', .15 + .7 * float(observer.belief[b])
        return text, 'Não notei nada de estranho.', .15 + .7 * float(observer.belief[b])

    def step(self, simulation, tick):
        self.current = []
        self.links = []
        n = self.population
        life = simulation.life
        observer = simulation.observer
        belief = observer.belief
        self.energy = np.clip(self.energy + .016 - .012 * self.social_need, 0, 1)
        self.social_need = np.clip(self.social_need + .009, 0, 1)
        adjacency = simulation.graph.adj
        for _ in range(min(32, max(2, n // 8))):
            a = int(self.rng.integers(n))
            if tick - self.last_spoken[a] < 2 or self.energy[a] < .13:
                continue
            visible = life.observe(a)['agents']
            if not visible:
                continue
            ties = adjacency[a]
            scores = np.array([.1 + self.affinity.get((a, b), .5) + .15 * self.social_need[a] + (.25 if b in ties else 0.) for b in visible], dtype=float)
            scores /= scores.sum()
            b = int(self.rng.choice(visible, p=scores))
            if a == b or self.energy[b] < .1:
                continue
            topic = int(self.goals[a])
            gap = float(abs(self.knowledge[a, topic] - self.knowledge[b, topic]))
            similarity = 1 - abs(float(simulation.opinions[a] - simulation.opinions[b])) / 2
            watched = bool(observer.watched[a])
            bold = float(observer.boldness[a])
            willingness = .12 + .25 * float(self.social_need[a]) + .20 * float(self.curiosity[a]) + .18 * similarity + .15 * float(self.autonomy[a])
            if watched:
                arousal = float(observer.arousal[a])
                willingness += .15 * arousal * bold - .25 * arousal * (1 - bold)
            self.attempts += 1
            if self.rng.random() > float(np.clip(willingness, .02, .95)):
                continue
            ctx = context(gap > .12, similarity > .7, watched, life.hunger[b] > .55)
            allowed = np.ones(len(ACTS), dtype=bool)
            allowed[ACT_INDEX['indicar_recurso']] = life.last_resource(a) is not None
            allowed[ACT_INDEX['alertar_observador']] = watched
            act_index = self.strategy.choose(a, ctx, allowed)
            act = ACTS[act_index]
            term, understood, heard = self.lexicon.interact(a, b, topic)
            text, reply, reward = self._realize(act, a, b, topic, term, gap, similarity, simulation, tick)
            if not understood:
                reply = (f'«{term}»? Eu chamo isso de «{heard}». ' if heard else f'Não conhecia «{term}»; vou usar esse termo. ') + reply
            else:
                reward = min(1., reward + .1)
                self.success += 1
            reward = float(np.clip(reward, 0., 1.))
            self.rewards.append(reward)
            self.strategy.update(a, ctx, act_index, reward)
            self.strategy.imitate(b, ctx, act_index, reward)
            self.act_counts[act_index] += 1
            for actor, partner in ((a, b), (b, a)):
                self.experience[actor] += 1
                self.affinity[(actor, partner)] = float(np.clip(.9 * self.affinity.get((actor, partner), .5) + .1 * reward, .05, .95))
                self.awareness[actor] = float(np.clip(self.awareness[actor] + .004 * (1 - self.awareness[actor]) + .006 * float(belief[actor]) * (1 - self.awareness[actor]), 0, .95))
                self.autonomy[actor] = float(np.clip(self.autonomy[actor] + .003 * (1 - self.autonomy[actor]), 0, .95))
                self.trust[actor] = float(np.clip(self.trust[actor] + .008 * (reward - .4), .05, .95))
            self.energy[a] = max(0, self.energy[a] - .028)
            self.energy[b] = max(0, self.energy[b] - .012)
            self.social_need[a] = max(0, self.social_need[a] - .10)
            self.social_need[b] = max(0, self.social_need[b] - .07)
            self.last_spoken[a] = tick
            self.last_spoken[b] = tick
            if self.experience[a] % 12 == 0:
                self.goals[a] = int(np.argmin(self.knowledge[a]))
            event = Utterance(tick, a, b, text, TOPICS[topic], act, reply, term)
            self.current.append(event)
            self.events.append(event)
            self.links.append((a, b, reward))
            life.remember(a, tick, 'conversation', b, reward)
            life.remember(b, tick, 'conversation', a, reward)
        if len(self.events) > self.max_events:
            del self.events[:-self.max_events]
        if len(self.affinity) > AFFINITY_LIMIT:
            ranked = sorted(self.affinity.items(), key=lambda item: (-abs(item[1] - .5), item[0]))
            self.affinity = dict(ranked[:AFFINITY_KEEP])
        if tick % 5 == 0 or tick <= 1:
            self._coherence = self.lexicon.coherence()
        return self.current

    @property
    def coherence(self):
        return self._coherence

    def recent_reward(self):
        return float(np.mean(self.rewards)) if self.rewards else 0.

    def understanding_rate(self):
        total = int(self.act_counts.sum())
        return self.success / total if total else 0.

    def describe(self, agent):
        vocabulary = {TOPICS[t]: w for t, w in self.lexicon.vocabulary(agent).items() if w}
        return {'energia': round(float(self.energy[agent]), 3), 'curiosidade': round(float(self.curiosity[agent]), 3), 'confiança': round(float(self.trust[agent]), 3), 'necessidade_social': round(float(self.social_need[agent]), 3), 'objetivo': TOPICS[int(self.goals[agent])], 'experiências': int(self.experience[agent]), 'autonomia_simulada': round(float(self.autonomy[agent]), 3), 'autopercepção_simulada': round(float(self.awareness[agent]), 3), 'estratégia_dominante': self.strategy.dominant(agent), 'vocabulário': vocabulary}
