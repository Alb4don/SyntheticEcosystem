import base64
import binascii
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from dialogue import ACTS, CONTEXTS

FORMAT = 'dal-dialogue-pack'
SCHEMA = 1
TOPIC_COUNT = 4
MAX_PACK_BYTES = 256 * 1024
MAX_WORDS = 12
LOCAL_CAP = 5000.
VERIFIED_CAP = 1000
UNVERIFIED_CAP = 200
PRIOR_STRENGTH = 20.
INHERIT_PROBABILITY = .6
DECAY = .9
MAX_CONTRIBUTIONS = 64
WORD = re.compile(r'^(?:[bdfgklmnprstvz][aeiou]){2,4}$')
PACK_KEYS = frozenset({'format', 'schema', 'app_version', 'acts', 'contexts', 'means', 'evidence', 'words', 'episodes'})
FILE_KEYS = frozenset({'pack', 'signature', 'key'})
VERSION = re.compile(r'^\d{1,4}\.\d{1,4}\.\d{1,4}$')


class KnowledgeError(ValueError):
    def __init__(self, message, cause='invalid'):
        super().__init__(message)
        self.cause = cause


def _number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KnowledgeError('Valor numérico inválido', 'type')
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise KnowledgeError('Valor fora do intervalo', 'range')
    return number


def _matrix(rows, low, high, integers=False):
    if not isinstance(rows, list) or len(rows) != CONTEXTS:
        raise KnowledgeError('Matriz com formato inválido', 'shape')
    result = np.zeros((CONTEXTS, len(ACTS)), dtype=float)
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(ACTS):
            raise KnowledgeError('Matriz com formato inválido', 'shape')
        for j, value in enumerate(row):
            if integers and (isinstance(value, bool) or not isinstance(value, int)):
                raise KnowledgeError('Evidência deve ser inteira', 'type')
            result[i, j] = _number(value, low, high)
    return result


def weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    values, weights = np.asarray(values, dtype=float)[order], np.asarray(weights, dtype=float)[order]
    total = weights.sum()
    if total <= 0:
        return float(np.median(values))
    cumulative = np.cumsum(weights)
    return float(values[int(np.searchsorted(cumulative, total / 2.))])


@dataclass
class DialogueKnowledge:
    means: np.ndarray = field(default_factory=lambda: np.full((CONTEXTS, len(ACTS)), .5))
    evidence: np.ndarray = field(default_factory=lambda: np.zeros((CONTEXTS, len(ACTS))))
    words: list = field(default_factory=lambda: [{} for _ in range(TOPIC_COUNT)])
    episodes: int = 0
    verified: bool = False

    @classmethod
    def from_simulation(cls, simulation):
        strategy = simulation.mind.strategy
        gained, lost = strategy.learned()
        success = gained.sum(axis=0)
        trials = (gained + lost).sum(axis=0)
        means = np.where(trials > 0, success / np.maximum(trials, 1e-9), .5)
        lexicon = simulation.mind.lexicon
        words = []
        for topic in range(TOPIC_COUNT):
            counts = Counter(w for w in (lexicon.top(a, topic) for a in range(simulation.n)) if w and WORD.match(w))
            total = sum(counts.values())
            words.append({w: c / total for w, c in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_WORDS]} if total else {})
        return cls(np.clip(means, 0., 1.), np.minimum(trials, LOCAL_CAP), words, 1, True)

    def merge(self, other, decay=DECAY):
        old, new = self.evidence * decay, other.evidence
        total = old + new
        means = np.where(total > 0, (self.means * old + other.means * new) / np.maximum(total, 1e-9), .5)
        words = []
        for mine, theirs in zip(self.words, other.words):
            combined = Counter({w: s * decay for w, s in mine.items()})
            combined.update(theirs)
            norm = sum(combined.values())
            ranked = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:MAX_WORDS]
            words.append({w: s / norm for w, s in ranked} if norm else {})
        return DialogueKnowledge(np.clip(means, 0., 1.), np.minimum(total, LOCAL_CAP), words, self.episodes + other.episodes, self.verified and other.verified)

    def prior(self, strength=PRIOR_STRENGTH):
        weight = np.minimum(self.evidence, strength)
        return 1. + self.means * weight, 1. + (1. - self.means) * weight

    def inherited(self, topic):
        words = self.words[topic] if 0 <= topic < len(self.words) else {}
        return max(words.items(), key=lambda item: (item[1], item[0]))[0] if words else None

    def strength(self):
        return float(self.evidence.sum())

    def to_pack(self, app_version):
        return {'format': FORMAT, 'schema': SCHEMA, 'app_version': app_version, 'acts': list(ACTS), 'contexts': CONTEXTS, 'means': [[round(float(v), 4) for v in row] for row in self.means], 'evidence': [[int(min(VERIFIED_CAP, round(float(v)))) for v in row] for row in self.evidence], 'words': {str(t): [[w, round(float(s), 4)] for w, s in sorted(ws.items(), key=lambda item: (-item[1], item[0]))] for t, ws in enumerate(self.words)}, 'episodes': int(min(self.episodes, 10**6))}

    def to_record(self):
        return {'means': self.means.round(5).tolist(), 'evidence': self.evidence.round(3).tolist(), 'words': [dict(w) for w in self.words], 'episodes': int(self.episodes), 'verified': bool(self.verified)}

    @classmethod
    def from_record(cls, record):
        if not isinstance(record, dict):
            raise KnowledgeError('Registro inválido', 'type')
        means = _matrix(record.get('means'), 0., 1.)
        evidence = _matrix(record.get('evidence'), 0., LOCAL_CAP)
        raw_words = record.get('words')
        if not isinstance(raw_words, list) or len(raw_words) != TOPIC_COUNT:
            raise KnowledgeError('Vocabulário inválido', 'shape')
        words = []
        for entry in raw_words:
            if not isinstance(entry, dict) or len(entry) > MAX_WORDS:
                raise KnowledgeError('Vocabulário inválido', 'shape')
            words.append({w: _number(s, 0., 1.) for w, s in entry.items() if isinstance(w, str) and WORD.match(w)})
        episodes = record.get('episodes', 0)
        if isinstance(episodes, bool) or not isinstance(episodes, int) or not 0 <= episodes <= 10**7:
            raise KnowledgeError('Contagem inválida', 'range')
        return cls(means, evidence, words, episodes, record.get('verified') is True)


def validate_pack(pack, verified=False):
    if not isinstance(pack, dict) or set(pack) != PACK_KEYS:
        raise KnowledgeError('Pacote com campos inválidos', 'schema')
    if pack['format'] != FORMAT or pack['schema'] != SCHEMA or pack['contexts'] != CONTEXTS:
        raise KnowledgeError('Formato de pacote não suportado', 'schema')
    if pack['acts'] != list(ACTS):
        raise KnowledgeError('Atos de fala incompatíveis', 'schema')
    if not isinstance(pack['app_version'], str) or not VERSION.match(pack['app_version']):
        raise KnowledgeError('Versão inválida', 'schema')
    cap = VERIFIED_CAP if verified else UNVERIFIED_CAP
    means = _matrix(pack['means'], 0., 1.)
    evidence = np.minimum(_matrix(pack['evidence'], 0, VERIFIED_CAP, integers=True), cap)
    raw = pack['words']
    if not isinstance(raw, dict) or set(raw) != {str(t) for t in range(TOPIC_COUNT)}:
        raise KnowledgeError('Vocabulário inválido', 'words')
    words = []
    for topic in range(TOPIC_COUNT):
        entries = raw[str(topic)]
        if not isinstance(entries, list) or len(entries) > MAX_WORDS:
            raise KnowledgeError('Vocabulário inválido', 'words')
        table = {}
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str) or not WORD.match(entry[0]) or entry[0] in table:
                raise KnowledgeError('Termo inválido', 'words')
            table[entry[0]] = _number(entry[1], 0., 1.)
        if sum(table.values()) > 1.001:
            raise KnowledgeError('Distribuição de termos inválida', 'words')
        words.append(table)
    episodes = pack['episodes']
    if isinstance(episodes, bool) or not isinstance(episodes, int) or not 0 <= episodes <= 10**6:
        raise KnowledgeError('Contagem inválida', 'range')
    return DialogueKnowledge(means, evidence, words, episodes, verified)


def pack_bytes(pack):
    return json.dumps(pack, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def pack_digest(pack):
    return hashlib.sha256(pack_bytes(pack)).hexdigest()


def load_pack_file(data, trusted_keys=()):
    if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_PACK_BYTES:
        raise KnowledgeError('Pacote excede o limite', 'size')
    try:
        document = json.loads(bytes(data).decode('utf-8'))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise KnowledgeError('Pacote ilegível', 'parse') from None
    if not isinstance(document, dict) or not set(document) <= FILE_KEYS or 'pack' not in document:
        raise KnowledgeError('Pacote com campos inválidos', 'schema')
    pack = document['pack']
    verified = False
    if 'signature' in document:
        try:
            signature = base64.b64decode(str(document['signature']).encode('ascii'), validate=True)
            key = bytes.fromhex(str(document.get('key', '')))
        except (binascii.Error, ValueError, UnicodeEncodeError):
            raise KnowledgeError('Assinatura inválida', 'signature') from None
        if key in trusted_keys:
            try:
                Ed25519PublicKey.from_public_bytes(key).verify(signature, pack_bytes(pack))
            except (InvalidSignature, ValueError):
                raise KnowledgeError('Assinatura inválida', 'signature') from None
            verified = True
    return validate_pack(pack, verified), pack_digest(pack)


def robust_merge(contributions):
    items = [c for c in contributions if c is not None and c.strength() > 0]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    means = np.full((CONTEXTS, len(ACTS)), .5)
    evidence = np.zeros((CONTEXTS, len(ACTS)))
    for i in range(CONTEXTS):
        for j in range(len(ACTS)):
            weights = np.array([c.evidence[i, j] for c in items])
            if weights.sum() <= 0:
                continue
            means[i, j] = weighted_median([c.means[i, j] for c in items], weights)
            evidence[i, j] = min(LOCAL_CAP, float(np.median(weights[weights > 0])) * min(len(items), 4))
    words = []
    for topic in range(TOPIC_COUNT):
        votes = Counter()
        for c in items:
            leader = c.inherited(topic)
            if leader:
                votes[leader] += 1
        quorum = max(2, (len(items) + 1) // 2)
        accepted = {w: v / len(items) for w, v in votes.items() if v >= quorum}
        words.append(dict(sorted(accepted.items(), key=lambda item: (-item[1], item[0]))[:MAX_WORDS]))
    return DialogueKnowledge(means, evidence, words, sum(c.episodes for c in items), all(c.verified for c in items))


def load_trusted_keys(path):
    keys = set()
    try:
        from pathlib import Path
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
            return frozenset()
        for line in path.read_text(encoding='ascii').splitlines():
            line = line.strip()
            if re.fullmatch(r'[0-9a-f]{64}', line):
                keys.add(bytes.fromhex(line))
    except (OSError, UnicodeDecodeError):
        return frozenset()
    return frozenset(keys)


def stored_contributions(store):
    local, packs = None, []
    for source, record in store.knowledge_items():
        try:
            item = DialogueKnowledge.from_record(record)
        except KnowledgeError:
            continue
        if source == 'local':
            local = item
        else:
            packs.append(item)
    return local, packs


def effective_knowledge(store):
    local, packs = stored_contributions(store)
    return robust_merge(([local] if local is not None else []) + packs)


def record_episode(store, learned):
    local, _ = stored_contributions(store)
    merged = learned if local is None else local.merge(learned)
    store.put_knowledge('local', merged.to_record())
    return merged


def import_pack(store, data, trusted_keys=()):
    item, digest = load_pack_file(data, trusted_keys)
    if item.strength() <= 0:
        raise KnowledgeError('Pacote sem evidência', 'empty')
    source = 'pack:' + digest[:16]
    store.put_knowledge(source, item.to_record())
    return source, item


def export_document(store, app_version):
    local, _ = stored_contributions(store)
    if local is None or local.strength() <= 0:
        raise KnowledgeError('Sem aprendizado local para exportar', 'empty')
    return {'pack': local.to_pack(app_version)}
