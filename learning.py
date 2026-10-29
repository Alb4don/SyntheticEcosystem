import math
from simulation import Config


class EpisodeLearner:
    def __init__(self, memory):
        self.memory = memory

    @staticmethod
    def _config(raw):
        if not isinstance(raw, dict):
            return None
        try:
            config = Config(raw.get('population'), raw.get('steps'), raw.get('seed')).validate()
        except (ValueError, TypeError):
            return None
        return config.as_dict()

    def recommend(self, target_polarization=0.25):
        try:
            target = float(target_polarization)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(target) or not 0 <= target <= 1:
            return None
        examples = []
        for episode in self.memory.recent(500):
            if not isinstance(episode, dict):
                continue
            config, final = self._config(episode.get('config')), episode.get('final')
            if config is None or not isinstance(final, dict):
                continue
            value = final.get('polarization')
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            p = float(value)
            if math.isfinite(p) and 0 <= p <= 1:
                examples.append((abs(p - target), config, p))
        if not examples:
            return None
        best = min(examples, key=lambda item: item[0])
        return {'config': best[1], 'observed_polarization': best[2], 'samples': len(examples), 'method': 'nearest-observed-episode', 'validated': False}
