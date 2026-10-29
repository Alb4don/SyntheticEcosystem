import logging
import re
import threading
import time
from collections import deque

SEVERITY = {'low': 1, 'medium': 2, 'high': 4}
_LABEL = re.compile(r'^[a-z_]{1,40}$')


class SecurityJournal:
    def __init__(self, window=900., threshold=8, clock=time.monotonic):
        self.window = float(window)
        self.threshold = int(threshold)
        self.clock = clock
        self.events = deque(maxlen=256)
        self.lock = threading.Lock()
        self.lockdown = False
        self.logger = logging.getLogger('dal.security')

    @staticmethod
    def _label(value):
        value = str(value)
        return value if _LABEL.match(value) else 'other'

    def record(self, category, severity='medium', cause='unknown'):
        severity = severity if severity in SEVERITY else 'medium'
        category, cause = self._label(category), self._label(cause)
        with self.lock:
            now = self.clock()
            self.events.append((now, category, severity, cause))
            score = sum(SEVERITY[s] for t, _, s, _ in self.events if now - t <= self.window)
            triggered = not self.lockdown and score >= self.threshold
            if triggered:
                self.lockdown = True
        self.logger.warning('security_event category=%s severity=%s cause=%s', category, severity, cause)
        if triggered:
            self.logger.error('security_lockdown score=%d window=%ds', score, int(self.window))
        return triggered

    def score(self):
        with self.lock:
            now = self.clock()
            return sum(SEVERITY[s] for t, _, s, _ in self.events if now - t <= self.window)

    def blocked(self):
        with self.lock:
            return self.lockdown
