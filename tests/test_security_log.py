import logging
import os
from pathlib import Path
import logsafe
from security import SecurityJournal


def test_redaction_of_sensitive_tokens():
    home = str(Path.home())
    text = logsafe.redact(f'{home}/x key={"ab" * 32} b64={"QUJD" * 12} user@example.com line\x1b[31mred\rinject')
    assert home not in text and '~/x' in text
    assert 'ab' * 32 not in text and '[REDACTED-HEX]' in text
    assert '[REDACTED-B64]' in text and '[REDACTED-EMAIL]' in text
    assert '\x1b' not in text and '\r' not in text
    assert len(logsafe.redact('x' * 20000)) < 8100


def test_configure_writes_redacted_private_log(tmp_path):
    handler = logsafe.configure(tmp_path)
    try:
        try:
            raise RuntimeError('secret ' + 'f' * 40)
        except RuntimeError:
            logging.getLogger('t').exception('falha')
        handler.flush()
        content = (tmp_path / 'errors.log').read_text(encoding='utf-8')
        assert 'falha' in content and 'f' * 40 not in content
        if os.name != 'nt':
            assert ((tmp_path / 'errors.log').stat().st_mode & 0o777) == 0o600
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_journal_threshold_avoids_false_positives():
    now = [0.]
    journal = SecurityJournal(window=100, threshold=8, clock=lambda: now[0])
    for _ in range(3):
        journal.record('update_state', 'low', 'io')
    assert not journal.blocked()
    journal.record('update_rejected', 'high', 'signature')
    assert not journal.blocked()
    now[0] = 500.
    journal.record('update_rejected', 'high', 'signature')
    assert not journal.blocked()
    assert journal.record('update_rejected', 'high', 'traversal')
    assert journal.blocked()


def test_journal_sanitizes_labels(caplog):
    journal = SecurityJournal()
    with caplog.at_level(logging.WARNING):
        journal.record('bad\nlabel', 'nonsense', '../x')
    assert 'category=other' in caplog.text and 'cause=other' in caplog.text and 'severity=medium' in caplog.text
