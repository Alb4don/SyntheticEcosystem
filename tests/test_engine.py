import threading
import time
import pytest
from PySide6.QtCore import QEventLoop, QObject, QThread, QTimer, Signal
from app import Engine, export_records, format_detail, speed_profile
from observer import Observation
from secure_store import VolatileStore
from simulation import Config, Simulation


class Driver(QObject):
    start = Signal(object, int, object)


def _wait(condition, timeout=30.):
    loop = QEventLoop()
    deadline = time.monotonic() + timeout
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if condition() or time.monotonic() > deadline else None)
    timer.start()
    loop.exec()
    timer.stop()
    return condition()


@pytest.fixture
def worker(qapp):
    thread = QThread()
    engine = Engine()
    engine.moveToThread(thread)
    thread.started.connect(engine.setup)
    yield thread, engine
    thread.quit()
    thread.wait(5000)


def test_engine_runs_in_background_thread(worker):
    thread, engine = worker
    driver = Driver()
    driver.start.connect(engine.start_run)
    results, frames = [], []
    engine.completed.connect(lambda run, payload: results.append((run, payload)))
    engine.frame_ready.connect(lambda: frames.append(engine.take_snapshot()))
    engine.configure_speed(16, 4)
    thread.start()
    engine.set_observation(Observation.sanitize(True, (-100, -100, 100, 100), (0, 0), 1, 40))
    engine.select(1)
    driver.start.emit(Config(40, 50, 3), 7, None)
    assert _wait(lambda: bool(results))
    run, payload = results[0]
    assert run == 7 and len(payload['records']) == 50 and payload['episode']['config']['population'] == 40
    assert payload['knowledge'] is not None and payload['knowledge'].strength() > 0
    snapshots = [f for f in frames if f is not None]
    assert snapshots and all(r == 7 for r, _ in snapshots)
    assert any(s.detail and s.detail['agente'] == 1 for _, s in snapshots)


def test_engine_reports_invalid_config_without_crashing(worker):
    thread, engine = worker
    driver = Driver()
    driver.start.connect(engine.start_run)
    failures = []
    engine.failed.connect(failures.append)
    thread.start()
    driver.start.emit(Config(1, 5, 1), 1, None)
    assert _wait(lambda: bool(failures))
    assert failures == ['Parâmetros inválidos.']


def test_window_end_to_end(qapp, tmp_path, make_window):
    memory = VolatileStore()
    window = make_window(memory)
    window.population.setValue(30)
    window.steps.setValue(40)
    window.speed.setValue(60)
    window.restart()
    assert _wait(lambda: window.records is not None)
    assert memory.count() == 1 and window.export.isEnabled()
    window.inspect(2)
    window.toggle_observer()
    window.toggle_links()
    window.toggle_conversations()
    window.suggest()
    path = export_records(window.records, tmp_path / 'out')
    assert path.suffix == '.csv' and path.read_text(encoding='utf-8').startswith('step,')
    window.close()


def test_restart_while_running_ignores_stale_frames(qapp, make_window):
    window = make_window()
    window.population.setValue(50)
    window.steps.setValue(2000)
    window.restart()
    _wait(lambda: window.last_step > 3, 10)
    window.population.setValue(25)
    window.steps.setValue(20)
    window.restart()
    assert _wait(lambda: window.records is not None)
    assert len(window.network.humanoids) == 25 and len(window.records) == 20
    window.close()


def test_export_sanitizes_formula_cells(tmp_path):
    path = export_records([{'step': 1, 'note': '=HYPERLINK("x")'}, {'step': 2, 'note': '+1'}], tmp_path / 'a.txt')
    text = path.read_text(encoding='utf-8')
    assert path.name == 'a.csv' and "'=HYPERLINK" in text and "'+1" in text


def test_format_detail_and_speed_profile():
    simulation = Simulation(Config(10, 3, 1))
    simulation.run()
    text = format_detail(simulation.detail(0))
    assert 'ORGANISMO SINTÉTICO #0' in text and 'Vocabulário' in text
    assert format_detail(None).startswith('Clique')
    assert speed_profile(1) == (500, 1) and speed_profile(60) == (16, 6) and speed_profile(10**9) == (16, 6)


def test_update_scan_runs_while_simulation_continues(qapp, tmp_path, make_window):
    from helpers import drop, keypair
    from updates import UpdateManager
    private, public = keypair()
    manager = UpdateManager(tmp_path, '9.0.0', public, clock=lambda: time.time() + 60)
    drop(manager.inbox, private, '9.1.0')
    window = make_window(manager=manager)
    window.population.setValue(40)
    window.steps.setValue(150)
    window.speed.setValue(60)
    window.restart()
    _wait(lambda: window.last_step > 2, 10)
    before = window.last_step
    window.check_updates()
    assert _wait(lambda: 'preparada' in window.notice.text())
    assert _wait(lambda: window.records is not None)
    assert window.last_step > before and threading.active_count() >= 1
    assert manager.load_state()['pending'] == '9.1.0'
    window.close()
