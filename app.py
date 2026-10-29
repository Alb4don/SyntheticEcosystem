import csv
import json
import logging
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QRect, QRunnable, QSettings, QThread, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget

import knowledge
import layout
from flow_layout import FlowLayout, flow_policy
import logsafe
import updates
from learning import EpisodeLearner
from living_view import LivingView
from local_store import LocalStore
from observer import Observation
from secure_store import StoreError, VolatileStore
from security import SecurityJournal
from simulation import Config, Simulation
from version import __version__

BASE = Path.home() / '.digital_anthropology_lab'
FRAME_BUDGET = .045
GENERIC_FAILURE = 'Não foi possível continuar a simulação. Detalhes técnicos foram registrados localmente.'
UNSAFE_CELL = ('=', '+', '-', '@', '\t', '\r')
STYLE = """QMainWindow,QWidget{{background:#101827;color:#e2e8f0;font-size:{font}px;}} QLabel#header{{font-size:{header}px;font-weight:700;color:#67e8f9;padding:{pad}px;}} QLabel#metrics{{font-size:{metrics}px;color:#a7f3d0;padding:8px;}} QLabel#notice{{background:#3b2f0b;color:#fde68a;padding:8px;border-radius:6px;}} QPushButton,QToolButton{{background:#1e3a5f;border:1px solid #334e68;border-radius:6px;padding:{button}px;}} QPushButton:hover,QToolButton:hover{{background:#285785;}} QPushButton:checked{{background:#0f766e;}} QPushButton:disabled{{color:#64748b;}} QSpinBox,QPlainTextEdit{{background:#0b1220;color:#e2e8f0;border:1px solid #334155;padding:4px;}} QProgressBar{{background:#17243b;color:white;max-height:{bar}px;}} QProgressBar::chunk{{background:#14b8a6;}} QMenu{{background:#0b1220;border:1px solid #334155;}} QMenu::item:selected{{background:#1e3a5f;}} QTabWidget::pane{{border:1px solid #334155;}} QTabBar::tab{{background:#17243b;padding:5px 12px;}} QTabBar::tab:selected{{background:#1e3a5f;color:#67e8f9;}}"""

__all__ = ['Config', 'Simulation', 'Engine', 'Window', 'main', 'export_records', 'format_detail', 'speed_profile']


def speed_profile(value):
    value = max(1, min(60, int(value)))
    return max(16, 500 // value), 1 if value <= 30 else 1 + (value - 30) // 6


def _cell(value):
    if isinstance(value, str) and value.startswith(UNSAFE_CELL):
        return "'" + value
    return value


def export_records(records, filename):
    path = Path(filename)
    if path.suffix.lower() != '.csv':
        path = path.with_suffix('.csv')
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('Destino inválido')
    if not records:
        raise ValueError('Sem dados')
    columns = list(records[0].keys())
    fd, temp = tempfile.mkstemp(prefix='.dal-export-', suffix='.csv', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([_cell(c) for c in columns])
            for row in records:
                writer.writerow([_cell(row.get(c, '')) for c in columns])
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return path


def format_detail(detail):
    if not detail:
        return 'Clique em um organismo para inspecionar seu estado sintético.'
    mind, life, seen = detail['mente'], detail['vida'], detail['observação']
    vocabulary = ', '.join(f'{topic}=«{word}»' for topic, word in mind['vocabulário'].items()) or '—'
    memories = '; '.join(f"t{m['tick']} {m['kind']}→{m['partner']} ({m['reward']:.2f})" for m in life['memórias_recentes']) or '—'
    status = 'percebe o observador' if seen['percebe_observador'] else 'não percebe o observador'
    awareness = f"estado {seen['estado_estimado']}, certeza {seen['confiança_metacognitiva']:.0%}, excitação {seen['excitação']:.2f}, habituação {seen['habituação']:.2f}"
    return (f"ORGANISMO SINTÉTICO #{detail['agente']}\n"
            f"Opinião {detail['opinião']:+.3f} · conexões {detail['conexões']} · grupos {detail['grupos']}\n"
            f"Intenção: {life['intenção']} · fome {life['fome']:.2f} · fadiga {life['fadiga']:.2f} · independência {life['independência']:.2f}\n"
            f"Energia {mind['energia']:.2f} · confiança {mind['confiança']:.2f} · experiências {mind['experiências']}\n"
            f"Estratégia dominante: {mind['estratégia_dominante']}\n"
            f"Vocabulário: {vocabulary}\n"
            f"Observação: crença {seen['crença_de_ser_observado']:.0%} ({status}; perfil {seen['perfil_sob_observação']})\n"
            f"Consciência modelada: {awareness}\n"
            f"Memórias: {memories}")


class Engine(QObject):
    frame_ready = Signal()
    completed = Signal(int, object)
    failed = Signal(str)
    state_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._observation = Observation()
        self._selected = None
        self._latest = None
        self._pending = False
        self.simulation = None
        self.run_id = 0
        self.timer = None
        self.interval = 40
        self.batch = 1

    @Slot()
    def setup(self):
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)

    def set_observation(self, observation):
        with self._lock:
            self._observation = observation if isinstance(observation, Observation) else Observation()

    def select(self, agent):
        with self._lock:
            self._selected = agent if isinstance(agent, int) and agent >= 0 else None

    def take_snapshot(self):
        with self._lock:
            latest, self._latest, self._pending = self._latest, None, False
        return latest

    def _stop(self):
        if self.timer is not None:
            self.timer.stop()

    @Slot(object, int, object)
    def start_run(self, config, run_id, inherited=None):
        self._stop()
        self.simulation = None
        with self._lock:
            self._latest, self._pending = None, False
        try:
            self.simulation = Simulation(config, inherited)
        except ValueError:
            self.failed.emit('Parâmetros inválidos.')
            return
        except Exception:
            logging.exception('Falha ao criar simulação')
            self.failed.emit(GENERIC_FAILURE)
            return
        self.run_id = run_id
        self._publish(force=True)
        self.timer.start(self.interval)
        self.state_changed.emit('running')

    @Slot()
    def pause(self):
        self._stop()
        if self.simulation is not None:
            self.state_changed.emit('paused')

    @Slot()
    def resume(self):
        if self.simulation is not None and self.simulation.tick < self.simulation.config.steps:
            self.timer.start(self.interval)
            self.state_changed.emit('running')

    @Slot()
    def step_once(self):
        self._stop()
        if self._advance():
            self._publish(force=True)
            if self.simulation is not None and self.simulation.tick < self.simulation.config.steps:
                self.state_changed.emit('paused')

    @Slot(int, int)
    def configure_speed(self, interval, batch):
        self.interval = max(16, min(1000, int(interval)))
        self.batch = max(1, min(10, int(batch)))
        if self.timer is not None and self.timer.isActive():
            self.timer.setInterval(self.interval)

    @Slot()
    def shutdown(self):
        self._stop()
        self.simulation = None

    @Slot()
    def _tick(self):
        deadline = time.perf_counter() + FRAME_BUDGET
        for _ in range(self.batch):
            if not self._advance() or time.perf_counter() > deadline:
                break
        self._publish()

    def _advance(self):
        simulation = self.simulation
        if simulation is None or simulation.tick >= simulation.config.steps:
            self._stop()
            return False
        with self._lock:
            observation = self._observation
        try:
            simulation.set_observation(observation)
            simulation.step(simulation.tick + 1)
        except Exception:
            logging.exception('Falha no passo da simulação')
            self._stop()
            self.simulation = None
            self.failed.emit(GENERIC_FAILURE)
            return False
        if simulation.tick >= simulation.config.steps:
            self._stop()
            self._publish(force=True)
            try:
                payload = {'episode': simulation.episode(), 'records': list(simulation.records), 'knowledge': simulation.knowledge()}
            except Exception:
                logging.exception('Falha ao consolidar episódio')
                payload = {'episode': None, 'records': list(simulation.records), 'knowledge': None}
            self.completed.emit(self.run_id, payload)
            self.state_changed.emit('finished')
        return True

    def _publish(self, force=False):
        simulation = self.simulation
        if simulation is None:
            return
        with self._lock:
            if self._pending and not force:
                return
            selected = self._selected
        try:
            snapshot = simulation.snapshot(selected)
        except Exception:
            logging.exception('Falha ao gerar quadro')
            return
        with self._lock:
            self._latest = (self.run_id, snapshot)
            self._pending = True
        self.frame_ready.emit()


class _UpdateSignals(QObject):
    done = Signal(object)


class UpdateTask(QRunnable):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.signals = _UpdateSignals()

    def run(self):
        try:
            result = self.manager.scan_inbox()
        except Exception:
            logging.exception('Falha ao verificar atualizações')
            result = {'status': 'error'}
        self.signals.done.emit(result)


UPDATE_MESSAGES = {
    'none': 'Nenhuma atualização na caixa de entrada.',
    'incomplete': 'Pacote de atualização incompleto ou ainda sendo copiado; tente novamente em instantes.',
    'rejected': 'Pacote de atualização rejeitado pela verificação de integridade e colocado em quarentena.',
    'locked': 'Atualizações suspensas nesta sessão após eventos de segurança repetidos.',
    'error': 'Não foi possível verificar atualizações agora.',
}


class Window(QMainWindow):
    request_start = Signal(object, int, object)
    request_pause = Signal()
    request_resume = Signal()
    request_step = Signal()
    request_speed = Signal(int, int)
    request_shutdown = Signal()

    def __init__(self, memory, journal=None, manager=None, notice=None, release_active=False, settings=None, trusted_pack_keys=frozenset()):
        super().__init__()
        self.memory = memory
        self.journal = journal or SecurityJournal()
        self.manager = manager
        self.release_active = release_active
        self.records = None
        self.run_id = 0
        self.population_value = 0
        self.last_step = -1
        self.running = False
        self.observer_enabled = True
        self.selected = None
        self.pool = QThreadPool.globalInstance()
        self.update_tasks = []
        self.stopped = False
        self.settings = settings if settings is not None else QSettings('DigitalAnthropologyLab', 'DAL')
        self.inherit_enabled = True
        self.trusted_pack_keys = trusted_pack_keys
        self.panel_window = None
        self.profile = None
        self.setWindowTitle(f'{__version__} | Humanoid Society Simulator')
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 4, 8, 4)
        self.header = QLabel('HUMANOID SOCIETY SIMULATOR')
        self.header.setObjectName('header')
        layout.addWidget(self.header)
        self.population = QSpinBox()
        self.population.setRange(5, 2000)
        self.population.setValue(120)
        self.steps = QSpinBox()
        self.steps.setRange(1, 10000)
        self.steps.setValue(400)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 60)
        self.speed.setValue(12)
        self.speed.setMinimumWidth(90)
        self.speed.setMaximumWidth(160)
        self.toolbar_widget = QWidget()
        self.toolbar = FlowLayout(self.toolbar_widget)
        flow_policy(self.toolbar_widget)
        for title, widget in [('AGENTES', self.population), ('PASSOS', self.steps), ('SEMENTE', self.seed), ('VELOCIDADE', self.speed)]:
            group = QWidget()
            group_layout = QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 4, 0)
            group_layout.addWidget(QLabel(title))
            group_layout.addWidget(widget)
            self.toolbar.addWidget(group)
        self.start = QPushButton('Iniciar / Reiniciar')
        self.pause = QPushButton('Pausar')
        self.once = QPushButton('Avançar 1')
        self.links = QPushButton('Alternar conexões')
        self.conversations = QPushButton('Conversas: ON')
        self.observer_button = QPushButton('Observador: ON')
        self.export = QPushButton('Exportar CSV')
        self.recommend = QPushButton('Memória: sugerir')
        self.update_button = QPushButton('Atualizações')
        self.knowledge_button = QToolButton()
        self.knowledge_button.setText('Conhecimento')
        self.knowledge_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.knowledge_button)
        self.inherit_action = QAction('Herdar aprendizado anterior', menu)
        self.inherit_action.setCheckable(True)
        self.inherit_action.setChecked(True)
        self.inherit_action.toggled.connect(self.set_inherit)
        menu.addAction(self.inherit_action)
        menu.addAction('Importar pacote da comunidade…', self.import_knowledge)
        menu.addAction('Exportar meu pacote…', self.export_knowledge)
        menu.addAction('Remover pacotes importados', self.clear_packs)
        self.knowledge_button.setMenu(menu)
        self.panel_button = QPushButton('Painel no 2º monitor')
        self.panel_button.setCheckable(True)
        self.panel_button.setVisible(False)
        for button in (self.start, self.pause, self.once, self.links, self.conversations, self.observer_button, self.export, self.recommend, self.knowledge_button, self.update_button, self.panel_button):
            self.toolbar.addWidget(button)
        layout.addWidget(self.toolbar_widget)
        self.notice = QLabel(notice or '')
        self.notice.setObjectName('notice')
        self.notice.setWordWrap(True)
        self.notice.setVisible(bool(notice))
        layout.addWidget(self.notice)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.network = LivingView()
        self.network.setMinimumSize(320, 240)
        self.split.addWidget(self.network)
        self.side = QWidget()
        self.side.setMinimumWidth(260)
        self.side_layout = QVBoxLayout(self.side)
        self.side_layout.setContentsMargins(4, 4, 4, 4)
        self.status = QLabel('Aguardando experimento')
        self.metrics = QLabel('Agentes: —\nConexões: —\nAlianças: —\nGrupos: —\nComponentes: —\nPolarização: —')
        self.metrics.setObjectName('metrics')
        self.metrics.setWordWrap(True)
        self.detail = QLabel(format_detail(None))
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(400)
        self.dialogue_title = QLabel('TELEMETRIA E DIÁLOGOS')
        self.tabs = QTabWidget()
        self.tabbed = None
        self.side_layout.addWidget(self.status)
        self.arrange_side(False)
        self.split.addWidget(self.side)
        layout.addWidget(self.split, 1)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.footer = QLabel('População 100% sintética. "Perceber o observador" é uma crença modelada a partir da sua navegação na tela, não consciência.')
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)
        self.thread = QThread(self)
        self.engine = Engine()
        self.engine.moveToThread(self.thread)
        self.thread.started.connect(self.engine.setup)
        self.thread.finished.connect(self.engine.deleteLater)
        self.request_start.connect(self.engine.start_run)
        self.request_pause.connect(self.engine.pause)
        self.request_resume.connect(self.engine.resume)
        self.request_step.connect(self.engine.step_once)
        self.request_speed.connect(self.engine.configure_speed)
        self.request_shutdown.connect(self.engine.shutdown)
        self.engine.frame_ready.connect(self.on_frame)
        self.engine.completed.connect(self.on_completed)
        self.engine.failed.connect(self.on_failed)
        self.engine.state_changed.connect(self.on_state)
        self.thread.start()
        self.observe_timer = QTimer(self)
        self.observe_timer.setInterval(120)
        self.observe_timer.timeout.connect(self.push_observation)
        self.observe_timer.start()
        self.start.clicked.connect(self.restart)
        self.pause.clicked.connect(self.toggle)
        self.once.clicked.connect(self.request_step.emit)
        self.export.clicked.connect(self.export_csv)
        self.recommend.clicked.connect(self.suggest)
        self.update_button.clicked.connect(self.check_updates)
        self.network.agent_selected.connect(self.inspect)
        self.links.clicked.connect(self.toggle_links)
        self.conversations.clicked.connect(self.toggle_conversations)
        self.observer_button.clicked.connect(self.toggle_observer)
        self.speed.valueChanged.connect(self.adjust_speed)
        self.export.setEnabled(False)
        self.update_button.setEnabled(manager is not None)
        if manager is None:
            self.update_button.setToolTip('Atualizações desativadas: chave pública de assinatura não configurada.')
        self.panel_button.toggled.connect(self.toggle_panel_window)
        self.apply_display_profile(initial=True)
        application = QGuiApplication.instance()
        if application is not None:
            application.screenAdded.connect(self.on_screens_changed)
            application.screenRemoved.connect(self.on_screens_changed)
        self.adjust_speed()
        if manager is not None:
            QTimer.singleShot(2500, self.check_updates)
            if release_active:
                QTimer.singleShot(5000, self.mark_healthy)

    def show_notice(self, text):
        self.notice.setText(text)
        self.notice.setVisible(True)

    def adjust_speed(self):
        self.request_speed.emit(*speed_profile(self.speed.value()))

    def restart(self):
        try:
            config = Config(self.population.value(), self.steps.value(), self.seed.value()).validate()
        except ValueError:
            self.show_notice('Parâmetros inválidos.')
            return
        self.run_id += 1
        self.population_value = config.population
        self.records = None
        self.last_step = -1
        self.selected = None
        self.engine.select(None)
        self.export.setEnabled(False)
        inherited = None
        message = 'Ecossistema inicializado. Agentes sintéticos em atividade.'
        if self.inherit_enabled:
            try:
                inherited = knowledge.effective_knowledge(self.memory)
            except (ValueError, StoreError):
                logging.exception('Falha ao carregar conhecimento')
                inherited = None
            if inherited is not None:
                message += f' Herdando aprendizado de {inherited.episodes} episódio(s) ({"verificado" if inherited.verified else "não verificado"}).'
        self.output.setPlainText(message)
        self.detail.setText(format_detail(None))
        self.request_start.emit(config, self.run_id, inherited)

    def toggle(self):
        if self.run_id == 0:
            self.restart()
        elif self.running:
            self.request_pause.emit()
        else:
            self.request_resume.emit()

    @Slot(str)
    def on_state(self, state):
        self.running = state == 'running'
        self.pause.setText('Pausar' if self.running else 'Continuar')
        if state == 'finished':
            self.status.setText('● EXPERIMENTO CONCLUÍDO')
        elif state == 'paused':
            self.status.setText(self.status.text().replace('EXECUTANDO', 'PAUSADO'))

    def push_observation(self):
        if not self.population_value:
            return
        if not self.observer_enabled or (not self.isActiveWindow() and not self.network.underMouse()):
            self.engine.set_observation(Observation())
            return
        observation = Observation.sanitize(True, self.network.current_view(), self.network.cursor_scene, self.selected, self.population_value)
        self.engine.set_observation(observation)

    @Slot()
    def on_frame(self):
        item = self.engine.take_snapshot()
        if item is None:
            return
        run_id, snapshot = item
        if run_id != self.run_id:
            return
        self.network.update_snapshot(snapshot)
        if snapshot.step != self.last_step:
            self.last_step = snapshot.step
            for event in snapshot.conversations[:3]:
                self.output.appendPlainText(f'#{event.speaker} → #{event.listener} [{event.act}]: {event.text}')
                self.output.appendPlainText(f'    ↳ #{event.listener}: {event.reply}')
            if snapshot.step and snapshot.step % 10 == 0:
                m = snapshot.metrics
                self.output.appendPlainText(f"t={snapshot.step:04d}  arestas={m['edges']}  alianças={m['alliances']}  observados={m['observed_agents']}  entendimento={m['understanding']:.0%}")
        m = snapshot.metrics
        if snapshot.step:
            self.status.setText(f"● {'EXECUTANDO' if self.running else 'PAUSADO'}   |   Ciclo {snapshot.step} / {snapshot.total_steps}")
            self.metrics.setText(f"Agentes: {snapshot.population}\nConexões: {m['edges']}\nAlianças: {m['alliances']}\nGrupos ativos: {m['groups']}\nComponentes: {m['components']}\nPolarização: {m['polarization']:.4f}\nDispersão espacial: {m['dispersion']:.1f}\nAgentes que se sentem observados: {m['observed_agents']}\nEntendimento lexical: {m['understanding']:.0%}\nCoerência do vocabulário: {m['lexicon_coherence']:.0%}\nAutonomia média: {m['autonomy']:.3f}\nInterações acumuladas: {m['interactions']}")
        self.progress.setValue(int(100 * snapshot.step / max(1, snapshot.total_steps)))
        if self.selected is not None and snapshot.detail is not None:
            self.detail.setText(format_detail(snapshot.detail))

    @Slot(int, object)
    def on_completed(self, run_id, payload):
        if run_id != self.run_id:
            return
        self.records = payload.get('records') or None
        self.export.setEnabled(bool(self.records))
        learned = payload.get('knowledge')
        if learned is not None:
            try:
                knowledge.record_episode(self.memory, learned)
            except (ValueError, StoreError, OSError):
                logging.exception('Falha ao registrar conhecimento')
        episode = payload.get('episode')
        if episode is None:
            self.output.appendPlainText('Episódio concluído, mas não pôde ser consolidado.')
            return
        try:
            self.memory.add(episode)
            message = 'Episódio registrado no armazenamento local.' if getattr(self.memory, 'persistent', False) else 'Episódio mantido apenas nesta sessão (armazenamento indisponível).'
        except (ValueError, OSError, StoreError):
            logging.exception('Falha ao persistir episódio')
            message = 'Falha ao persistir episódio.'
        self.output.appendPlainText(message)

    @Slot(str)
    def on_failed(self, message):
        self.running = False
        self.pause.setText('Continuar')
        self.status.setText('● FALHA')
        self.show_notice(message)

    def toggle_conversations(self):
        self.network.show_conversations = not self.network.show_conversations
        self.conversations.setText('Conversas: ON' if self.network.show_conversations else 'Conversas: OFF')

    def toggle_links(self):
        self.network.show_links = not self.network.show_links

    def toggle_observer(self):
        self.observer_enabled = not self.observer_enabled
        self.observer_button.setText('Observador: ON' if self.observer_enabled else 'Observador: OFF')
        self.push_observation()

    def inspect(self, agent):
        self.selected = agent
        self.engine.select(agent)
        self.push_observation()

    def suggest(self):
        try:
            suggestion = EpisodeLearner(self.memory).recommend()
        except (ValueError, StoreError):
            logging.exception('Falha ao consultar memória')
            suggestion = None
        if suggestion is None:
            self.output.appendPlainText('Memória insuficiente para recomendações.')
            return
        config = suggestion['config']
        self.population.setValue(int(config['population']))
        self.steps.setValue(int(config['steps']))
        self.seed.setValue(int(config['seed']))
        self.output.appendPlainText(f"Parâmetros sugeridos a partir de {suggestion['samples']} episódio(s) anteriores (não validados).")

    def check_updates(self):
        if self.manager is None or self.update_tasks:
            return
        task = UpdateTask(self.manager)
        task.setAutoDelete(False)
        task.signals.done.connect(self.on_update_result)
        self.update_tasks.append(task)
        self.update_button.setEnabled(False)
        self.pool.start(task)

    @Slot(object)
    def on_update_result(self, result):
        self.update_tasks.clear()
        self.update_button.setEnabled(self.manager is not None and not self.journal.blocked())
        status = result.get('status') if isinstance(result, dict) else 'error'
        if status == 'staged':
            self.show_notice(f"Atualização {result['version']} verificada e preparada. Ela será aplicada na próxima inicialização, sem interromper esta sessão.")
        elif status in UPDATE_MESSAGES:
            if status != 'none':
                self.show_notice(UPDATE_MESSAGES[status])
            self.output.appendPlainText(UPDATE_MESSAGES[status])

    def mark_healthy(self):
        try:
            self.manager.mark_healthy()
        except (OSError, ValueError):
            logging.exception('Falha ao registrar inicialização saudável')

    def export_csv(self):
        if not self.records:
            return
        filename, _ = QFileDialog.getSaveFileName(self, 'Exportar CSV', 'experimento.csv', 'CSV (*.csv)')
        if not filename:
            return
        try:
            export_records(self.records, filename)
        except (OSError, ValueError):
            logging.exception('Falha ao exportar CSV')
            QMessageBox.warning(self, 'Falha', 'Não foi possível exportar o CSV.')

    def screens(self):
        application = QGuiApplication.instance()
        if application is None:
            return [], 0
        return layout.screens_from_qt(application)

    def apply_display_profile(self, initial=False):
        screens, primary = self.screens()
        handle = self.windowHandle()
        if handle is not None and handle.screen() is not None and screens:
            current = handle.screen().availableGeometry()
            for index, info in enumerate(screens):
                if (info.x, info.y) == (current.x(), current.y()):
                    primary = index
        self.profile = layout.classify(screens, primary)
        font = self.profile.font_px
        compact = self.profile.kind == 'compact'
        self.setStyleSheet(STYLE.format(font=font, header=font + (4 if compact else 8), pad=4 if compact else 10, metrics=font + 1, button=4 if compact else 7, bar=14 if compact else 20))
        self.header.setVisible(not compact)
        self.panel_button.setVisible(self.profile.secondary is not None)
        if self.profile.secondary is None and self.panel_window is not None:
            self.panel_button.setChecked(False)
        if initial:
            self.restore_geometry(screens)
        self.reflow()

    def arrange_side(self, tabbed):
        if tabbed == self.tabbed:
            return
        self.tabbed = tabbed
        pages = ((self.metrics, 'Métricas'), (self.detail, 'Organismo'), (self.output, 'Diálogos'))
        while self.tabs.count():
            self.tabs.removeTab(0)
        for widget in (self.metrics, self.detail, self.dialogue_title, self.output, self.tabs):
            self.side_layout.removeWidget(widget)
        if tabbed:
            for widget, title in pages:
                self.tabs.addTab(widget, title)
            self.dialogue_title.hide()
            self.side_layout.addWidget(self.tabs, 1)
            self.tabs.show()
        else:
            self.tabs.hide()
            self.side_layout.addWidget(self.metrics)
            self.side_layout.addWidget(self.detail)
            self.side_layout.addWidget(self.dialogue_title)
            self.side_layout.addWidget(self.output, 1)
            for widget in (self.metrics, self.detail, self.dialogue_title, self.output):
                widget.show()

    def reflow(self):
        arrangement = layout.arrangement(self.width())
        stacked = arrangement['stacked'] and self.panel_window is None
        self.arrange_side(stacked or (self.height() < 760 and self.panel_window is None))
        self.toolbar_widget.updateGeometry()
        self.split.setOrientation(Qt.Orientation.Vertical if stacked else Qt.Orientation.Horizontal)
        if self.panel_window is None:
            total = max(1, self.split.height() if stacked else self.split.width())
            fraction = .38 if stacked else (self.profile.side_fraction if self.profile and self.profile.side_fraction else .3)
            self.split.setSizes([int(total * (1 - fraction)), int(total * fraction)])
        self.footer.setVisible(not stacked)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.profile is not None:
            self.reflow()

    def showEvent(self, event):
        super().showEvent(event)
        handle = self.windowHandle()
        if handle is not None and not getattr(self, '_screen_hooked', False):
            handle.screenChanged.connect(lambda _screen: self.apply_display_profile())
            self._screen_hooked = True

    @Slot()
    def on_screens_changed(self, *args):
        QTimer.singleShot(0, self.apply_display_profile)

    def _geometry_key(self):
        screens, _ = self.screens()
        return 'geometry/' + layout.signature(screens).replace('/', '_')

    def restore_geometry(self, screens):
        saved = self.settings.value(self._geometry_key())
        rect = None
        if isinstance(saved, (list, tuple)) and len(saved) == 4:
            rect = layout.safe_geometry(saved, screens)
        if rect is None and self.profile is not None:
            rect = layout.initial_geometry(self.profile)
        if rect is not None:
            self.setGeometry(QRect(*rect))
        if self.profile is not None and self.profile.kind == 'compact':
            self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

    def save_geometry(self):
        try:
            rect = self.normalGeometry() if self.isMaximized() else self.geometry()
            self.settings.setValue(self._geometry_key(), [rect.x(), rect.y(), rect.width(), rect.height()])
            self.settings.sync()
        except RuntimeError:
            logging.exception('Falha ao salvar geometria')

    def toggle_panel_window(self, enabled):
        if enabled and self.panel_window is None and self.profile is not None and self.profile.secondary is not None:
            window = QWidget(None, Qt.WindowType.Window)
            window.setWindowTitle('Digital Anthropology Lab | Painel')
            window.setStyleSheet(self.styleSheet())
            box = QVBoxLayout(window)
            box.setContentsMargins(6, 6, 6, 6)
            box.addWidget(self.side)
            window.setGeometry(QRect(*layout.panel_geometry(self.profile.secondary)))
            window.closeEvent = lambda event: (self.panel_button.setChecked(False), event.accept())
            self.panel_window = window
            window.show()
            self.reflow()
        elif not enabled and self.panel_window is not None:
            self.attach_panel()

    def attach_panel(self):
        window, self.panel_window = self.panel_window, None
        if window is None:
            return
        self.split.addWidget(self.side)
        self.side.show()
        window.closeEvent = lambda event: event.accept()
        window.close()
        window.deleteLater()
        self.reflow()

    def set_inherit(self, enabled):
        self.inherit_enabled = bool(enabled)

    def import_knowledge(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Importar pacote de conhecimento', '', 'Pacote JSON (*.json)')
        if filename:
            self.import_knowledge_file(filename)

    def import_knowledge_file(self, filename):
        path = Path(filename)
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > knowledge.MAX_PACK_BYTES:
                raise knowledge.KnowledgeError('Arquivo inválido', 'file')
            source, item = knowledge.import_pack(self.memory, path.read_bytes(), self.trusted_pack_keys)
        except knowledge.KnowledgeError as error:
            self.journal.record('knowledge_rejected', 'medium', error.cause)
            self.show_notice('Pacote de conhecimento rejeitado: formato ou valores inválidos.')
            return False
        except (ValueError, OSError, StoreError):
            logging.exception('Falha ao importar pacote')
            self.show_notice('Não foi possível importar o pacote (limite de pacotes ou erro de armazenamento).')
            return False
        label = 'verificado' if item.verified else 'não verificado (peso reduzido)'
        self.show_notice(f'Pacote importado ({label}): {item.episodes} episódio(s). Será usado na próxima execução.')
        return True

    def export_knowledge(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Exportar pacote de conhecimento', 'dialogue-pack.json', 'Pacote JSON (*.json)')
        if filename:
            self.export_knowledge_file(filename)

    def export_knowledge_file(self, filename):
        path = Path(filename)
        if path.suffix.lower() != '.json':
            path = path.with_suffix('.json')
        try:
            if path.is_symlink():
                raise ValueError('Destino inválido')
            document = knowledge.export_document(self.memory, __version__)
            fd, temp = tempfile.mkstemp(prefix='.dal-pack-', suffix='.json', dir=path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                    json.dump(document, handle, sort_keys=True, indent=1)
                os.replace(temp, path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        except knowledge.KnowledgeError:
            self.show_notice('Ainda não há aprendizado local para exportar. Conclua ao menos um experimento.')
            return None
        except (ValueError, OSError):
            logging.exception('Falha ao exportar pacote')
            QMessageBox.warning(self, 'Falha', 'Não foi possível exportar o pacote.')
            return None
        self.show_notice('Pacote exportado. Ele contém apenas estatísticas agregadas, sem diálogos nem memórias.')
        return path

    def clear_packs(self):
        removed = 0
        for source, _ in self.memory.knowledge_items():
            if source != 'local' and self.memory.delete_knowledge(source):
                removed += 1
        self.show_notice(f'{removed} pacote(s) importado(s) removido(s).')
        return removed

    def shutdown(self):
        if self.stopped:
            return
        self.stopped = True
        self.save_geometry()
        if self.panel_window is not None:
            self.attach_panel()
        self.observe_timer.stop()
        self.request_shutdown.emit()
        self.thread.quit()
        if not self.thread.wait(5000):
            logging.error('Thread de simulação não finalizou a tempo')
        self.pool.waitForDone(3000)
        try:
            self.memory.close()
        except Exception:
            logging.exception('Falha ao fechar armazenamento')

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def _install_hooks():
    def handle(kind, value, trace):
        logging.getLogger('dal').critical('Exceção não tratada', exc_info=(kind, value, trace))

    def handle_thread(args):
        logging.getLogger('dal').critical('Exceção não tratada em thread', exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = handle
    threading.excepthook = handle_thread


def open_store(journal):
    try:
        return LocalStore(BASE / 'episodes-v3.secure'), None
    except StoreError as error:
        journal.record('store_open', 'high' if error.cause == 'authentication' else 'medium', error.cause)
        return VolatileStore(), 'O armazenamento local não pôde ser validado. A sessão continua sem salvar episódios; os arquivos originais foram preservados sem alteração.'
    except OSError:
        logging.exception('Falha de E/S ao abrir armazenamento')
        return VolatileStore(), 'O armazenamento local está inacessível (permissões ou disco). A sessão continua sem salvar episódios.'


def open_updates(journal, key_path):
    key = updates.load_trusted_key(key_path)
    if key is None:
        return None
    try:
        return updates.UpdateManager(BASE, __version__, key, journal)
    except (OSError, ValueError):
        logging.exception('Falha ao iniciar gerenciador de atualizações')
        return None


def main(argv=None, release_active=False, key_path=None):
    BASE.mkdir(parents=True, exist_ok=True)
    logsafe.configure(BASE)
    _install_hooks()
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    journal = SecurityJournal()
    memory, notice = open_store(journal)
    manager = open_updates(journal, Path(key_path) if key_path else Path(__file__).resolve().parent / 'update_key.pub')
    pack_keys = knowledge.load_trusted_keys(Path(__file__).resolve().parent / 'community_keys.txt')
    window = Window(memory, journal, manager, notice, release_active, trusted_pack_keys=pack_keys)
    app.aboutToQuit.connect(window.shutdown)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
