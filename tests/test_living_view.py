import math
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from app import Config, Simulation
from living_view import BUBBLES, Humanoid, LivingView


@pytest.mark.parametrize('watched', [False, True])
def test_humanoid_paints_without_exception(qapp, watched):
    agent = Humanoid(7)
    agent.watched = watched
    agent.selected_agent = watched
    image = QImage(96, 96, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.translate(48, 60)
    agent.paint(painter, None)
    painter.end()
    assert not image.isNull()


def test_living_view_snapshot_cycle(qapp):
    simulation = Simulation(Config(population=20, steps=6, seed=13))
    view = LivingView()
    view.resize(800, 600)
    view.initialize(simulation.snapshot())
    assert len(view.humanoids) == 20
    view.select_agent(3)
    assert view.selected_id == 3
    view.select_agent(999)
    assert view.selected_id == 3
    for t in range(1, 7):
        simulation.step(t)
        view.update_snapshot(simulation.snapshot(3))
    assert view.link_count == len(simulation.snapshot().edges)
    assert sum(b.isVisible() for b in view.bubbles) <= BUBBLES
    view.show_links = False
    view.show_conversations = False
    view.update_snapshot(simulation.snapshot())
    assert view.link_count == 0 and not any(b.isVisible() for b in view.bubbles)
    bigger = Simulation(Config(population=30, steps=2, seed=1))
    view.update_snapshot(bigger.snapshot())
    assert len(view.humanoids) == 30 and view.selected_id is None
    assert all(math.isfinite(v) for v in view.current_view())


def test_display_is_capped_for_large_populations(qapp):
    simulation = Simulation(Config(population=600, steps=1, seed=1))
    view = LivingView()
    view.initialize(simulation.snapshot())
    assert len(view.humanoids) == 400
