import math
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QColor, QPen, QPainter, QPainterPath, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsObject, QGraphicsTextItem, QGraphicsPathItem

PALETTE = tuple(QColor(c) for c in ('#34d399', '#60a5fa', '#f472b6', '#fbbf24', '#a78bfa', '#2dd4bf'))
UNGROUPED = QColor('#94a3b8')
BUBBLES = 8
TOOLTIP_EVERY = 10


class Humanoid(QGraphicsObject):
    clicked = Signal(int)

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.phase = agent * 2.39996
        self.moving = 0.0
        self.color = QColor('#8b5cf6')
        self.selected_agent = False
        self.watched = False
        self.heading = 1.0
        self.setZValue(2)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCacheMode(QGraphicsObject.CacheMode.DeviceCoordinateCache)

    def boundingRect(self):
        return QRectF(-17, -50, 34, 61)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 95))
        painter.drawEllipse(QRectF(-11, 5, 22, 5))
        swing = math.sin(self.phase) * self.moving * 4.8
        opposite = -swing
        painter.setPen(QPen(QColor('#141e33'), 5.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(-4, -8), QPointF(-5 + swing, 6))
        painter.drawLine(QPointF(4, -8), QPointF(5 + opposite, 6))
        painter.setPen(QPen(QColor('#718096'), 2.1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(-5 + swing, 0), QPointF(-5 + swing, 6))
        painter.drawLine(QPointF(5 + opposite, 0), QPointF(5 + opposite, 6))
        painter.setPen(QPen(QColor('#19243b'), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(-8, -24), QPointF(-12 - swing, -10))
        painter.drawLine(QPointF(8, -24), QPointF(12 + swing, -10))
        painter.setPen(QPen(self.color.lighter(135), 2.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(-12 - swing, -15), QPointF(-12 - swing, -10))
        painter.drawLine(QPointF(12 + swing, -15), QPointF(12 + swing, -10))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#111827'))
        painter.drawRoundedRect(QRectF(-8.5, -27, 17, 20), 4, 4)
        painter.setBrush(self.color.darker(125))
        painter.drawRoundedRect(QRectF(-7, -26, 14, 17), 3, 3)
        armor = QPainterPath()
        armor.moveTo(-7, -26)
        armor.lineTo(0, -22)
        armor.lineTo(7, -26)
        armor.lineTo(5, -16)
        armor.lineTo(0, -12)
        armor.lineTo(-5, -16)
        armor.closeSubpath()
        painter.setBrush(QColor('#202b43'))
        painter.drawPath(armor)
        painter.setPen(QPen(self.color.lighter(160), 1.15))
        painter.drawLine(QPointF(-5, -23), QPointF(-2, -18))
        painter.drawLine(QPointF(5, -23), QPointF(2, -18))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#5eead4'))
        painter.drawEllipse(QRectF(-2, -20, 4, 4))
        painter.setBrush(QColor('#0b1220'))
        painter.drawRoundedRect(QRectF(-6, -38, 12, 13), 4, 4)
        painter.setBrush(QColor('#65c88a'))
        painter.drawRoundedRect(QRectF(-5, -37, 10, 11), 3, 3)
        painter.setPen(QPen(QColor('#1a684c'), .85))
        painter.drawLine(QPointF(-3, -35), QPointF(-1, -32))
        painter.drawLine(QPointF(3, -35), QPointF(1, -32))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#0b2331'))
        painter.drawRoundedRect(QRectF(-4.5, -33.5, 9, 3), 1.2, 1.2)
        painter.setBrush(QColor('#b9fff1'))
        painter.drawRect(QRectF(-3.5, -32.7, 2.5, .8))
        painter.drawRect(QRectF(1, -32.7, 2.5, .8))
        painter.setPen(QPen(QColor('#7cf7cb'), .8))
        painter.drawLine(QPointF(-4, -29), QPointF(-2, -28))
        painter.drawLine(QPointF(4, -29), QPointF(2, -28))
        if self.watched:
            painter.setPen(QPen(QColor('#fde68a'), 1.1))
            painter.setBrush(QColor('#1f2937'))
            painter.drawEllipse(QRectF(-5, -49, 10, 6))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor('#fde68a'))
            painter.drawEllipse(QRectF(-1.6, -47.6, 3.2, 3.2))
        if self.selected_agent:
            painter.setPen(QPen(QColor('#fef08a'), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(-16, -39, 32, 48))

    def mousePressEvent(self, event):
        self.clicked.emit(self.agent)
        super().mousePressEvent(event)


class LivingView(QGraphicsView):
    agent_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor('#0b1421')))
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.humanoids = {}
        self.bubbles = []
        self.links = None
        self.alliance_links = None
        self.link_count = 0
        self.frames = 0
        self.selected_id = None
        self.cursor_scene = None
        self.show_links = True
        self.show_conversations = True
        self.user_zoomed = False
        self.normal_pen = QPen(QColor(100, 116, 139, 40), .55)
        self.alliance_pen = QPen(QColor(94, 234, 212, 125), 1.1)

    def wheelEvent(self, event):
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        if .25 < self.transform().m11() * factor < 5:
            self.scale(factor, factor)
            self.user_zoomed = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.user_zoomed and self.links is not None:
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def reset_zoom(self):
        self.user_zoomed = False
        self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mouseMoveEvent(self, event):
        point = self.mapToScene(event.position().toPoint())
        self.cursor_scene = (float(point.x()), float(point.y()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.cursor_scene = None
        super().leaveEvent(event)

    def current_view(self):
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        return float(rect.left()), float(rect.top()), float(rect.right()), float(rect.bottom())

    def initialize(self, snapshot):
        scene = self.scene()
        scene.clear()
        self.humanoids.clear()
        self.bubbles.clear()
        self.selected_id = None
        self.frames = 0
        self.user_zoomed = False
        for index, point in enumerate(snapshot.resources):
            marker = scene.addEllipse(float(point[0] - 13), float(point[1] - 13), 26, 26, QPen(QColor('#38bdf8'), 2), QBrush(QColor('#0e7490')))
            marker.setZValue(-1)
            marker.setToolTip(f'Recurso sintético {index}')
        self.links = QGraphicsPathItem()
        self.links.setPen(self.normal_pen)
        self.links.setZValue(-3)
        self.alliance_links = QGraphicsPathItem()
        self.alliance_links.setPen(self.alliance_pen)
        self.alliance_links.setZValue(-2)
        scene.addItem(self.links)
        scene.addItem(self.alliance_links)
        for agent in range(len(snapshot.positions)):
            person = Humanoid(agent)
            person.clicked.connect(self.select_agent)
            scene.addItem(person)
            self.humanoids[agent] = person
        for _ in range(BUBBLES):
            bubble = QGraphicsTextItem()
            bubble.setDefaultTextColor(QColor('#e2fef8'))
            bubble.setTextWidth(170)
            bubble.setScale(.75)
            bubble.setZValue(8)
            bubble.setVisible(False)
            scene.addItem(bubble)
            self.bubbles.append(bubble)
        scene.setSceneRect(-520, -520, 1040, 1040)
        self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.update_snapshot(snapshot)

    def select_agent(self, agent):
        if agent not in self.humanoids:
            return
        if self.selected_id in self.humanoids:
            self.humanoids[self.selected_id].selected_agent = False
            self.humanoids[self.selected_id].update()
        self.selected_id = agent
        self.humanoids[agent].selected_agent = True
        self.humanoids[agent].update()
        self.agent_selected.emit(agent)

    def update_snapshot(self, snapshot):
        if len(snapshot.positions) != len(self.humanoids) or self.links is None:
            self.initialize(snapshot)
            return
        self.frames += 1
        tooltips = self.frames % TOOLTIP_EVERY == 1
        positions = snapshot.positions.tolist()
        speeds = snapshot.speeds.tolist()
        for agent, person in self.humanoids.items():
            x, y = positions[agent]
            person.setPos(x, y)
            moving = min(1., speeds[agent] / 1.5)
            group = snapshot.colors[agent]
            color = PALETTE[group % len(PALETTE)] if group >= 0 else UNGROUPED
            watched = bool(snapshot.watched[agent])
            changed = moving > .01 or person.moving > .01 or color != person.color or watched != person.watched
            person.phase += speeds[agent] * .18
            person.moving = moving
            person.color = color
            person.watched = watched
            if tooltips:
                person.setToolTip(f'Agente sintético {agent} | {snapshot.intents[agent]} | crença de ser observado {snapshot.belief[agent]:.0%}')
            if changed:
                person.update()
        self._draw_links(snapshot, positions)
        self.update_conversations(snapshot)

    def _draw_links(self, snapshot, positions):
        normal, alliance = QPainterPath(), QPainterPath()
        count = 0
        if self.show_links:
            for a, b, allied in snapshot.edges:
                path = alliance if allied else normal
                path.moveTo(positions[a][0], positions[a][1] - 8)
                path.lineTo(positions[b][0], positions[b][1] - 8)
                count += 1
        self.links.setPath(normal)
        self.alliance_links.setPath(alliance)
        self.link_count = count

    def update_conversations(self, snapshot):
        events = [e for e in snapshot.conversations if e.speaker in self.humanoids] if self.show_conversations else []
        positions = snapshot.positions
        for index, bubble in enumerate(self.bubbles):
            if index < len(events):
                event = events[index]
                bubble.setPlainText(event.text)
                x, y = positions[event.speaker]
                bubble.setPos(float(x) + 12, float(y) - 84)
                bubble.setVisible(True)
            else:
                bubble.setVisible(False)
