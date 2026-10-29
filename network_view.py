import networkx as nx
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush, QPen, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene

class NetworkView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumHeight(320)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def show_graph(self, graph, memberships, seed=42):
        scene = self.scene()
        scene.clear()
        if not graph.number_of_nodes():
            return
        sampled = sorted(graph.nodes())[:350]
        subgraph = graph.subgraph(sampled)
        positions = nx.spring_layout(subgraph, seed=seed, iterations=35)
        coords = {n: (float(p[0]) * 330, float(p[1]) * 260) for n, p in positions.items()}
        for a, b, data in subgraph.edges(data=True):
            x1, y1 = coords[a]
            x2, y2 = coords[b]
            scene.addLine(x1, y1, x2, y2, QPen(QColor('#2c9b91' if data.get('alliance') else '#8493a5'), 1.1))
        palette = ['#3b82f6', '#ef8a47', '#9b59b6', '#24a88a', '#d45a8a', '#d1a735']
        for node in sampled:
            x, y = coords[node]
            group = min(memberships[node]) if memberships[node] else -1
            dot = scene.addEllipse(x-4, y-4, 8, 8, QPen(Qt.PenStyle.NoPen), QBrush(QColor(palette[group % len(palette)] if group >= 0 else '#64748b')))
            dot.setToolTip(f'Agente sintético {node} | grupos: {sorted(memberships[node])}')
        self.fitInView(scene.itemsBoundingRect().adjusted(-15, -15, 15, 15), Qt.AspectRatioMode.KeepAspectRatio)
