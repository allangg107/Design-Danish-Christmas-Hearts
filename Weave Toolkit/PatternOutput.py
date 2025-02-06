from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene

class WeaveView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        self.draw_grid(painter)

    def draw_grid(self, painter):
        # Set grid line color and style
        painter.setPen(QColor(200, 200, 200))  # Light gray grid lines
        grid_size = 20  # Change as needed

        # Draw vertical lines
        for x in range(0, self.width(), grid_size):
            painter.drawLine(x, 0, x, self.height())

        # Draw horizontal lines
        for y in range(0, self.height(), grid_size):
            painter.drawLine(0, y, self.width(), y)
