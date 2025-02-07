from PyQt6.QtCore import Qt, QRectF, QPoint
from PyQt6.QtGui import QPainter, QColor, QPainterPath
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from ShapeMode import ShapeMode
import math

class WeaveView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.designShapes = []
        self.begin = QPoint()
        self.end = QPoint()

        start = QPoint(5, 5)  # Top-left corner of the heart
        end = QPoint(1150, 500) # Bottom-right corner of the heart
        self.designShapes.append([start, end, ShapeMode.Heart, QColor(255, 0, 0, 255)])

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        self.draw_grid(painter)
        self.drawShapes(painter)

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

    def drawShapes(self, painter):
        for shape in self.designShapes:
            start, end, shape_type, color = shape
            painter.setBrush(color)

            if shape_type == ShapeMode.Heart:
                self.drawHeart(painter, start, end)

    def drawHeart(self, qp, start, end):
        drawpath = QPainterPath()
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())
        x_offset, y_offset = start.x() + width // 2, start.y() + height // 2

        # Scale factor to fit heart inside the bounding box
        scale_x = width / 32  
        scale_y = height / 32  

        # Start drawing the heart shape using parametric equations
        t = 0
        first_point = True
        while t <= 2 * math.pi:
            x = int(16 * math.sin(t) ** 3 * scale_x) + x_offset
            y = int(- (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) * scale_y) + y_offset
            
            if first_point:
                drawpath.moveTo(x, y)
                first_point = False
            else:
                drawpath.lineTo(x, y)
            t += 0.1
        qp.drawPath(drawpath)
