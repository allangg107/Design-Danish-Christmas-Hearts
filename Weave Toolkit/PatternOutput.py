from PyQt6.QtCore import Qt, QRectF, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPainterPath
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from ShapeMode import ShapeMode
import math

class WeaveView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.designShapes = []
        self.heartList = []
        self.begin = QPoint()
        self.end = QPoint()

        #start = QPoint(0, 0)  # Top-left corner of the heart
        #end = QPoint(1170, 530) # Bottom-right corner of the heart
        #self.heartList.append([start, end, ShapeMode.Heart, QColor(255, 0, 0, 255)])
        #self.drawHeartOutLine()

    def setShapes(self, shapes):
        self.designShapes = shapes
        self.viewport().update()
   
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        self.drawShapes(painter)
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

    def drawShapes(self, painter):
        #heart = self.heartList[0]
        #start, end, shape_type, color = heart
        #self.drawHeart(painter, start, end, color, isFilled=True)
        self.drawHeart(painter, None, None, None, isFilled=True, isOutline=True)
        for shape in self.designShapes:
            shape_type = shape[2]
            painter.setBrush(shape[3])
            if shape_type == ShapeMode.Square:
                painter.drawRect(QRect(shape[0], shape[1]))
            elif shape_type == ShapeMode.Circle:
                center = shape[0]
                radius = int((abs(center.x() - shape[1].x()) + abs(center.y() - shape[1].y())) / 2)
                painter.drawEllipse(center, radius, radius)
            elif shape_type == ShapeMode.Heart:
                self.drawHeart(painter, shape[0], shape[1], shape[3])

    def drawHeart(self, qp, start, end, color, isFilled = True, isOutline = False):
        if isOutline:
            start = QPoint(0, 0)  # Top-left corner of the heart
            end = QPoint(1170, 530) # Bottom-right corner of the heart
            color = QColor(255, 0, 0, 255) # Reassigns color to red
        if isFilled == False:
            qp.setPen(color)
            qp.setBrush(Qt.BrushStyle.NoBrush)
        else:
            qp.setBrush(color)
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

        #if extra_shapes:
            #heart_center_x = start.x() + end.x()/2 
            
            #for shape in extra_shapes:
                #shape_type, color = shape[2], shape[3]
                #qp.setBrush(color)

                #shape_width = width // 4  # Scale shapes to fit inside heart
                #shape_height = height // 4
                
                #shape_x = heart_center_x - shape_width // 2
                #shape_y = heart_center_y - shape_height // 2
                
                            
                
                #if shape_type == ShapeMode.Square:
                #    qp.drawRect(QRect(shape_x, shape_y, shape_width, shape_height))
                #elif shape_type == ShapeMode.Circle:
                #    qp.drawEllipse(QPoint(shape_x + shape_width // 2, shape_y + shape_height // 2), shape_width // 2, shape_height // 2)
                