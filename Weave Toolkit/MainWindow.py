import sys
import math
import svgwrite
import cv2 as cv
import numpy as np
import copy

from svgpathtools import svg2paths, svg2paths2, wsvg, Line

from functools import partial

from ShapeMode import (
    ShapeMode
)

from PatternType import (
    PatternType
)

from SideType import (
    SideType
)
from PyQt6.QtSvg import (
    QSvgGenerator,
    QSvgRenderer
)

from PyQt6.QtCore import (
    QSize,
    Qt,
    QRectF,
    QRect,
    QPoint,
    QPointF,
    QEvent,
)

from PyQt6.QtWidgets import  (
    QApplication,
    QMainWindow,
    QPushButton,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QToolBar,
    QMenu,
    QLabel,
    QWidgetAction,
    QStackedLayout,
    QStackedWidget,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsProxyWidget,
    QSlider,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem
)

from PyQt6.QtGui import (
    QColor,
    QAction,
    QIcon,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
    QPainterPath,
    QPainterPathStroker,
    QImage,
    QTransform,
    QPolygon,
    QPolygonF
)

from VectorAlgo import (
    mainAlgorithmSvg
)

from VectorAlgoStencils import (
    snapShapeToClassicCuts
)

from VectorAlgoUtils import (
    pre_process_user_input,
    calculate_distance
)

from GuideWindow import (
    GuideWindow
)
from GlobalVariables import(
    getShapeMode,
    setShapeMode,
    getShapeColor,
    setShapeColor,
    getBackgroundColor,
    setBackgroundColor,
    getPenWidth,
    setPenWidth,
    getFilled,
    setFilled,
    getUserOutputSVGFileName,
    setUserOutputSVGFileName,
    getUserPreprocessedPattern,
    setUserPreprocessedPattern,
    getCurrentPatternType,
    setCurrentPatternType,
    setCurrentSideType,
    getCurrentSideType,
    getNumClassicLines,
    setNumClassicLines,
    setClassicIndicesLineDeleteList,
    getClassicIndicesLineDeleteList,
    setSymmetryLine
)

from ErrorHandling import (
    shapeNotTouchingSymmetrylineError
)



class DrawingWidget(QWidget):
    # Defining the initial state of the canvas
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.shapes = [] #Stores shapes
        self.free_form_points = []  # Store points for free form drawing
        self.cells = [] #Stores cells
        self.setGeometry(30,30,600,400)
        self.begin = QPoint()
        self.end = QPoint()
        self.drawing_mode = False
        self.show()


    # Draws the current canvas state
    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setBrush(getShapeColor())
        # Fill the background with light gray color
        qp.fillRect(self.rect(), Qt.GlobalColor.lightGray)
        shape_color = getShapeColor()

        width, height = self.width(), self.height()
        margin = 0
        x1, y1 = margin, margin
        x2, y2 = width - margin, height - margin
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Coordinates of the corners of the inner rotated square
        inner_coords = [
            (center_x, y1),
            (x2, center_y),
            (center_x, y2),
            (x1, center_y)
        ]

        self.drawRotatedSquareEffect(qp, inner_coords)

        if getCurrentPatternType() == PatternType.Classic:
            self.drawCheckerboard(qp, inner_coords)
            shape_color = self.flipSquareColor(shape_color)

        # Redraw all the previous shapes
        self.redrawAllShapes(qp)

        qp.setBrush(getShapeColor())
        qp.setPen(QPen(getShapeColor(), getPenWidth(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))

        # Draws the current shape as it is being created
        if self.begin != self.end:
            if (getShapeMode() == ShapeMode.Square):
                self.drawSquare(qp, self.begin, self.end, shape_color, 1, getFilled())
            elif (getShapeMode() == ShapeMode.Circle):
                self.drawCircle(qp, self.begin, self.end, shape_color, 1, getFilled())
            elif (getShapeMode() == ShapeMode.Heart):
                self.drawHeart(qp, self.begin, self.end, shape_color, 1, getFilled())
            elif (getShapeMode() == ShapeMode.Line):
                qp.drawLine(self.begin, self.end)
            elif getShapeMode() == ShapeMode.FreeForm:
                for free_form_point in range(len(self.free_form_points) - 1):
                    qp.drawLine(self.free_form_points[free_form_point], self.free_form_points[free_form_point + 1])
            elif getShapeMode() == ShapeMode.Semicircle:
                angle = self.calculateAngle(self.begin, self.end)
                self.drawSemicircle(qp, self.begin, self.end, shape_color, 2 , getFilled(), rotation_angle=angle)

        self.redrawBorder(qp, inner_coords)

    def drawRotatedSquareEffect(self, qp, inner_coords):
        pen = QPen(Qt.GlobalColor.black, 3)
        qp.setPen(pen)
        brush = QBrush(Qt.GlobalColor.lightGray)
        qp.setBrush(brush)

        # Drawing outer square
        qp.drawRect(inner_coords[3][0], inner_coords[0][1], inner_coords[1][0] - inner_coords[3][0], inner_coords[2][1] - inner_coords[0][1])

        # Drawing inner rotated square with selected background fill
        brush = QBrush(getBackgroundColor())
        qp.setBrush(brush)
        path = QPainterPath()
        path.moveTo(inner_coords[0][0], inner_coords[0][1])
        for point in inner_coords[1:]:
            path.lineTo(point[0], point[1])
        path.closeSubpath()
        qp.drawPath(path)

        # Draw the edges of the inner rotated square
        for i in range(len(inner_coords)):
            qp.drawLine(int(inner_coords[i][0]), int(inner_coords[i][1]),
                             int(inner_coords[(i+1) % len(inner_coords)][0]),
                             int(inner_coords[(i+1) % len(inner_coords)][1]))

        pen = QPen(getShapeColor(), getPenWidth())
        qp.setPen(pen)
        brush = QBrush(getShapeColor())
        qp.setBrush(brush)

    def flipSquareColor(self, shape_color):

        for polygon, cell_color in self.cells:
            if polygon.containsPoint(QPointF(self.end), Qt.FillRule.OddEvenFill):
                if cell_color == shape_color:
                    if shape_color == getBackgroundColor():
                        shape_color = getShapeColor()
                    else:
                        shape_color = getBackgroundColor()
                    break

        return shape_color

    # Redraws all the shapes, while removing the ones that are erased
    def redrawAllShapes(self, qp):
        for shape in self.shapes[:]:  # Use a copy of the list to avoid modification issues
            shape_type = shape[2]
            shape_color = getShapeColor()
            if getCurrentPatternType() == PatternType.Classic:
                shape_color = shape[3]
            qp.setBrush(shape_color) # set to shape[3] if we want to change color to stored shape color instead of global color
            qp.setPen(QPen(shape_color, getPenWidth(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin)) # pen width should needs to be saved in the shape list

            # if in eraser mode, removes shapes that contain the point clicked
            if getShapeMode() == ShapeMode.Eraser:
                point = self.begin
                if shape_type == ShapeMode.Square:
                    rect = QRect(shape[0], shape[1])
                    if rect.contains(point):
                        self.shapes.remove(shape)
                        # continue  # Skip drawing since it's erased

                elif shape_type == ShapeMode.Circle:
                    center = shape[0]
                    radius = int((abs(center.x() - shape[1].x()) + abs(center.y() - shape[1].y())) / 2)
                    distance = ((point.x() - center.x()) ** 2 + (point.y() - center.y()) ** 2) ** 0.5
                    if distance <= radius:
                        self.shapes.remove(shape)
                        # continue  # Skip drawing since it's erased

                elif shape_type == ShapeMode.Heart:
                    if self.heartContainsPoint(point, shape[0], shape[1]):
                        self.shapes.remove(shape)
                        # continue  # Skip drawing since it's erased

                elif shape_type == ShapeMode.Line:
                    if self.lineContainsPoint(point, shape[0], shape[1]):
                        self.shapes.remove(shape)
                        # continue # Skip drawing since it's erased

                elif shape_type == ShapeMode.Semicircle:
                    start = shape[0]
                    end = shape[1]

                    # Calculate the distance (size of the semicircle)
                    dx, dy, distance = self.calculateSemiDistance(start, end)
                    radius = distance / 2

                    # Get the center of the semicircle
                    center_x = (start.x() + end.x()) / 2
                    center_y = (start.y() + end.y()) / 2
                    # Check if the point is within the bounds of the semicircle's radius (ignoring rotation for now)
                    dist_to_center = math.hypot(point.x() - center_x, point.y() - center_y)

                    # Calculate the angle between the point and the center
                    angle = self.calculateAngle(QPoint(int(center_x), int(center_y)), point)

                    # Check if the point is within the semicircle (half-circle)
                    if dist_to_center <= radius and 0 <= angle <= 180:
                        # Point is inside the semicircle area, so remove the shape
                        self.shapes.remove(shape)

                elif shape_type == ShapeMode.FreeForm:
                    for free_form_point in range(len(shape[4]) - 1):
                        if self.lineContainsPoint(point, shape[4][free_form_point], shape[4][free_form_point + 1]):
                            self.shapes.remove(shape)
                            break  # Skip drawing since it's erased

            # Draw the shape if not in eraser mode
            if shape_type == ShapeMode.Square:
                self.drawSquare(qp, shape[0], shape[1], shape_color, shape[5], shape[6])

            elif shape_type == ShapeMode.Circle:
                self.drawCircle(qp, shape[0], shape[1], shape_color, shape[5], shape[6])

            elif shape_type == ShapeMode.Heart:
                self.drawHeart(qp, shape[0], shape[1], shape_color, shape[5], shape[6])

            elif shape_type == ShapeMode.Line:
                qp.setPen(QPen(shape_color, shape[5], Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))
                qp.drawLine(shape[0], shape[1])

            elif shape_type == ShapeMode.FreeForm:
                qp.setPen(QPen(shape_color, shape[5], Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))
                for free_form_point in range(len(shape[4]) - 1):
                    qp.drawLine(shape[4][free_form_point], shape[4][free_form_point + 1])

            elif shape_type == ShapeMode.Semicircle:
                angle = self.calculateAngle(shape[0], shape[1])
                self.drawSemicircle(qp, shape[0], shape[1], shape_color, shape[5], shape[6], angle)

            qp.setPen(QPen(getShapeColor(), getPenWidth(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))


    def redrawBorder(self, qp, inner_coords):
        pen = QPen(Qt.GlobalColor.black, 3)
        qp.setPen(pen)
        brush = QBrush(Qt.GlobalColor.lightGray)
        qp.setBrush(brush)

        width, height = self.width(), self.height()
        margin = 0

        # Coordinates of the corners of the outer square
        x1, y1 = margin, margin
        x2, y2 = width - margin, height - margin

        corner1Points = [QPoint(0,0), QPoint((x1 + x2) // 2,0), QPoint(0, (y1 + y2) // 2)]
        corner1 = QPolygon(corner1Points)

        corner2Points = [QPoint(width,0), QPoint((x1 + x2) // 2,0), QPoint(width, (y1 + y2) // 2)]
        corner2 = QPolygon(corner2Points)

        corner3Points = [QPoint(0,height), QPoint((x1 + x2) // 2,height), QPoint(0, (y1 + y2) // 2)]
        corner3 = QPolygon(corner3Points)

        corner4Points = [QPoint(width,height), QPoint((x1 + x2) // 2,height), QPoint(width, (y1 + y2) // 2)]
        corner4 = QPolygon(corner4Points)

        # Drawing outer square
        qp.drawPolygon (corner1)
        qp.drawPolygon (corner2)
        qp.drawPolygon (corner3)
        qp.drawPolygon (corner4)

        # Draw the edges of the inner rotated square
        for i in range(len(inner_coords)):
            qp.drawLine(int(inner_coords[i][0]), int(inner_coords[i][1]),
                             int(inner_coords[(i+1) % len(inner_coords)][0]),
                             int(inner_coords[(i+1) % len(inner_coords)][1]))

        if getCurrentPatternType() == PatternType.Symmetric or getCurrentPatternType() == PatternType.Asymmetric:
            # draw a dashed line in the middle of the canvas
            pen.setStyle(Qt.PenStyle.DashLine)
            qp.setPen(pen)
            qp.drawLine(int(inner_coords[0][0]), int(y1), int(inner_coords[0][0]), int(y2))
            setSymmetryLine([int(inner_coords[0][0]), int(y1), int(y2)])

        brush = QBrush(getShapeColor())
        qp.setBrush(brush)
        pen = QPen(getShapeColor(), getPenWidth())
        qp.setPen(pen)


    def drawCheckerboard(self, qp, inner_coords):
        pen = QPen(Qt.GlobalColor.black, 3)
        qp.setPen(pen)
        brush = QBrush(Qt.GlobalColor.lightGray)
        qp.setBrush(brush)

        num_classic_lines = getNumClassicLines()
        # draw 3 dashed lines going from lower left to upper right and upper left to lower right
        pen.setStyle(Qt.PenStyle.DashLine)
        qp.setPen(pen)
        distance = calculate_distance(inner_coords[0], inner_coords[2])
        offset = distance / (num_classic_lines + 1) / 2
        line_distance = distance / 2
        # Draw 3 parallel dashed lines going from bottom left to top right
        classic_cuts = []
        current_index = 1
        for i in range(1, num_classic_lines + 1):
            if current_index not in getClassicIndicesLineDeleteList():
                # Calculate start and end points for each line
                start_x_bottom = inner_coords[3][0] + (i * offset)
                start_y_bottom = inner_coords[3][1] + (i * offset)

                end_x_bottom = start_x_bottom + line_distance
                end_y_bottom = start_y_bottom - line_distance

                # Draw the dashed line
                qp.drawLine(int(start_x_bottom), int(start_y_bottom), int(end_x_bottom), int(end_y_bottom))
                classic_cuts.append(([start_x_bottom, start_y_bottom, end_x_bottom, end_y_bottom], current_index))

            current_index += 1

            if current_index not in getClassicIndicesLineDeleteList():
                start_x_top = inner_coords[3][0] + (i * offset)
                start_y_top = inner_coords[3][1] - (i * offset)

                end_x_top = start_x_top + line_distance
                end_y_top = start_y_top + line_distance

                # Draw the dashed line
                qp.drawLine(int(start_x_top), int(start_y_top), int(end_x_top), int(end_y_top))
                classic_cuts.append(([start_x_top, start_y_top, end_x_top, end_y_top], current_index))

            current_index += 1

        self.classic_cuts = classic_cuts

        # THIS DYNAMICALLY DRAWS THE CHECKERBOARD PATTERN, BUT DOES NOT WORK WITH DELETING LINES YET

        # # Create a dynamic checkerboard pattern over the inner diamond
        # if inner_coords and len(inner_coords) == 4:
        #     # Determine the grid size from the number of dashed lines
        #     num_dashed_lines = len(self.classic_cuts) // 2  # Each line has two entries in classic_cuts
        #     grid_size = num_dashed_lines + 1

        #     # Draw a rotated checkerboard that matches the diamond orientation
        #     qp.setBrush(QBrush(getBackgroundColor()))

        #     # Calculate all grid points for an NxN grid within the diamond
        #     grid_points = []
        #     for i in range(4):  # For each corner
        #         start = inner_coords[i]
        #         end = inner_coords[(i+1) % 4]  # Next corner

        #         # Create grid_size+1 points along each edge (including corners)
        #         for j in range(grid_size + 1):
        #             t = j / grid_size  # Parameter from 0 to 1
        #             x = start[0] + t * (end[0] - start[0])
        #             y = start[1] + t * (end[1] - start[1])
        #             grid_points.append((i, j, (x, y)))  # Store edge index, position, and coordinates

        #     # Create the internal grid points using bilinear interpolation
        #     for row in range(grid_size + 1):
        #         for col in range(grid_size + 1):
        #             # Calculate the four corners of this cell
        #             top_left = (inner_coords[0][0] + col/grid_size * (inner_coords[1][0] - inner_coords[0][0]) +
        #                         row/grid_size * (inner_coords[3][0] - inner_coords[0][0]),
        #                         inner_coords[0][1] + col/grid_size * (inner_coords[1][1] - inner_coords[0][1]) +
        #                         row/grid_size * (inner_coords[3][1] - inner_coords[0][1]))

        #             top_right = (inner_coords[0][0] + (col+1)/grid_size * (inner_coords[1][0] - inner_coords[0][0]) +
        #                             row/grid_size * (inner_coords[3][0] - inner_coords[0][0]),
        #                             inner_coords[0][1] + (col+1)/grid_size * (inner_coords[1][1] - inner_coords[0][1]) +
        #                             row/grid_size * (inner_coords[3][1] - inner_coords[0][1]))

        #             bottom_left = (inner_coords[0][0] + col/grid_size * (inner_coords[1][0] - inner_coords[0][0]) +
        #                             (row+1)/grid_size * (inner_coords[3][0] - inner_coords[0][0]),
        #                             inner_coords[0][1] + col/grid_size * (inner_coords[1][1] - inner_coords[0][1]) +
        #                             (row+1)/grid_size * (inner_coords[3][1] - inner_coords[0][1]))

        #             bottom_right = (inner_coords[0][0] + (col+1)/grid_size * (inner_coords[1][0] - inner_coords[0][0]) +
        #                             (row+1)/grid_size * (inner_coords[3][0] - inner_coords[0][0]),
        #                             inner_coords[0][1] + (col+1)/grid_size * (inner_coords[1][1] - inner_coords[0][1]) +
        #                             (row+1)/grid_size * (inner_coords[3][1] - inner_coords[0][1]))

        #             # Only fill cells where (row + col) is even (checkerboard pattern)
        #             if (row + col) % 2 == 0:
        #                 polygon = QPolygonF([
        #                     QPointF(top_left[0], top_left[1]),
        #                     QPointF(top_right[0], top_right[1]),
        #                     QPointF(bottom_right[0], bottom_right[1]),
        #                     QPointF(bottom_left[0], bottom_left[1])
        #                 ])
        #                 qp.drawPolygon(polygon)
        #                 self.cells.append([polygon, getBackgroundColor()])


        #             else:
        #                 qp.setBrush(QBrush(getShapeColor()))
        #                 qp.drawPolygon(polygon)
        #                 self.cells.append([polygon, getShapeColor()])

        brush = QBrush(getShapeColor())
        qp.setBrush(brush)
        pen = QPen(getShapeColor(), getPenWidth())
        qp.setPen(pen)




    def get_drawing_image(self):
        image = QImage(self.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)  # Fill with transparent color

        painter = QPainter(image)
        self.render(painter)  # Render the current drawing to the image
        return image


    def drawSquare(self, qp, start, end, color, pen_width, filled):
        self.penAndBrushSetup(qp, color, pen_width, filled)

        rect = QRectF(QPointF(start), QPointF(end))

        path = QPainterPath()
        path.addRect(rect)  # Define rectangle with the given points

        qp.drawRect(QRect(start, end))

        qp.setPen(QPen(getShapeColor(), getPenWidth(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))


    def drawCircle(self, qp, start, end, color, pen_width, filled):
        self.penAndBrushSetup(qp, color, pen_width, filled)

        # Calculate the radius using the manhattan length between start and end.
        radius = int((start - end).manhattanLength() / 2)

        # Define a bounding rectangle for the circle centered at 'start'
        rect = QRectF(start.x() - radius, start.y() - radius, 2 * radius, 2 * radius)

        path = QPainterPath()
        path.addEllipse(rect)

        qp.drawPath(path)
        qp.setPen(QPen(getShapeColor(), getPenWidth(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))


    def drawHeart(self, qp, start, end, color, pen_width, filled):
        self.penAndBrushSetup(qp, color, pen_width, filled)

        # Calculate width and height
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())
        x_offset = start.x()
        y_offset = start.y()

        # Create the heart shape using QPainterPath
        drawpath = QPainterPath()
        drawpath.moveTo(x_offset + width / 2, y_offset + height / 4)
        drawpath.cubicTo(x_offset + width * 0.75, y_offset - height / 4, x_offset + width * 1.5, y_offset + height / 2, x_offset + width / 2, y_offset + height)
        drawpath.cubicTo(x_offset - width * 0.5, y_offset + height / 2, x_offset + width * 0.25, y_offset - height / 4, x_offset + width / 2, y_offset + height / 4)

        qp.drawPath(drawpath)

        qp.setPen(QPen(getShapeColor(), getPenWidth(), Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))


    def drawSemicircle(self, qp, start, end, color, pen_width, filled, rotation_angle=0):
        self.penAndBrushSetup(qp, color, pen_width, filled)

        dx, dy, distance = self.calculateSemiDistance(start, end)
        size = distance

        # Calculate center of semicircle
        center_x = (start.x() + end.x()) / 2
        center_y = (start.y() + end.y()) / 2

        # Create a square bounding box for the semicircle, centered at the midpoint
        rect_x = int(center_x - size / 2)
        rect_y = int(center_y - size / 2)
        size_int = int(size)
        rect = QRect(rect_x, rect_y, size_int, size_int)

        # Set the rotation as the start angle for drawPie
        # Note: drawPie expects angle in 1/16th of degrees, and 0° is to the right, counter-clockwise positive
        start_angle = int((-180 + rotation_angle) * -16)  # Rotates the direction the semicircle "opens"
        span_angle = int(180 * -16)  # Draw counterclockwise to create a filled top semicircle

        qp.drawPie(rect, start_angle, span_angle)


    def calculateSemiDistance(self, start, end):
        dx = end.x() - start.x()
        dy = end.y() -  start.y()
        distance = math.hypot(dx, dy)
        return dx, dy, distance


    def calculateAngle(self, start, end):
        dx, dy, distance = self.calculateSemiDistance(start, end)

        angle = math.degrees(math.atan2(dy, dx)) % 360
        return angle


    def penAndBrushSetup(self, qp, color, pen_width, filled):
        if filled:
            qp.setBrush(color)
        else:
            qp.setBrush(Qt.BrushStyle.NoBrush)

        qp.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))


    # Checks if the point is within a certain threshold of the line
    def lineContainsPoint(self, point, begin, end, threshold=4.0):
        area = abs((end.x() - begin.x()) * (begin.y() - point.y()) - (begin.x() - point.x()) * (end.y() - begin.y()))
        line_length = math.sqrt((end.x() - begin.x())**2 + (end.y() - begin.y())**2)

        if line_length == 0:
            return False

        distance = area / line_length

        within_bounds = (min(begin.x(), end.x()) <= point.x() <= max(begin.x(), end.x())) and \
                        (min(begin.y(), end.y()) <= point.y() <= max(begin.y(), end.y()))

        return distance <= threshold and within_bounds


    def heartContainsPoint(self, point, start, end):
        width = abs(end.x() - start.x())
        height = abs(end.y() - start.y())
        x_offset, y_offset = start.x() + width // 2, start.y() + height // 2

        # Scale factor
        scale_x = width / 32
        scale_y = height / 32

        # Check if the point is inside the heart shape using parametric equations
        t = 0
        while t <= 2 * math.pi:
            x = int(16 * math.sin(t) ** 3 * scale_x) + x_offset
            y = int(- (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) * scale_y) + y_offset

            # Check if the point is close to any part of the heart curve
            if abs(point.x() - x) < 5 and abs(point.y() - y) < 5:
                return True  # Point is inside the heart

            t += 0.1

        return False  # Point is outside the heart


    def mousePressEvent(self, event):
        if self.drawing_mode:
            self.begin = event.pos()
            self.end = event.pos()

            if getShapeMode() == ShapeMode.FreeForm:
                self.free_form_points = [event.position().toPoint()]
            self.update()


    def mouseMoveEvent(self, event):
        if self.drawing_mode:
            self.end = event.pos()

            if getShapeMode() == ShapeMode.FreeForm:
                self.free_form_points.append(event.position().toPoint())
            self.update()


    def mouseReleaseEvent(self, event):
        if self.drawing_mode:
            shape_color = getShapeColor()
            if getShapeMode() == ShapeMode.FreeForm:
                self.shapes.append([self.begin, self.end, getShapeMode(), getShapeColor(), list(self.free_form_points), getPenWidth(), False])

            else:
                if getCurrentPatternType() == PatternType.Classic:
                    shape_color = self.flipSquareColor(shape_color)
                    self.begin, self.end = snapShapeToClassicCuts(self.classic_cuts, getShapeMode(), self.begin, self.end, self.width(), self.height())

                if getShapeMode() == ShapeMode.Line:
                    self.shapes.append([self.begin, self.end, getShapeMode(), shape_color, [], getPenWidth(), False])

                elif getShapeMode() == ShapeMode.Semicircle:
                    self.shapes.append([self.begin, self.end, getShapeMode(), shape_color, [], 2, getFilled()])

                else:
                    self.shapes.append([self.begin, self.end, getShapeMode(), shape_color, [], 1, getFilled()])

            if getCurrentPatternType() == PatternType.Symmetric:
                self.shapes.append([QPoint(self.width() - self.begin.x(), self.begin.y()), QPoint(self.width() - self.end.x(), self.end.y()), getShapeMode(), getShapeColor(), [], 1, getFilled()])

            self.begin = event.pos()
            self.end = event.pos()
            self.update()
            self.main_window.update_backside_image()


    def set_drawing_mode(self, enabled):
        self.drawing_mode = enabled
        self.update()



# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    cursor_button = None
    free_form_button = None
    eraser_button = None
    line_button = None
    square_button = None
    circle_button = None
    heart_button = None
    semicircle_button = None


    def __init__(self):
        super().__init__()
        self.shape_attributes = []

        # Save the color buttons in a dictionary
        self.foreground_buttons = {}
        self.background_buttons = {}

        self.setWindowTitle("Weave Toolkit")
        self.setStyleSheet("background-color: white;")

        # Create the central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create the Menu toolbar
        menu_toolbar = self.createMenuToolbar()
        main_layout.addWidget(menu_toolbar)

        # Create the Shapes toolbar
        shapes_toolbar = self.createShapesToolbar()
        main_layout.addWidget(shapes_toolbar)

        # Create the Colors toolbar
        colors_toolbar = self.createColorsToolbar()
        main_layout.addWidget(colors_toolbar)

        # Create the Drawing space (where users draw)
        self.drawing_layout = QHBoxLayout()
        self.drawing_widget_layout = QVBoxLayout()
        self.drawing_backside_layout = QVBoxLayout()

        self.drawing_layout.addLayout(self.drawing_widget_layout)
        self.drawing_layout.addLayout(self.drawing_backside_layout)

        self.drawing_label = QLabel("Front Side:")
        self.drawing_label.setStyleSheet("color: black;")
        self.backside_label = QLabel("Back Side (not modifiable), Pattern Type is set to: ")
        self.backside_label.setStyleSheet("color: black;")

        self.drawing_widget_layout.addWidget(self.drawing_label)
        self.drawing_backside_layout.addWidget(self.backside_label)

        self.drawing_widget = DrawingWidget(self)
        self.drawing_backside = QLabel(self)

        # (drawing_widget background color controlled in the DrawingWidget class inside paintEvent)
        # (drawing_backside background color copied from drawing_widget)

        self.drawing_widget_layout.addWidget(self.drawing_widget)
        self.drawing_backside_layout.addWidget(self.drawing_backside)

        # Create the container widget and set the background color
        self.drawing_container = QWidget()
        self.drawing_container.setStyleSheet("background-color: lightgrey;")  # Set light grey background
        self.drawing_container.setLayout(self.drawing_layout)

        self.drawing_widget.installEventFilter(self)

        self.scene = QGraphicsScene()

        # Create a stacked widget to switch between views
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.addWidget(self.drawing_container)
        main_layout.addWidget(self.stacked_widget)

        self.drawing_widget.setFixedSize(500, 500)
        self.drawing_backside.setFixedSize(500, 500)
        self.setFixedSize(QSize(1200, 700))
        # self.setWindowState(Qt.WindowState.WindowMaximized)

        self.update_backside_image()


    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.MouseButtonRelease and source == self.drawing_widget:
            self.update_backside_image()
        return super().eventFilter(source, event)


    def update_backside_image(self):
        self.backside_label.setText(f"Back Side (not modifiable) and the Pattern Type is: {getCurrentPatternType()}")
        drawing_image = self.drawing_widget.get_drawing_image()
        mirrored_image = drawing_image.mirrored(True, False)  # Mirror horizontally
        pixmap = QPixmap.fromImage(mirrored_image)
        self.drawing_backside.setPixmap(pixmap)


    def createMenuToolbar(self):
        menu_toolbar = QToolBar("Menu toolbar")

        file_button = QPushButton("File", self)
        file_button.setStyleSheet("background-color: lightgray; color: black;")
        file_button_menu = self.createFileDropdownMenu()
        file_button.setMenu(file_button_menu)
        menu_toolbar.addWidget(file_button)

        view_button = QPushButton("View", self)
        view_button.setStyleSheet("background-color: lightgray; color: black;")
        view_button_menu = self.createViewDropdownMenu()
        view_button.setMenu(view_button_menu)
        menu_toolbar.addWidget(view_button)

        update_button_svg = QPushButton("Update SVG", self)
        update_button_svg.setStyleSheet("background-color: lightgray; color: black;")
        update_button_svg.clicked.connect(lambda: self.save_as_svg(getUserOutputSVGFileName() + ".svg", self.drawing_widget.size()))
        update_button_svg.clicked.connect(lambda: self.updateDisplaySvg())
        menu_toolbar.addWidget(update_button_svg)

        edit_button = QPushButton("Edit", self)
        edit_button.setStyleSheet("background-color: lightgray; color: black;")
        edit_button.clicked.connect(lambda: self.editDisplay())
        menu_toolbar.addWidget(edit_button)

        weaving_pattern_button = QPushButton("Weaving Pattern", self)
        weaving_pattern_button.setStyleSheet("background-color: lightgray; color: black;")
        weaving_pattern_button_menu = self.createWeavingPatternDropdownMenu()
        weaving_pattern_button.setMenu(weaving_pattern_button_menu)
        menu_toolbar.addWidget(weaving_pattern_button)

        side_types_button = QPushButton("Side Type", self)
        side_types_button.setStyleSheet("background-color: lightgray; color: black;")
        side_types_button_menu = self.createSidesDropdownMenu()
        side_types_button.setMenu(side_types_button_menu)
        menu_toolbar.addWidget(side_types_button)

        return menu_toolbar


    def createFileDropdownMenu(self):
        file_menu = QMenu("File", self)
        file_menu.setStyleSheet("""
        QMenu::item {
            color: black;
            background: transparent;
        }
        QMenu::item:selected {
            background-color: #D3D3D3;  /* Lighter gray */
            color: black;
        }
        """)

        # Create actions
        action_new = QAction("New", self)
        action_new.triggered.connect(lambda: self.clear_canvas())
        action_open = QAction("Open", self)
        action_open.triggered.connect(lambda: self.open_png())
        action_save = QAction("Save", self)
        action_save.triggered.connect(lambda: self.save_canvas_as_png())
        action_save_svg = QAction("Export SVG", self)
        action_save_svg.triggered.connect(lambda: self.exportSVG())
        action_guide_export = QAction("Export Guide", self)
        action_guide_export.triggered.connect(lambda: self.exportGuide())
        action_undo = QAction("Undo (ctrl + z)", self)
        action_undo.triggered.connect(self.undo_last_shape)
        action_undo.setShortcut("Ctrl+Z")

        # Add actions to the menu
        file_menu.addAction(action_new)
        file_menu.addAction(action_open)
        file_menu.addAction(action_save)
        file_menu.addAction(action_save_svg)
        file_menu.addAction(action_guide_export)
        file_menu.addAction(action_undo)

        return file_menu


    def createViewDropdownMenu(self):
        view_menu = QMenu("View", self)
        view_menu.setStyleSheet("color: black;")
        action_zoom = QAction("Zoom", self)
        action_fullscreen = QAction("Fullscreen", self)
        action_gridlines = QAction("Toggle Gridlines", self)
        action_show_backside = QAction("Show/Hide Backside", self)
        action_print_size = QAction("Change Print Size", self)
        view_menu.addAction(action_zoom)
        view_menu.addAction(action_fullscreen)
        view_menu.addAction(action_gridlines)
        view_menu.addAction(action_show_backside)
        view_menu.addAction(action_print_size)

        return view_menu


    def clear_canvas(self):
            self.drawing_widget.shapes = []  # Clear all shapes
            self.drawing_widget.free_form_points = []  # Clear all free form points
            self.drawing_widget.begin = QPoint()  # Reset begin point
            self.drawing_widget.end = QPoint()  # Reset end point
            setClassicIndicesLineDeleteList([])  # Clear classic indices line delete list
            self.drawing_widget.update()  # Trigger repaint of the drawing widget
            self.update_backside_image()  # Update the backside image
            if getCurrentPatternType() == PatternType.Classic:

                self.setPatternType(PatternType.Classic)  # Reset pattern type to Classic
            
            else:
                self.setPatternType(PatternType.Simple)  # Reset pattern type to default

    def setPatternType(self, pattern):
        setCurrentPatternType(pattern)

        self.classic_lines_container.setVisible(pattern == PatternType.Classic)
        if pattern == PatternType.Classic:
            self.classic_lines_container.adjustSize()
            shapes_toolbar = self.findChild(QToolBar, "Shapes toolbar")
            if shapes_toolbar:
                shapes_toolbar.adjustSize()

        self.update()
        self.update_backside_image()

        QApplication.processEvents()


    def setSideType(self, side):
        setCurrentSideType(side)
        self.update()
        self.update_backside_image()


    def createWeavingPatternDropdownMenu(self):
        weaving_pattern_menu = QMenu("Weaving Pattern", self)
        weaving_pattern_menu.setToolTipsVisible(True)  # Enable tooltips in the menu
        weaving_pattern_menu.setStyleSheet("""
        QMenu::item {
            color: black;
            background: transparent;
        }
        QMenu::item:selected {
            background-color: #D3D3D3;  /* Lighter gray */
            color: black;
        }
        """)
        # Create actions
        action_simple = QAction("Simple", self)
        action_simple.setToolTip("Simple pattern: Cuts out the pattern making minimal weaving necessary")
        action_symmetrical = QAction("Symmetrical", self)
        action_symmetrical.setToolTip("Symmetrical pattern: Mirrors whatever is drawn on one half to the other, creating a perfectly symmetrical pattern")
        action_asymmetrical = QAction("Asymmetrical", self)
        action_asymmetrical.setToolTip("Asymmetrical pattern: Creates an asymmetric pattern")
        action_classic = QAction("Classic", self)
        action_classic.setToolTip("Classic pattern: Creates a traditional weaving pattern, with the possibility of adding patterns")

        action_simple.triggered.connect(lambda: (self.setPatternType(PatternType.Simple)))
        action_symmetrical.triggered.connect(lambda: (self.setPatternType(PatternType.Symmetric)))
        action_asymmetrical.triggered.connect(lambda: (self.setPatternType(PatternType.Asymmetric)))
        action_classic.triggered.connect(lambda: (self.setPatternType(PatternType.Classic), self.clear_canvas()))

        # Add actions to the menu
        weaving_pattern_menu.addAction(action_simple)
        weaving_pattern_menu.addAction(action_symmetrical)
        weaving_pattern_menu.addAction(action_asymmetrical)
        weaving_pattern_menu.addAction(action_classic)

        return weaving_pattern_menu


    def createSidesDropdownMenu(self):
        sides_menu = QMenu("Sides", self)
        sides_menu.setStyleSheet("""
        QMenu::item {
            color: black;
            background: transparent;
        }
        QMenu::item:selected {
            background-color: #D3D3D3;  /* Lighter gray */
            color: black;
        }
        """)

        # Create actions
        action_one_sided = QAction("One-sided", self)
        action_two_sided = QAction("Two-sided", self)
        action_one_sided.triggered.connect(lambda: (self.setSideType(SideType.OneSided)))
        action_two_sided.triggered.connect(lambda: (self.setSideType(SideType.TwoSided)))

        # Add actions to the menu
        sides_menu.addAction(action_one_sided)
        sides_menu.addAction(action_two_sided)

        return sides_menu


    def updateDisplaySvg(self):
        self.backside_label.setText("Front Side final product:")
        heart = self.cvImageToPixmap(mainAlgorithmSvg(getUserPreprocessedPattern(), getCurrentSideType(), getCurrentPatternType(), "show"))

        # Shows the design created by the users on the heart
        pixmap = QPixmap(heart)

        scaled_pixmap = pixmap.scaled(
            self.drawing_backside.width() * 2,
            self.drawing_backside.height() * 2,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
            )
        self.drawing_backside.setPixmap(scaled_pixmap)
        self.drawing_backside.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def editDisplay(self):
        self.backside_label.setText(f"Back Side (not modifiable) - {getCurrentPatternType()}:")
        self.stacked_widget.setCurrentWidget(self.drawing_widget)
        self.update_backside_image()


    def createShapesToolbar(self):
        shapes_toolbar = QToolBar("Shapes toolbar")

        # Cursor Button
        MainWindow.cursor_button = self.createShapeButton("icons/cursor.png", "Cursor", ShapeMode.Cursor)
        shapes_toolbar.addAction(MainWindow.cursor_button)

        shapes_toolbar.addSeparator()

        # Free Form Button
        MainWindow.free_form_button = self.createShapeButton("icons/free_form.png", "Free Form", ShapeMode.FreeForm)
        shapes_toolbar.addAction(MainWindow.free_form_button)

        shapes_toolbar.addSeparator()

        # Eraser Button
        MainWindow.eraser_button = self.createShapeButton("icons/eraser.png", "Eraser", ShapeMode.Eraser)
        shapes_toolbar.addAction(MainWindow.eraser_button)

        shapes_toolbar.addSeparator()

        # Line Button
        MainWindow.line_button = self.createShapeButton("icons/line.png", "Line", ShapeMode.Line)
        shapes_toolbar.addAction(MainWindow.line_button)

        shapes_toolbar.addSeparator()

        # Square Button
        MainWindow.square_button = self.createShapeButton("icons/square.png", "Square", ShapeMode.Square)
        shapes_toolbar.addAction(MainWindow.square_button)

        shapes_toolbar.addSeparator()

        # Circle Button
        MainWindow.circle_button = self.createShapeButton("icons/circle.png", "Circle", ShapeMode.Circle)
        shapes_toolbar.addAction(MainWindow.circle_button)

        # Heart Button
        MainWindow.heart_button = self.createShapeButton("icons/heart.png", "Heart", ShapeMode.Heart)
        shapes_toolbar.addAction(MainWindow.heart_button)

        # Semicircle Button
        MainWindow.semicircle_button = self.createShapeButton("icons/semicircle.png", "Semicircle", ShapeMode.Semicircle)
        shapes_toolbar.addAction(MainWindow.semicircle_button)

        classic_container_widget = QWidget()
        container_layout = QVBoxLayout(classic_container_widget)

        self.classic_lines_container = self.createClassicLinesWidget()
        container_layout.addWidget(self.classic_lines_container)

        shapes_toolbar.addWidget(classic_container_widget)

        return shapes_toolbar


    def createShapeButton(self, icon_path, button_text, shape_mode):
        shape_button = QAction(QIcon(icon_path), button_text, self)
        shape_button.triggered.connect(lambda: self.setMode(shape_mode))
        return shape_button


    # When a shape button is clicked, it is then set to that drawing mode
    def setMode(self, shape_mode):
        if shape_mode == ShapeMode.Cursor:
            self.drawing_widget.set_drawing_mode(False)

        elif shape_mode == ShapeMode.Eraser:
            self.drawing_widget.begin = QPoint(-999, -999) # Reset the begin and end points so the most recent shape isn't erased
            self.drawing_widget.end = QPoint(-1, -1)
            self.drawing_widget.set_drawing_mode(True)

        else:
            self.drawing_widget.set_drawing_mode(True)

        setShapeMode(shape_mode)


    def createColorsToolbar(self):
        colors_toolbar = QToolBar("Colors toolbar")

        foreground_label = QLabel("Foreground Colors: ")
        foreground_label.setStyleSheet("color: black;")
        colors_toolbar.addWidget(foreground_label)

        foreground_colors = [("Red", "red"), ("Green", "green"), ("Orange", "orange"), ("Blue", "blue")]

        for color_name, color_value in foreground_colors:
            button = QPushButton(color_name, self, styleSheet=f"background-color: {color_value}; color: black;")
            self.foreground_buttons[color_name] = button
            button.clicked.connect(partial(self.change_foreground_color, color_value))
            colors_toolbar.addWidget(button)

        rainbow_button = QPushButton("Rainbow Button", self)
        rainbow_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #FF0000,
                    stop: 0.16 #FF7F00,
                    stop: 0.33 #FFFF00,
                    stop: 0.5 #00FF00,
                    stop: 0.66 #0000FF,
                    stop: 0.83 #4B0082,
                    stop: 1 #8B00FF
                );
            }
        """)

        colors_toolbar.addWidget(rainbow_button)

        background_label = QLabel("Background Colors: ")
        background_label.setStyleSheet("color: black;")
        colors_toolbar.addWidget(background_label)

        background_colors = [("Red", "red"), ("Green", "green"), ("Orange", "orange"), ("Blue", "blue"), ("White", "white")]

        for color_name, color_value in background_colors:
            button = QPushButton(color_name, self, styleSheet=f"background-color: {color_value}; color: black;")
            self.background_buttons[color_name] = button
            button.clicked.connect(partial(self.change_background_color, color_value))
            colors_toolbar.addWidget(button)

        colors_toolbar.addWidget(self.createStrokeWidthWidget())

        return colors_toolbar


    def createStrokeWidthWidget(self):
        self.stroke_width_layout = QVBoxLayout()

        initial_stroke_width = 1
        self.stroke_width_label = QLabel(f'Stroke Width: {initial_stroke_width}', self)
        self.stroke_width_label.setStyleSheet("color: black;")
        self.stroke_width_layout.addWidget(self.stroke_width_label)

        self.stroke_width_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.stroke_width_slider.setMinimum(1)
        self.stroke_width_slider.setMaximum(20)
        self.stroke_width_slider.setValue(initial_stroke_width)
        self.stroke_width_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.stroke_width_slider.setTickInterval(1)
        self.stroke_width_slider.valueChanged.connect(self.updateStrokeWidth)
        self.stroke_width_layout.addWidget(self.stroke_width_slider)

        stroke_width_container = QWidget()
        stroke_width_container.setLayout(self.stroke_width_layout)
        return stroke_width_container


    def updateStrokeWidth(self, value):
        self.stroke_width_label.setText(f'Stroke Width: {value}')
        setPenWidth(value)


    def createClassicLinesWidget(self):
        classic_lines_container = QWidget()
        classic_lines_container.setObjectName("ClassicLinesContainer")
        classic_lines_container.setFixedHeight(28)

        self.classic_lines_layout = QHBoxLayout(classic_lines_container)

        initial_num_lines = getNumClassicLines()
        self.classic_lines_label = QLabel(f'Classic Lines: {initial_num_lines}', self)
        self.classic_lines_label.setStyleSheet("color: black;")
        self.classic_lines_layout.addWidget(self.classic_lines_label)

        self.num_classic_lines_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.num_classic_lines_slider.setMinimum(1)
        self.num_classic_lines_slider.setMaximum(9)
        self.num_classic_lines_slider.setValue(initial_num_lines)
        self.num_classic_lines_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.num_classic_lines_slider.setTickInterval(1)
        self.num_classic_lines_slider.valueChanged.connect(self.updateClassicLines)
        self.classic_lines_layout.addWidget(self.num_classic_lines_slider)

        classic_lines_container.setVisible(getCurrentPatternType() == PatternType.Classic)

        return classic_lines_container

    def updateClassicLines(self, value):
        """Update the classic lines count when slider changes"""
        self.classic_lines_label.setText(f'Classic Lines: {value}')
        setNumClassicLines(value)
        self.drawing_widget.update()  # Refresh the drawing

    def get_color_name(self, qcolor):
        color_map = {
            QColor("red").name(): "Red",
            QColor("green").name(): "Green",
            QColor("blue").name(): "Blue",
             QColor("orange").name(): "Orange",
             QColor("white").name(): "White"
        }
        return color_map.get(qcolor.name(), "Unkown")

    def change_active_color(self, old_color, foreground_bool):
        if getCurrentPatternType() == PatternType.Classic:
            for shape in self.drawing_widget.shapes:

                if shape[3] == old_color:
                    if foreground_bool:

                        shape[3] = getShapeColor()
                    else:
                        shape[3] = getBackgroundColor()

    def disable_enable_color_button(self, foreground_bool):
        if foreground_bool:
            color = self.get_color_name(getShapeColor())
            for color_name, button in self.background_buttons.items():
                button.setEnabled(color_name != color)

        else:
            color = self.get_color_name(getBackgroundColor())
            for color_name, button in self.foreground_buttons.items():
                button.setEnabled(color_name != color)


    def change_foreground_color(self, color):
        old_color = getShapeColor()
        setShapeColor(QColor(color))
        self.disable_enable_color_button(True)
        self.change_active_color(old_color, True)
        self.update()
        self.update_backside_image()


    def change_background_color(self, color):
        old_color = getBackgroundColor()
        setBackgroundColor(QColor(color))
        self.disable_enable_color_button(False)
        self.change_active_color(old_color, False)
        self.update()
        self.update_backside_image()


    def keyPressEvent(self, event):
        """Handle key press events for the main window."""
        # Check for Ctrl+Z (undo)
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo_last_shape()
        else:
            super().keyPressEvent(event)


    def undo_last_shape(self):
        """Remove the most recently added shape."""
        if self.drawing_widget.shapes:
            self.drawing_widget.shapes.pop()  # Remove the last shape

            # If using symmetrical pattern, also remove the mirrored shape
            if getCurrentPatternType() == PatternType.Symmetric and self.drawing_widget.shapes:
                self.drawing_widget.shapes.pop()

            self.drawing_widget.update()  # Redraw the canvas
            self.update_backside_image()  # Update the mirrored image


    def save_canvas_as_png(self, filename="canvas_output.png"):
        pixmap = QPixmap(self.drawing_widget.size())  # Create pixmap of the same size
        self.drawing_widget.render(pixmap)  # Render the widget onto the pixmap
        pixmap.save(filename, "PNG")  # Save as PNG


    def save_as_svg(self, file_name, canvas_size):
        # calculate the min/max x/y of the inner square
        width = canvas_size.width()
        height = canvas_size.height()
        x1, y1 = 0, 0
        x2, y2 = width, height
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Coordinates of the corners of the inner rotated square
        inner_coords = [
            (center_x, y1),
            (x2, center_y)
        ]

        square_size = calculate_distance(inner_coords[0], inner_coords[1])

        # save the drawing canvas as an svg
        svg_generator = QSvgGenerator()
        svg_generator.setFileName(file_name)  # Path to save the SVG file
        svg_generator.setSize(canvas_size)      # Set the size of the SVG to match the widget size
        svg_view_box = QRect(0, 0, width + 40, height + 40)
        svg_generator.setViewBox(svg_view_box)
        painter = QPainter(svg_generator)

        # when saving the svg, only the shapes (and not the drawing border) are saved
        self.drawing_widget.redrawAllShapes(painter)
        painter.end()

        paths, attributes = svg2paths(file_name)

        # Copy shapes and attributes
        shapes_copy = copy.deepcopy(self.drawing_widget.shapes)
        attributes_copy = copy.deepcopy(attributes)

        shape_attr_list = []
        shape_types = [shape[2] for shape in shapes_copy]

        print("attributes: ", attributes_copy)
        print("number of shapes: ", len(shapes_copy))
        print("number of attributes: ", len(attributes_copy))
        print("number of paths: ", len(paths))

        for attr, shape in zip(attributes_copy, shapes_copy):
            if getCurrentPatternType() == PatternType.Classic:
                shape_color = shape[3]
            else:
                shape_color = getShapeColor()
            pen_width = shape[5]
            filled = shape[6]

            updated_attr = attr.copy()

            updated_attr['stroke'] = shape_color.name()
            updated_attr['stroke-width'] = pen_width
            updated_attr['fill'] = shape_color.name()

            shape_attr_list.append(updated_attr)

        print("shape_attr_list: ", shape_attr_list)

        file_with_attributes = "svg_file_2.svg"

        print("paths: ", paths)
        if paths == []:
            pre_process_user_input(None, None, width, height, square_size)

        else:
            if len(shape_attr_list) < len(paths):
                missing_attrs =[{}] * (len(paths) - len(shape_attr_list)) # use empty attributes as placeholders
                shape_attr_list.extend(missing_attrs)

            wsvg(paths,
                attributes=shape_attr_list,
                filename=file_with_attributes,
                dimensions=(width, height))

            pre_process_user_input(file_with_attributes, shape_types, width, height, square_size)


    def exportGuide(self):
        guide_window = GuideWindow(getCurrentPatternType())
        guide_window.exec()


    def exportSVG(self):
        if getCurrentPatternType() == PatternType.Asymmetric or getCurrentPatternType() == PatternType.Symmetric:
            if shapeNotTouchingSymmetrylineError(self.drawing_widget.shapes): 
                svg_file_path = getUserOutputSVGFileName() + ".svg"
                mainAlgorithmSvg(svg_file_path, getCurrentSideType(), getCurrentPatternType(), function=' ', n_lines=3)
        else:
            svg_file_path = getUserOutputSVGFileName() + ".svg"
            mainAlgorithmSvg(svg_file_path, getCurrentSideType(), getCurrentPatternType(), function=' ', n_lines=3)


    def pixmapToCvImage(self):
        pixmap = QPixmap(self.drawing_widget.size())  # Create pixmap of the same size
        self.drawing_widget.render(pixmap)
        image = pixmap.toImage()
        width, height = image.width(), image.height()

        # Convert QImage to format RGB888 (3 channels)
        image = image.convertToFormat(QImage.Format.Format_RGB888)

        # Get image data as bytes
        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())

        # Convert to NumPy array and reshape (H, W, 3)
        arr = np.array(ptr).reshape((height, width, 3))

        # Convert RGB to BGR for OpenCV
        arr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)

        return arr


    def cvImageToPixmap(self, cv_img):
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width  # RGB format uses 3 bytes per pixel
        cv_img_rgb = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
        q_image = QImage(cv_img_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def open_png(self):
        options = QFileDialog.Option.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open PNG Image",
            "",
            "PNG Files (*.png);;All Files(*))",
            options=options
        )
        if file_name:
            pixmap = QPixmap(file_name)
            if not pixmap.isNull():
                item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(item)
            else:
                print("Failed to load image")

import subprocess

batch_script_path = "clear_svg_files.bat"
bash_script_path = "clear_svg_files.sh"
# Path to the batch script
os_name = sys.platform
if os_name.startswith("win"):
    # Run the batch script
    subprocess.run(batch_script_path, shell=True)
elif os_name.startswith("darwin"):
    # Run the bash script
    subprocess.run(["bash", bash_script_path])

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()