from PyQt6.QtWidgets import (
    QMessageBox,
    QToolTip
)
from PyQt6.QtCore import (
    QPoint,
    QPointF,
    QRectF
)
from PyQt6.QtGui import (
    QCursor,
    QPainterPath
)
from GlobalVariables import (
    getSymmetryLine
)

from ShapeMode import ShapeMode

# Message when user draws out of bounds
def outOfBoundDrawingMessage():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle("Warning")
    msg.setText("This is a warning message.")
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()

def showWarningTooltip(message, position = None):
    if position == None:
        position = QCursor.pos()
    QToolTip.showText(position, message, msecShowTime=3000)

def any_shape_intersects_symmetry_line(shapes, x_symmetry, y1_line, y2_line):
    symmetry_path = symmetryLineToPath(x_symmetry, y1_line, y2_line)
    
    for shape in shapes:
        shape_path = shape_to_path(shape)
        if shape_path.intersects(symmetry_path):
            return True  # At least one shape intersects
    return False


def shapeNotTouchingSymmetrylineError(shapes):
    symmetry_line = getSymmetryLine()
    for shape in shapes:
        if any_shape_intersects_symmetry_line(shapes, symmetry_line[0], symmetry_line[1], symmetry_line[2]):
            return True
    
    showWarningTooltip("At least one shape needs to touch the line of symmetry")
    return False

def shape_to_path(shape):

    if isinstance(shape, list):
        begin, end, shape_type, color, _, pen_width, filled = shape
    
    if shape_type == ShapeMode.Circle:
        return createCirclePath(begin, end)
    
    elif shape_type == ShapeMode.Square:
        return createSquarePath(begin, end)
    
    elif shape_type == ShapeMode.Heart:
        return createHeartPath(begin, end)
    
    elif shape_type == ShapeMode.Semicircle:
        return createSemicirclePath(begin, end)
    

def check_all_shapes_overlap(shape_list):
    if len(shape_list) < 2:
        return False  # Cannot form a group with 1 or 0 shapes

    paths = [shape_to_path(s) for s in shape_list]
    n = len(paths)

    # Build adjacency list based on intersection
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if paths[i].intersects(paths[j]):
                graph[i].add(j)
                graph[j].add(i)

    # Run BFS to check connected component size
    visited = set()
    stack = [0]
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            stack.extend(graph[current] - visited)

    return len(visited) == n


def allShapesOverlapError(shapes):
    if len(shapes) < 2:
        return True
    
    if check_all_shapes_overlap(shapes):
        return True
    
    
    showWarningTooltip("All shapes needs to be combined into one shape")
    return False

# Converts shapes and the symmetry line into QPainterPaths for intersection detection

def symmetryLineToPath(x, y1, y2):
    path = QPainterPath()
    path.moveTo(QPointF(x, y1))
    path.lineTo(QPointF(x, y2))
    return path

def createSquarePath(start, end):
    rect = QRectF(QPointF(start), QPointF(end))
    path = QPainterPath()
    path.addRect(rect)
    return path

def createCirclePath(start, end):
    radius = (start - end).manhattanLength() / 2
    rect = QRectF(start.x() - radius, start.y() - radius, 2 * radius, 2 * radius)
    path = QPainterPath()
    path.addEllipse(rect)
    return path

def createHeartPath(start, end):
    width = abs(end.x() - start.x())
    height = abs(end.y() - start.y())
    x_offset = start.x()
    y_offset = start.y()

    path = QPainterPath()
    path.moveTo(x_offset + width / 2, y_offset + height / 4)
    path.cubicTo(x_offset + width * 0.75, y_offset - height / 4,
                 x_offset + width * 1.5, y_offset + height / 2,
                 x_offset + width / 2, y_offset + height)
    path.cubicTo(x_offset - width * 0.5, y_offset + height / 2,
                 x_offset + width * 0.25, y_offset - height / 4,
                 x_offset + width / 2, y_offset + height / 4)
    return path

def createSemicirclePath(start, end, rotation_angle=0):
    from PyQt6.QtGui import QTransform

    dx = end.x() - start.x()
    dy = end.y() - start.y()
    size = (dx**2 + dy**2) ** 0.5

    center_x = (start.x() + end.x()) / 2
    center_y = (start.y() + end.y()) / 2
    rect = QRectF(center_x - size / 2, center_y - size / 2, size, size)

    path = QPainterPath()
    path.moveTo(center_x, center_y)
    path.arcTo(rect, 0, 180)
    path.lineTo(center_x, center_y)

    if rotation_angle != 0:
        transform = QTransform()
        transform.translate(center_x, center_y)
        transform.rotate(rotation_angle)
        transform.translate(-center_x, -center_y)
        path = transform.map(path)

    return path



