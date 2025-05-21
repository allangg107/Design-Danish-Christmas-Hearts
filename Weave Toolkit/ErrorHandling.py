from PyQt6.QtWidgets import (
    QMessageBox,
    QToolTip
)
from PyQt6.QtCore import (
    QPoint,
    QPointF,
    QRectF,
    Qt
)
from PyQt6.QtGui import (
    QCursor,
    QPainterPath,
    QPen,
    QPainterPathStroker
)
from GlobalVariables import (
    getSymmetryLine,
    getDegreeLineLeft,
    getDegreeLineRight,
    SetDegreeLineLeft,
    SetDegreeLineRight,
    getShapeMode
)
import math
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


## Detects if shapes do not touch the symmetryline

def shapeNotTouchingSymmetrylineError(shapes):
    symmetry_line = getSymmetryLine()
    for shape in shapes:
        if any_shape_intersects_symmetry_line(shapes, symmetry_line[0], symmetry_line[1], symmetry_line[2]):
            return True
    
    showWarningTooltip("At least one shape needs to touch the line of symmetry")
    return False

def any_shape_intersects_symmetry_line(shapes, x_symmetry, y1_line, y2_line):
    symmetry_path = symmetryLineToPath(x_symmetry, y1_line, y2_line)
    
    for shape in shapes:
        shape_path = shape_to_path(shape)
        if shape_path.intersects(symmetry_path):
            return True  # At least one shape intersects
    return False

## Detects if a shape is more than 45 degrees to the lower-left for the topmost point
##  and more than 45 degrees to upper-right for the bottommost point

def MoreThan45DegreesError(shapes):
    start_top, end_top = getDegreeLineLeft()
    start_bottom, end_bottom = getDegreeLineRight()

    top_45_path = make_angled_line_path(start_top, end_top)
    bottom_45_path = make_angled_line_path(start_bottom, end_bottom)
    
    if len(shapes) < 2:
        return True
    
    for shape in shapes:
        path = shape_to_path(shape)

        if path.intersects(top_45_path):
            showWarningTooltip("A shape is placed more than 45 degrees from either the top or bottom point")
            return False
        
        elif path.intersects(bottom_45_path):
            showWarningTooltip("A shape is placed more than 45 degrees from either the top or bottom point")
            return False
    
    return True

def find_top_and_bottommost_points(shape_list, error_margin=5):
    topmost_point = None
    bottommost_point = None
    min_y = float('inf')
    max_y = float('-inf')

    for shape in shape_list:
        path = shape_to_path(shape)

        if ShapeMode.Circle == getShapeMode():
            for i in range(path.elementCount()):
                pt = path.elementAt(i)
                
                if pt.y < min_y:
                    min_y = pt.y
                    topmost_point = QPointF(pt.x, pt.y)
                
                if pt.y > max_y:
                    max_y = pt.y
                    bottommost_point = QPointF(pt.x, pt.y)
        else: 
            bounding_rect = path.boundingRect()
            top_candidate = QPointF(bounding_rect.left() - error_margin, bounding_rect.top() - error_margin)
            bottom_candidate = QPointF(bounding_rect.right() + error_margin, bounding_rect.bottom() + error_margin)
            
            if top_candidate.y() < min_y:
                min_y = top_candidate.y()
                topmost_point = top_candidate

            if bottom_candidate.y() > max_y:
                max_y = bottom_candidate.y()
                bottommost_point = bottom_candidate
            

    return topmost_point, bottommost_point

# Draws dotted lines at 45 degrees to inform the users of where they are allowed to draw for the pattern to work
def draw_dotted_45degree_lines(qp, shapes, width, height):
    if len(shapes) < 1:
        return None
    top, bottom = find_top_and_bottommost_points(shapes)

    # 1. Line from topmost point down-left (45°)
    angle_rad_top = math.radians(135)  # 135° = down-left
    dx_top = math.cos(angle_rad_top)
    dy_top = math.sin(angle_rad_top)
    scale_top = max(
        (top.x() / abs(dx_top)) if dx_top != 0 else float('inf'),
        ((height - top.y()) / abs(dy_top)) if dy_top != 0 else float('inf')
    )
    end_top = QPointF(top.x() + scale_top * dx_top, top.y() + scale_top * dy_top)
    qp.drawLine(top, end_top)
    SetDegreeLineLeft([top, end_top])
    

    # 2. Line from bottommost point up-right (45°)
    angle_rad_bottom = math.radians(-45)  # -45° = up-right
    dx_bottom = math.cos(angle_rad_bottom)
    dy_bottom = math.sin(angle_rad_bottom)
    scale_bottom = max(
        ((width - bottom.x()) / abs(dx_bottom)) if dx_bottom != 0 else float('inf'),
        (bottom.y() / abs(dy_bottom)) if dy_bottom != 0 else float('inf')
    )
    end_bottom = QPointF(bottom.x() + scale_bottom * dx_bottom, bottom.y() + scale_bottom * dy_bottom)
    qp.drawLine(bottom, end_bottom)
    SetDegreeLineRight([bottom, end_bottom])

## Detects if all shapes are connected together

def allShapesOverlapError(shapes):
    if len(shapes) < 2:
        return True
    
    if check_all_shapes_overlap(shapes):
        return True
    
    showWarningTooltip("All shapes needs to be combined into one shape")
    return False
  

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

## Converts shapes and the symmetry line into QPainterPaths for intersection detection

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

def make_angled_line_path(start_point: QPointF, end_point: QPointF) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(start_point)
    path.lineTo(end_point)
    return path




