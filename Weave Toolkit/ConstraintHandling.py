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
    QPainterPathStroker,
    QTransform
)
from GlobalVariables import (
    getSymmetryLine,
    getDegreeLinesTop,
    setDegreeLinesTop,
    setDegreeLinesBot,
    getDegreeLinesBot,
    getShapeMode,
    getClassicCells,
    setCellAdjacencyMap,
    getCellAdjacencyMap,
    getCurrentPatternType
)
import math
from ShapeMode import ShapeMode
from PatternType import PatternType 


# Shows a warning as a tooltip
def showWarningTooltip(message, position = None):
    if position == None:
        position = QCursor.pos()
    QToolTip.showText(position, message, msecShowTime=3000)

# Utility function for debugging
def get_shape_cell_indices(shape_path):
    """
    Returns a list of cell indices that the given shape path intersects with.
    """
    cells = getClassicCells()
    #print("length: ", len(getClassicCells()))
    #print("the cells: ", getClassicCells())
    
    indices = []

    for polygon, index in cells:
        cell_path = QPainterPath()
        cell_path.addPolygon(polygon)

        if shape_path.intersects(cell_path):
            indices.append(index)

    print(indices)

# Gets the outermost cell indices
def get_outer_cell_indices(rows, cols):
    outer_indices = set()

    # Top row
    outer_indices.update(range(cols))

    # Bottom row
    outer_indices.update(range((rows - 1) * cols, rows * cols))

    # Left column
    outer_indices.update(i * cols for i in range(rows))

    # Right column
    outer_indices.update((i + 1) * cols - 1 for i in range(rows))

    return sorted(outer_indices)

## Detects if the semicircle is about to get snapped to the border
def does_semicircle_snap_to_border_error(semicircle_shape, borders):
    semi_path = createSemicirclePath(semicircle_shape[0], semicircle_shape[1])
    border_paths = convert_borders_to_paths(borders)
    return semicircle_intersects_border(semi_path, border_paths)

def semicircle_intersects_border(semicircle_path, border_paths):
    for cut_path in border_paths:
        if semicircle_path.intersects(cut_path):
            showWarningTooltip("Semicircles cannot be placed on the border")
            return True  # Intersects the border
    return False


## Detects if a shape can be placed in a cell as well as prevent shapes from being placed in the same cell and adjacent cells
def is_shape_placement_valid(shape, shapes):
    new_path = shape_to_path(shape)
    shape_type = shape[2]
    new_index = get_shape_cell_index(new_path, shape_type, shape_points=(shape[0], shape[1]))
    cell_adjacency_map = getCellAdjacencyMap()

    if new_index is None and shape_type != ShapeMode.Semicircle:
        return False
    
    #print("index: ", new_index)
    total_cells = len(getClassicCells())
    rows = cols = int(total_cells**0.5)  # assumes square
    outer_cells = get_outer_cell_indices(rows, cols)

    # Checks if a non-semicircle is about to be placed in the outermost cells
    if new_index in outer_cells and shape_type != ShapeMode.Semicircle:   
        showWarningTooltip("Only semicircles can be placed in the outermost cells")
        return False  # Not in a valid cell
    
    # Checks if the semicircles placed in outermost cells covers more than one cell by looking at
    # the distance between the borders and the semicircle
    #elif shape_type == ShapeMode.Semicircle:
    #    semicircle_height = get_semicircle_size(shape[0], shape[1])["height"]
    #    cell_height = get_cell_size(new_index)["height"]
    #    if abs(semicircle_height) > cell_height:
    #        showWarningTooltip("Semicircles in the outermost cells cannot be larger than one cell")
    #        return False
            
    for existing_shape in shapes:
        existing_path = shape_to_path(existing_shape)
        existing_shape_type = existing_shape[2]
        existing_index = get_shape_cell_index(
            existing_path,
            existing_shape_type,
            shape_points=(existing_shape[0], existing_shape[1])
            )
        
        
        # Checks if shapes a placed in a cell that shares a border with another shapes cell
        if existing_index in cell_adjacency_map.get(new_index, []) and shape_type != ShapeMode.Semicircle:
            showWarningTooltip("Shapes cannot be placed in adjacent cells")
            return False
            
        # Checks if shapes are about to be placed in the same cell
        elif existing_index == new_index:
            showWarningTooltip("Shapes cannot be placed in the same cell")
            return False  # Same cell occupied
        
    return True

# Tells if a line slope is positive
def is_positive_slope(p1, p2):
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            if dx == 0:
                return False  # vertical line case, can adjust if needed
            return dy / dx > 0


# Gets the index of the cell of a shape
def get_shape_cell_index(shape_path, shape_type=None, shape_points=None):
    """
    shape_path: QPainterPath of the shape
    shape_type: the ShapeMode enum of the shape (e.g., ShapeMode.Semicircle)
    shape_points: tuple (start_point: QPointF, end_point: QPointF) used only for semicircles
    """
    cells = getClassicCells()

    if shape_type == ShapeMode.Semicircle and shape_points:
        start, end = shape_points

        # Midpoint between start and end
        mid_x = (start.x() + end.x()) / 2
        mid_y = (start.y() + end.y()) / 2

        # Angle between start and end (in radians)
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        angle = math.atan2(dy, dx)  # From start to end

        # Normal (perpendicular) angle pointing toward the flat base = bottom
        normal_angle = angle - math.pi / 2  # ⬅ opposite of arc direction

        # Approximate radius (half distance between start and end)
        radius = math.hypot(dx, dy) / 2

        # Bottom point lies along the normal vector from the midpoint
        bottom_x = mid_x + radius * math.cos(normal_angle)
        bottom_y = mid_y + radius * math.sin(normal_angle)
        test_point = QPointF(bottom_x, bottom_y)

    else:
        # Use center for other shapes
        test_point = shape_path.boundingRect().center()

    for cell, index in cells:
        if cell.containsPoint(test_point, Qt.FillRule.OddEvenFill):
            return index

    return None

# Utility function for get_semicircle_size
def find_direction(start_point, end_point):
    # Calculates the angle
    dx = end_point.x() - start_point.x()
    dy = end_point.y() - start_point.y()
    angle_rad = math.atan2(dy, dx)  # angle in radians
    angle_deg = math.degrees(angle_rad) % 360  # normalize angle

    # Return only 'up' or 'down' based on vertical orientation
    if 90 < angle_deg < 270:
        return "up"
    else:
        return "down"

# Gets the size of semicircles and cells to use for checking if a semicircle is too large in the outermost cells
def get_semicircle_size(start_point, end_point):
    rect = QRectF(QPointF(start_point), QPointF(end_point))
    width = rect.width()
    height = rect.height()
    radius_x = width / 2
    radius_y = height / 2
    direction = find_direction(start_point,end_point)
    return {
        "width": width,
        "height": height,
        "radius_x": radius_x,
        "radius_y": radius_y,
        "direction": direction
    }

def get_cell_size(cell_index):
    cells = getClassicCells()
    polygon = None
    for poly, index in cells:
        if index == cell_index:
            polygon = poly   
    if polygon:
        rect = polygon.boundingRect()
        return {
            "width": rect.width(),
            "height": rect.height(),
            "area": rect.width() * rect.height()
            }
    else:
        return cells[0][0]

# Builds an cells adjacency map for detecting of shapes adjacencies
def build_cell_adjacency_map(tolerance=0.01):
    cells = getClassicCells()
    adjacency = {index: set() for _, index in cells}

    def edge_list(poly):
        return [(poly[i], poly[(i + 1) % poly.count()]) for i in range(poly.count())]

    def points_are_close(p1, p2):
        return (p1 - p2).manhattanLength() < tolerance

    for i, (poly1, idx1) in enumerate(cells):
        edges1 = edge_list(poly1)
        for j in range(i + 1, len(cells)):
            poly2, idx2 = cells[j]
            edges2 = edge_list(poly2)
            for a1, a2 in edges1:
                for b1, b2 in edges2:
                    if (points_are_close(a1, b1) and points_are_close(a2, b2)) or \
                       (points_are_close(a1, b2) and points_are_close(a2, b1)):
                        adjacency[idx1].add(idx2)
                        adjacency[idx2].add(idx1)
                        break
    setCellAdjacencyMap(adjacency)



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
    top_line_left, top_line_right = getDegreeLinesTop()
    top_45_path_left = make_angled_line_path(top_line_left[0], top_line_left[1])
    top_45_path_right = make_angled_line_path(top_line_right[0], top_line_right[1])
    
    if getCurrentPatternType() == PatternType.Asymmetric:
        bot_line_left, bot_line_right = getDegreeLinesBot()
        bottom_45_path_left = make_angled_line_path(bot_line_left[0], bot_line_left[1])
        bottom_45_path_right = make_angled_line_path(bot_line_right[0], bot_line_right[1])
    
    if len(shapes) < 2:
        return True
    
    for shape in shapes:
        path = shape_to_path(shape)

        if path.intersects(top_45_path_left) or path.intersects(top_45_path_right):
            showWarningTooltip("A shape is placed more than 45 degrees from the topmost points")
            return False
        
        elif getCurrentPatternType() == PatternType.Asymmetric and (path.intersects(bottom_45_path_left) or path.intersects(bottom_45_path_right)):
            showWarningTooltip("A shape is placed more than 45 degrees from the bottommost points")
            return False
    
    return True

def find_top_and_bottommost_points(shape_list, error_margin=2):
    top_y = float('inf')
    bottom_y = float('-inf')
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None

    for shape in shape_list:
        if shape[2] == ShapeMode.Semicircle:
            path = semicircle_to_path(shape[0], shape[1])
        
        else:
            path = shape_to_path(shape)

        # For circles and semicircles, inspect individual path elements
        if getShapeMode() == ShapeMode.Circle or getShapeMode() == ShapeMode.Semicircle:
            for i in range(path.elementCount()):
                pt = path.elementAt(i)
                point = QPointF(pt.x, pt.y)

                # Top
                if abs(pt.y - top_y) <= error_margin:
                    if top_left is None or pt.x < top_left.x():
                        top_left = point
                    if top_right is None or pt.x > top_right.x():
                        top_right = point
                elif pt.y < top_y:
                    top_y = pt.y
                    top_left = top_right = point

                # Bottom
                if abs(pt.y - bottom_y) <= error_margin:
                    if bottom_left is None or pt.x < bottom_left.x():
                        bottom_left = point
                    if bottom_right is None or pt.x > bottom_right.x():
                        bottom_right = point
                elif pt.y > bottom_y:
                    bottom_y = pt.y
                    bottom_left = bottom_right = point

        else:
            rect = path.boundingRect()
            # Top candidates
            t_left = QPointF(rect.left() - error_margin, rect.top() - error_margin)
            t_right = QPointF(rect.right() + error_margin, rect.top() - error_margin)
            # Bottom candidates
            b_left = QPointF(rect.left() - error_margin, rect.bottom() + error_margin)
            b_right = QPointF(rect.right() + error_margin, rect.bottom() + error_margin)

            if rect.top() < top_y - error_margin:
                top_y = rect.top()
                top_left = t_left
                top_right = t_right
            elif abs(rect.top() - top_y) <= error_margin:
                if top_left is None or t_left.x() < top_left.x():
                    top_left = t_left
                if top_right is None or t_right.x() > top_right.x():
                    top_right = t_right

            if rect.bottom() > bottom_y + error_margin:
                bottom_y = rect.bottom()
                bottom_left = b_left
                bottom_right = b_right
            elif abs(rect.bottom() - bottom_y) <= error_margin:
                if bottom_left is None or b_left.x() < bottom_left.x():
                    bottom_left = b_left
                if bottom_right is None or b_right.x() > bottom_right.x():
                    bottom_right = b_right

    return top_left, top_right, bottom_left, bottom_right

def draw_dotted_45degree_lines(qp, shapes):
    if len(shapes) < 1:
        return None
    
    symmetry_x, sym_y1, sym_y2 = getSymmetryLine()
    
    # Filter shapes that intersect the symmetry line
    def intersects_symmetry_line(shape):
        path = shape_to_path(shape)
        rect = path.boundingRect()
        return rect.left() <= symmetry_x <= rect.right()

    touching_shapes = [s for s in shapes if intersects_symmetry_line(s)]

    if not touching_shapes:
        return  # No shapes touch the symmetry line, so don't draw lines

    # Get extreme points only from shapes that touch the symmetry line
    top_left, top_right, bottom_left, bottom_right = find_top_and_bottommost_points(touching_shapes)

    def draw_angle_line(origin_point, angle_deg, length=400):
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        end_point = QPointF(origin_point.x() + length * dx, origin_point.y() + length * dy)
        qp.drawLine(origin_point, end_point)
        return [origin_point, end_point]
    
    
    # --- Lines from topmost corners ---
    left_top_line = draw_angle_line(top_left, 225)   # 225° = up-left
    right_top_line = draw_angle_line(top_right, 315)  # 315° = up-right
    
    if PatternType.Asymmetric == getCurrentPatternType():
        # --- Lines from bottommost corners ---
        left_bottom_line = draw_angle_line(bottom_left, 135)  # 135° = down-left
        right_bottom_line = draw_angle_line(bottom_right, 45)  # 45° = down-right
        setDegreeLinesBot(left_bottom_line, right_bottom_line)
    
    setDegreeLinesTop(left_top_line, right_top_line)

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

## Converts shapes, borders, 45 degree lines and the symmetry line into QPainterPaths for intersection detection

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


def make_angled_line_path(start_point: QPointF, end_point: QPointF) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(start_point)
    path.lineTo(end_point)
    return path

def convert_borders_to_paths(borders):
    path_list = []
    for cut in borders:
        x1, y1, x2, y2 = cut
        path = QPainterPath()
        path.moveTo(QPointF(x1, y1))
        path.lineTo(QPointF(x2, y2))
        path_list.append(path)
    return path_list

def createSemicirclePath(start_point, end_point):
    rect = QRectF(QPointF(start_point), QPointF(end_point))
    path = QPainterPath()
    path.moveTo(rect.center())
    path.arcTo(rect, 0, 180)
    return path

# An alternative version of createSemiCirclePath, which is only used to find the topmost and bottommost points
def semicircle_to_path(start, end, rotation_angle=0):
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



