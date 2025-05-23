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
    getDegreeLineLeft,
    getDegreeLineRight,
    setDegreeLineLeft,
    setDegreeLineRight,
    getShapeMode,
    getClassicCells,
    setCellAdjacencyMap,
    getCellAdjacencyMap,
    getCellAdjacencyCheck,
    setCellAdjacencyCheck
)
import math
from ShapeMode import ShapeMode
from shapely.geometry import LineString

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
    cell_height = get_cell_size()["height"]
    cell_adjacency_map = getCellAdjacencyMap()
    
    if new_index is None and shape_type != ShapeMode.Semicircle:
        return False
    
    #print("index: ", new_index)
    total_cells = len(getClassicCells())
    rows = cols = int(total_cells**0.5)  # assumes square
    outer_cells = get_outer_cell_indices(rows, cols)

    if getCellAdjacencyCheck() >= 2:
        return True

    # Checks if a non-semicircle is about to be placed in the outermost cells
    if new_index in outer_cells and shape_type != ShapeMode.Semicircle:   
        showWarningTooltip("Only semicircles can be placed in the outermost cells")
        return False  # Not in a valid cell
    
    # Checks if the semicircles placed in outermost cells covers more than one cell by looking at
    # the height of the semicircle in comparison to cells
    elif new_index in outer_cells and shape_type == ShapeMode.Semicircle:
        semicircle_height = get_semicircle_size(shape[0], shape[1])["height"]
        if abs(semicircle_height) > cell_height:
            showWarningTooltip("Semicircles in the outermost cells cannot be larger than one cell")
            return False
    
    # Updates the rules for nesting purposes
    elif shape_type == ShapeMode.Semicircle:
        semicircle_height = get_semicircle_size(shape[0], shape[1])["height"]
        semicircle_direction = get_semicircle_size(shape[0], shape[1])["direction"]
        #print(semicircle_direction)
        if abs(semicircle_height) > cell_height and semicircle_direction == "down":
            print("here")
            setCellAdjacencyCheck(getCellAdjacencyCheck() + 1) 

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
            if 2 > getCellAdjacencyCheck():
                showWarningTooltip("Shapes cannot be placed in adjacent cells")
                return False
            
        # Checks if shapes are about to be placed in the same cell
        elif existing_index == new_index:
            showWarningTooltip("Shapes cannot be placed in the same cell")
            return False  # Same cell occupied
        
    return True

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

def get_cell_size():
    cells = getClassicCells()
    sizes = []
    polygon = cells[0][0]   
    rect = polygon.boundingRect()
    return {
            "width": rect.width(),
            "height": rect.height(),
            "area": rect.width() * rect.height()
        }

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
    setDegreeLineLeft([top, end_top])
    

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
    setDegreeLineRight([bottom, end_bottom])

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


