from PyQt6.QtWidgets import (
    QMessageBox,
    QToolTip
)
from PyQt6.QtCore import (
    QPoint,
    QPointF
)
from PyQt6.QtGui import QCursor
from GlobalVariables import (
    getSymmetryLine
)

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

# Checks if any shape intersects with the symmetryline
def intersects_vertical_line_with_segment(x_dashed, y1_line, y2_line, old_start, old_end, tolerance=1e-6):
    if isinstance(old_start, QPoint) or isinstance(old_start, QPointF):
        start = (old_start.x(), old_start.y())
        end = (old_end.x(), old_end.y())
    else:
        start = old_start
        end = old_end
    # 1. Check horizontal overlap
    if (x_dashed < min(start[0], end[0]) - tolerance) or (x_dashed > max(start[0], end[0]) + tolerance):
        return False

    # 2. Handle vertical segment (avoid division by zero)
    if abs(end[0] - start[0]) < tolerance:
        # If the segment itself is vertical, check if x matches and y-ranges overlap
        if abs(x_dashed - start[0]) > tolerance:
            return False
        # Check y overlap with line segment and dashed line
        return not (max(start[1], end[1]) < y1_line or min(start[1], end[1]) > y2_line)

    # 3. Calculate slope and find y at x = x_dashed
    slope = (end[1] - start[1]) / (end[0] - start[0])
    y_at_x_dashed = start[1] + slope * (x_dashed - start[0])

    # 4. Check if y_at_x_dashed is within both shape segment y-range and dashed line y-range
    in_shape_y_range = min(start[1], end[1]) - tolerance <= y_at_x_dashed <= max(start[1], end[1]) + tolerance
    in_line_y_range = y1_line <= y_at_x_dashed <= y2_line

    return in_shape_y_range and in_line_y_range


def shapeNotTouchingSymmetrylineError(shapes):
    symmetry_line = getSymmetryLine()
    for shape in shapes:
        if intersects_vertical_line_with_segment(symmetry_line[0], symmetry_line[1], symmetry_line[2], shape[0], shape[1]):
            return True
    
    showWarningTooltip("At least one shape needs to touch the line of symmetry")
    return False
