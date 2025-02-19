import sys
import math
import cv2 as cv
import numpy as np

from functools import partial

#from PatternOutput import (
#    WeaveView
#)

from ShapeMode import (
    ShapeMode
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
    QSlider
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
    QImage,
    QTransform,
    QPolygon
)
from Algorithm import (
    mainAlgorithm
)

# Global variables for the shape mode and shape color
SHAPE_MODE = ShapeMode.Cursor
SHAPE_COLOR = QColor(0, 0, 0, 255)
BACKGROUND_COLOR = QColor(255, 255, 255, 255)
PEN_WIDTH = 3

class DrawingWidget(QWidget):
    # Defining the initial state of the canvas
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.shapes = [] #Stores shapes
        self.free_form_points = []  # Store points for free form drawing
        self.setGeometry(30,30,600,400)
        self.begin = QPoint()
        self.end = QPoint()
        self.drawing_mode = False
        self.show()

    # Draws the current canvas state
    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setBrush(SHAPE_COLOR)
        # Fill the background with light gray color
        qp.fillRect(self.rect(), Qt.GlobalColor.lightGray)

        self.drawRotatedSquareEffect(qp)

        # Redraw all the previous shapes
        self.redrawAllShapes(qp)

        qp.setBrush(SHAPE_COLOR)
        qp.setPen(QPen(SHAPE_COLOR, PEN_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

        # Draws the current shape as it is being created
        if self.begin != self.end:
            if (SHAPE_MODE == ShapeMode.Square):
                qp.drawRect(QRect(self.begin, self.end))
            elif (SHAPE_MODE == ShapeMode.Circle):
                center = self.begin
                radius = int((self.begin-self.end).manhattanLength() / 2)
                qp.drawEllipse(center, radius, radius)
            elif (SHAPE_MODE == ShapeMode.Heart):
                self.drawHeart(qp, self.begin, self.end, SHAPE_COLOR, PEN_WIDTH)
            elif (SHAPE_MODE == ShapeMode.Line):
                qp.drawLine(self.begin, self.end)
            elif SHAPE_MODE == ShapeMode.FreeForm:
                for free_form_point in range(len(self.free_form_points) - 1):
                    qp.drawLine(self.free_form_points[free_form_point], self.free_form_points[free_form_point + 1])

        self.redrawBorder(qp)

    def drawRotatedSquareEffect(self, qp):
        pen = QPen(Qt.GlobalColor.black, 3)
        qp.setPen(pen)
        brush = QBrush(Qt.GlobalColor.lightGray)
        qp.setBrush(brush)

        width, height = self.width(), self.height()
        margin = 0

        # Coordinates of the corners of the outer square
        x1, y1 = margin, margin
        x2, y2 = width - margin, height - margin

        # Drawing outer square
        qp.drawRect(x1, y1, x2 - x1, y2 - y1)

        # Calculate the center and half-diagonal
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Coordinates of the corners of the inner rotated square
        inner_coords = [
            (center_x, y1),
            (x2, center_y),
            (center_x, y2),
            (x1, center_y)
        ]

        # Drawing inner rotated square with selected background fill
        brush = QBrush(BACKGROUND_COLOR)
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

        pen = QPen(SHAPE_COLOR, PEN_WIDTH)
        qp.setPen(pen)
        brush = QBrush(SHAPE_COLOR)
        qp.setBrush(brush)

    # Redraws all the shapes, while removing the ones that are erased
    def redrawAllShapes(self, qp):
        for shape in self.shapes[:]:  # Use a copy of the list to avoid modification issues
            shape_type = shape[2]
            qp.setBrush(SHAPE_COLOR) # set to shape[3] if we want to change color to stored shape color instead of global color
            qp.setPen(QPen(SHAPE_COLOR, PEN_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)) # pen width should needs to be saved in the shape list
            # if in eraser mode, removes shapes that contain the point clicked
            if SHAPE_MODE == ShapeMode.Eraser:
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
                elif shape_type == ShapeMode.FreeForm:
                    for free_form_point in range(len(shape[4]) - 1):
                        if self.lineContainsPoint(point, shape[4][free_form_point], shape[4][free_form_point + 1]):
                            self.shapes.remove(shape)
                            break  # Skip drawing since it's erased
            # Draw the shape if not in eraser mode
            if shape_type == ShapeMode.Square:
                qp.drawRect(QRect(shape[0], shape[1]))
            elif shape_type == ShapeMode.Circle:
                center = shape[0]
                radius = int((abs(center.x() - shape[1].x()) + abs(center.y() - shape[1].y())) / 2)
                qp.drawEllipse(center, radius, radius)
            elif shape_type == ShapeMode.Heart:
                self.drawHeart(qp, shape[0], shape[1], SHAPE_COLOR, shape[5])
            elif shape_type == ShapeMode.Line:
                qp.setPen(QPen(SHAPE_COLOR, shape[5], Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                qp.drawLine(shape[0], shape[1])
            elif shape_type == ShapeMode.FreeForm:
                qp.setPen(QPen(SHAPE_COLOR, shape[5], Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                for free_form_point in range(len(shape[4]) - 1):
                    qp.drawLine(shape[4][free_form_point], shape[4][free_form_point + 1])
            qp.setPen(QPen(SHAPE_COLOR, PEN_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

    def redrawBorder(self, qp):
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

        # Calculate the center and half-diagonal
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Coordinates of the corners of the inner rotated square
        inner_coords = [
            (center_x, y1),
            (x2, center_y),
            (center_x, y2),
            (x1, center_y)
        ]

        # Draw the edges of the inner rotated square
        for i in range(len(inner_coords)):
            qp.drawLine(int(inner_coords[i][0]), int(inner_coords[i][1]),
                             int(inner_coords[(i+1) % len(inner_coords)][0]),
                             int(inner_coords[(i+1) % len(inner_coords)][1]))

        brush = QBrush(SHAPE_COLOR)
        qp.setBrush(brush)
        pen = QPen(SHAPE_COLOR, PEN_WIDTH)
        qp.setPen(pen)

    def get_drawing_image(self):
        image = QImage(self.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)  # Fill with transparent color
        painter = QPainter(image)
        self.render(painter)  # Render the current drawing to the image
        return image

    def drawHeart(self, qp, start, end, color, pen_width):
        qp.setBrush(color)
        qp.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
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

        qp.setPen(QPen(SHAPE_COLOR, PEN_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

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
            if SHAPE_MODE == ShapeMode.FreeForm:
                self.free_form_points = [event.position().toPoint()]
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_mode:
            self.end = event.pos()
            if SHAPE_MODE == ShapeMode.FreeForm:
                self.free_form_points.append(event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_mode:
            if SHAPE_MODE == ShapeMode.FreeForm:
                self.shapes.append([self.begin, self.end, SHAPE_MODE, SHAPE_COLOR, list(self.free_form_points), PEN_WIDTH])
            else:
                if (SHAPE_MODE == ShapeMode.Line) or (SHAPE_MODE == ShapeMode.Heart):
                    self.shapes.append([self.begin, self.end, SHAPE_MODE, SHAPE_COLOR, [], PEN_WIDTH])
                else:
                    self.shapes.append([self.begin, self.end, SHAPE_MODE, SHAPE_COLOR])

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

    def __init__(self):
        super().__init__()

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
        self.backside_label = QLabel("Back Side (not modifiable):")
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
        # main_layout.addWidget(self.drawing_widget)

        # Create the output display widget
        #self.display_widget = QLabel(self)
        #pixmap = QPixmap("Weave Toolkit/icons/20201208_203402.jpg")
        #self.display_widget.setPixmap(pixmap)
        #self.display_widget.hide()  # Hide initially
        #main_layout.addWidget(self.display_widget)

        # Graphic Screen
        self.scene = QGraphicsScene()

        # Create the WeaveView and add to the layout
        #self.weave_widget = WeaveView(self.scene)  # This is where the grid rendering happens
        #self.weave_widget.hide()  # Hide initially, show it on button click
        #main_layout.addWidget(self.weave_widget)

        # Create a stacked widget to switch between views
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.addWidget(self.drawing_container)
        #self.stacked_widget.addWidget(self.weave_widget)
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

        update_button = QPushButton("Update", self)
        update_button.setStyleSheet("background-color: lightgray; color: black;")
        update_button.clicked.connect(lambda: self.updateDisplay())
        menu_toolbar.addWidget(update_button)

        edit_button = QPushButton("Edit", self)
        edit_button.setStyleSheet("background-color: lightgray; color: black;")
        edit_button.clicked.connect(lambda: self.editDisplay())
        menu_toolbar.addWidget(edit_button)

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
        action_open = QAction("Open", self)
        action_save = QAction("Save", self)
        action_save.triggered.connect(lambda: self.save_canvas_as_png())
        action_export = QAction("Export", self)
        action_export.triggered.connect(lambda: self.exportHeart())
        action_undo = QAction("Undo (shrt cut: ctrl + z)", self) # Need to implement stack to store shapes

        # Add actions to the menu
        file_menu.addAction(action_new)
        file_menu.addAction(action_open)
        file_menu.addAction(action_save)
        file_menu.addAction(action_export)
        file_menu.addAction(action_undo)

        return file_menu

    def createViewDropdownMenu(self):
        view_menu = QMenu("View", self)
        view_menu.setStyleSheet("color: black;")
        action_zoom = QAction("Zoom", self)
        action_fullscreen = QAction("Fullscreen", self)
        action_gridlines = QAction("Toggle Gridlines", self)
        action_show_backside = QAction("Show/Hide Backside", self)
        action_background_color = QAction("Change Background Color", self)
        action_print_size = QAction("Change Print Size", self)
        view_menu.addAction(action_zoom)
        view_menu.addAction(action_fullscreen)
        view_menu.addAction(action_gridlines)
        view_menu.addAction(action_show_backside)
        view_menu.addAction(action_background_color)
        view_menu.addAction(action_print_size)

        return view_menu

    def updateDisplay(self, write_to_image = False):
        #self.stacked_widget.setCurrentWidget(self.weave_widget)
        #self.weave_widget.setShapes(self.drawing_widget.shapes)
        #self.save_canvas_as_png()

        self.backside_label.setText("Front Side final product:")

        arr = self.pixmapToCvImage()
        #mainAlgorithm(arr,'create')
        heart = self.cvImageToPixmap(mainAlgorithm(arr, 'show'))

        if write_to_image:
            cv.imwrite('image.png', heart)

        # Shows the design created by the users on the heart
        pixmap = QPixmap(heart)
        self.drawing_backside.setPixmap(pixmap)
        self.drawing_backside.setScaledContents(True)


    def editDisplay(self):
        self.backside_label.setText("Back Side (not modifiable):")
        self.stacked_widget.setCurrentWidget(self.drawing_widget)
        #self.display_widget.hide()
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

        return shapes_toolbar

    def createShapeButton(self, icon_path, button_text, shape_mode):
        shape_button = QAction(QIcon(icon_path), button_text, self)
        # shape_button.setStatusTip(status_tip)
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

        global SHAPE_MODE
        SHAPE_MODE = shape_mode


    def createColorsToolbar(self):
        colors_toolbar = QToolBar("Colors toolbar")

        foreground_label = QLabel("Foreground Colors: ")
        foreground_label.setStyleSheet("color: black;")
        colors_toolbar.addWidget(foreground_label)

        foreground_colors = [("Red", "red"), ("Green", "green"), ("Orange", "orange"), ("Blue", "blue")]
        for color_name, color_value in foreground_colors:
            button = QPushButton(color_name, self, styleSheet=f"background-color: {color_value}")
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
            button = QPushButton(color_name, self, styleSheet=f"background-color: {color_value}")
            button.clicked.connect(partial(self.change_background_color, color_value))
            colors_toolbar.addWidget(button)

        colors_toolbar.addWidget(self.addStrokeWidthWidget())

        return colors_toolbar

    def addStrokeWidthWidget(self):
        self.stroke_width_layout = QVBoxLayout()
        
        # Create a label to show the current stroke width
        initial_stroke_width = 3
        self.stroke_width_label = QLabel(f'Stroke Width: {initial_stroke_width}', self)
        self.stroke_width_label.setStyleSheet("color: black;")
        self.stroke_width_layout.addWidget(self.stroke_width_label)
        
        # Create a slider for selecting stroke width
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
        global PEN_WIDTH
        PEN_WIDTH = value

    def change_foreground_color(self, color):
        global SHAPE_COLOR
        SHAPE_COLOR = QColor(color)
        self.update()
        self.update_backside_image()

    def change_background_color(self, color):
        global BACKGROUND_COLOR
        BACKGROUND_COLOR = QColor(color)
        self.update()
        self.update_backside_image()

    def save_canvas_as_png(self, filename="canvas_output.png"):
        pixmap = QPixmap(self.drawing_widget.size())  # Create pixmap of the same size
        self.drawing_widget.render(pixmap)  # Render the widget onto the pixmap
        pixmap.save(filename, "PNG")  # Save as PNG
    def exportHeart(self):
        arr = self.pixmapToCvImage()
        mainAlgorithm(arr,'create')

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


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()