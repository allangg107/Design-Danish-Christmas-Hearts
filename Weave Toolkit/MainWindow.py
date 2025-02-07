import sys
import math

from functools import partial

from PatternOutput import (
    WeaveView
)

from ShapeMode import (
    ShapeMode
)

from PyQt6.QtCore import (
    QSize,
    Qt,
    QRectF,
    QRect,
    QPoint,
    QPointF
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
    QGraphicsScene
)

from PyQt6.QtGui import (
    QColor,
    QAction,
    QIcon,
    QPainter,
    QPen,
    QBrush,
    QPixmap,
    QPainterPath
)

# Global variable for the shape mode
SHAPE_MODE = ShapeMode.Cursor
SHAPE_COLOR = QColor(0, 0, 0, 255)


class DrawingWidget(QWidget):
    # Defining the initial state
    def __init__(self):
        super().__init__()
        self.shapes = [] #Stores shapes
        self.setGeometry(30,30,600,400)
        self.begin = QPoint()
        self.end = QPoint()
        self.drawing_mode = False

        #start = QPoint(200, 200)  # Top-left corner of the heart
        #end = QPoint(300, 300)  # Bottom-right corner of the heart
        #self.shapes.append([start, end, ShapeMode.Heart, QColor(255, 0, 0, 255)])

        self.show()

    # Draws the current shape
    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setBrush(SHAPE_COLOR)

        # Redraw all the previous shapes
        self.redrawAllShapes(qp)

        # Draw the current shape being created
        qp.setBrush(SHAPE_COLOR)

        if self.begin != self.end:
            if (SHAPE_MODE == ShapeMode.Square):
                qp.drawRect(QRect(self.begin, self.end))
            elif (SHAPE_MODE == ShapeMode.Circle):
                center = self.begin
                radius = int((self.begin-self.end).manhattanLength() / 2)
                qp.drawEllipse(center, radius, radius)
            elif (SHAPE_MODE == ShapeMode.Heart):
                WeaveView.drawHeart(self, qp, self.begin, self.end)


    # Redraws all the shapes, while removing the ones that are erased
    def redrawAllShapes(self, qp):
        for shape in self.shapes[:]:  # Use a copy of the list to avoid modification issues
            shape_type = shape[2]
            qp.setBrush(shape[3])
            # if in eraser mode, removes shapes that contain the point clicked
            if SHAPE_MODE == ShapeMode.Eraser:
                point = self.begin
                if shape_type == ShapeMode.Square:
                    rect = QRect(shape[0], shape[1])
                    if rect.contains(point):
                        self.shapes.remove(shape)
                        continue  # Skip drawing since it's erased
                elif shape_type == ShapeMode.Circle:
                    center = shape[0]
                    radius = int((abs(center.x() - shape[1].x()) + abs(center.y() - shape[1].y())) / 2)
                    distance = ((point.x() - center.x()) ** 2 + (point.y() - center.y()) ** 2) ** 0.5
                    if distance <= radius:
                        self.shapes.remove(shape)
                        continue  # Skip drawing since it's erased
                elif shape_type == ShapeMode.Heart:
                    if self.RemoveHeart(point, shape[0], shape[1]):
                        self.shapes.remove(shape)
                        continue  # Skip drawing since it's erased
            # Draw the shape
            if shape_type == ShapeMode.Square:
                qp.drawRect(QRect(shape[0], shape[1]))
            elif shape_type == ShapeMode.Circle:
                center = shape[0]
                radius = int((abs(center.x() - shape[1].x()) + abs(center.y() - shape[1].y())) / 2)
                qp.drawEllipse(center, radius, radius)
            elif shape_type == ShapeMode.Heart:
                WeaveView.drawHeart(self ,qp, shape[0], shape[1])

    

    def RemoveHeart(self, point, start, end):
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
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_mode:
            self.end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_mode:
            self.shapes.append([self.begin, self.end, SHAPE_MODE, SHAPE_COLOR])
            self.begin = event.pos()
            self.end = event.pos()
            self.update()

    def set_drawing_mode(self, enabled):
        self.drawing_mode = enabled
        self.update()




# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    cursor_button = None
    eraser_button = None
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

        # Create the Drawing widget (where users draw)
        self.drawing_widget = DrawingWidget()
        main_layout.addWidget(self.drawing_widget)

        # Create the output display widget
        #self.display_widget = QLabel(self)
        #pixmap = QPixmap("Weave Toolkit/icons/20201208_203402.jpg")
        #self.display_widget.setPixmap(pixmap)
        #self.display_widget.hide()  # Hide initially
        #main_layout.addWidget(self.display_widget)

        # Graphic Screen
        self.scene = QGraphicsScene()

        # Create the WeaveView and add to the layout
        self.weave_widget = WeaveView(self.scene)  # This is where the grid rendering happens
        self.weave_widget.hide()  # Hide initially, show it on button click
        main_layout.addWidget(self.weave_widget)

        # Create a stacked widget to switch between views
        self.stacked_widget = QStackedWidget(self)
        self.stacked_widget.addWidget(self.drawing_widget)
        self.stacked_widget.addWidget(self.weave_widget)
        main_layout.addWidget(self.stacked_widget)

        self.setFixedSize(QSize(1200, 700))

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
        file_menu.setStyleSheet("color: black;")
        action_new = QAction("New", self)
        action_open = QAction("Open", self)
        action_save = QAction("Save", self)
        action_export = QAction("Export", self)
        file_menu.addAction(action_new)
        file_menu.addAction(action_open)
        file_menu.addAction(action_save)
        file_menu.addAction(action_export)

        return file_menu

    def createViewDropdownMenu(self):
        view_menu = QMenu("View", self)
        view_menu.setStyleSheet("color: black;")
        action_zoom = QAction("Zoom", self)
        action_fullscreen = QAction("Fullscreen", self)
        view_menu.addAction(action_zoom)
        view_menu.addAction(action_fullscreen)

        return view_menu

    def updateDisplay(self):
        self.stacked_widget.setCurrentWidget(self.weave_widget)
        #self.display_widget.show()

    def editDisplay(self):
        self.stacked_widget.setCurrentWidget(self.drawing_widget)
        #self.display_widget.hide()

    def createShapesToolbar(self):
        shapes_toolbar = QToolBar("Shapes toolbar")

        # Cursor Button
        MainWindow.cursor_button = self.createShapeButton("icons/cursor.png", "Cursor button", "This is the cursor button", ShapeMode.Cursor)
        shapes_toolbar.addAction(MainWindow.cursor_button)

        shapes_toolbar.addSeparator()

        # Eraser Button
        MainWindow.eraser_button = self.createShapeButton("icons/eraser.png", "Eraser button", "This is the eraser button", ShapeMode.Eraser)
        shapes_toolbar.addAction(MainWindow.eraser_button)

        shapes_toolbar.addSeparator()

        # Square Button
        MainWindow.square_button = self.createShapeButton("icons/square.png", "Square button", "This is the square button", ShapeMode.Square)
        shapes_toolbar.addAction(MainWindow.square_button)

        shapes_toolbar.addSeparator()

        # Circle Button
        MainWindow.circle_button = self.createShapeButton("icons/circle.png", "Circle button", "This is the circle button", ShapeMode.Circle)
        shapes_toolbar.addAction(MainWindow.circle_button)

        # Heart Button
        MainWindow.heart_button = self.createShapeButton("icons/heart.png", "Heart button", "This is the heart button", ShapeMode.Heart)
        shapes_toolbar.addAction(MainWindow.heart_button)

        return shapes_toolbar

    def createShapeButton(self, icon_path, button_text, status_tip, shape_mode):
        shape_button = QAction(QIcon(icon_path), button_text, self)
        shape_button.setStatusTip(status_tip)
        shape_button.triggered.connect(lambda: self.setMode(shape_mode))
        return shape_button

    # When a shape button is clicked, it is then set to that drawing mode
    def setMode(self, shape_mode):
        if shape_mode == ShapeMode.Cursor:
            self.drawing_widget.set_drawing_mode(False)
        elif shape_mode == ShapeMode.Eraser:
            self.drawing_widget.begin = QPoint(-1, -1) # Reset the begin point so the most recent shape isn't erased
            self.drawing_widget.end = QPoint(-1, -1)
            self.drawing_widget.set_drawing_mode(True)
        else:
            self.drawing_widget.set_drawing_mode(True)

        global SHAPE_MODE
        SHAPE_MODE = shape_mode


    def createColorsToolbar(self):
        colors_toolbar = QToolBar("Colors toolbar")

        colors = [("Red", "red"), ("Green", "green"), ("Orange", "orange"), ("Blue", "blue")]
        for color_name, color_value in colors:
            button = QPushButton(color_name, self, styleSheet=f"background-color: {color_value}")
            button.clicked.connect(partial(self.change_color, color_value))
            colors_toolbar.addWidget(button)

        return colors_toolbar

    def change_color(self, color):
        global SHAPE_COLOR
        SHAPE_COLOR = QColor(color)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()