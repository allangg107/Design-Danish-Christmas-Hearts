import sys

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
    QWidgetAction
)

from PyQt6.QtGui import (
    QColor,
    QAction,
    QIcon,
    QPainter,
    QPen,
    QBrush
)

# Global variable for the shape mode
SHAPE_MODE = ShapeMode.Cursor

class DrawingWidget(QWidget):
    # Defining the initial state
    def __init__(self):
        super().__init__()
        self.shapes = [] #Stores shapes
        self.setGeometry(30,30,600,400)
        self.begin = QPoint()
        self.end = QPoint()
        self.drawing_mode = False
        self.show()

    # Draws the current shape
    def paintEvent(self, event):
        qp = QPainter(self)
        br = QBrush(QColor(100, 10, 10, 40))
        qp.setBrush(br)
        for shape in self.shapes:
            if (shape[2] == ShapeMode.Square):
                qp.drawRect(QRect(shape[0], shape[1]))  # Draw the square
            elif (shape[2] == ShapeMode.Circle):
                center = shape[0]
                radius = int((abs(center.x() - shape[1].x()) + abs(center.y() - shape[1].y())) / 2)
                qp.drawEllipse(center, radius, radius)  # Draw the circle


        # Draw the current shape being created
        if self.begin != self.end:
            if (SHAPE_MODE == ShapeMode.Square):
                qp.drawRect(QRect(self.begin, self.end))
            elif (SHAPE_MODE == ShapeMode.Circle):
                center = self.begin
                radius = int((self.begin-self.end).manhattanLength() / 2)
                qp.drawEllipse(center, radius, radius)

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
            if (SHAPE_MODE == ShapeMode.Square):
                self.shapes.append([self.begin, self.end, ShapeMode.Square])
            elif (SHAPE_MODE == ShapeMode.Circle):
                self.shapes.append([self.begin, self.end, ShapeMode.Circle])
            self.begin = event.pos()
            self.end = event.pos()
            self.update()

    def set_drawing_mode(self, enabled):
        self.drawing_mode = enabled
        self.update()

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    cursor_button = None
    square_button = None
    circle_button = None
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Weave Toolkit")

        self.setStyleSheet("background-color: white;")

        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Create Menu toolbar and its actions
        menu_toolbar = QToolBar("Menu toolbar")

        file_button = QAction("File", self)
        file_button_menu = self.createFileDropdownMenu()
        file_button.setMenu(file_button_menu)
        menu_toolbar.addAction(file_button)

        view_button = QAction("View", self)
        view_button_menu = self.createViewDropdownMenu()
        view_button.setMenu(view_button_menu)
        menu_toolbar.addAction(view_button)

        main_layout.addWidget(menu_toolbar)

        # Create Shapes toolbar and its actions
        shapes_toolbar = QToolBar("Shapes toolbar")

        # Cursor Button
        MainWindow.cursor_button = self.createCursorButton()
        shapes_toolbar.addAction(MainWindow.cursor_button)

        shapes_toolbar.addSeparator()

        # Square Button
        MainWindow.square_button = self.createSquareButton()
        shapes_toolbar.addAction(MainWindow.square_button)

        shapes_toolbar.addSeparator()

        # Circle Button
        MainWindow.circle_button = self.createCircleButton()

        shapes_toolbar.addAction(MainWindow.circle_button)

        main_layout.addWidget(shapes_toolbar)

        # Create Colors toolbar
        colors_toolbar = QToolBar("Colors toolbar")

        colors_toolbar.addWidget(QPushButton("Red", self, styleSheet="background-color: red"))
        colors_toolbar.addWidget(QPushButton("Green", self, styleSheet="background-color: green"))
        # colors_toolbar.addWidget(QPushButton("Orange", self, styleSheet="background-color: orange"))
        # colors_toolbar.addWidget(QPushButton("Blue", self, styleSheet="background-color: blue"))

        main_layout.addWidget(colors_toolbar)

        # Create Drawing widget
        self.drawing_widget = DrawingWidget()
        main_layout.addWidget(self.drawing_widget)
        self.setFixedSize(QSize(1200, 700))

    def createShapeButton(self, icon_path, button_text, status_tip, shape_mode):
        shape_button = QAction(QIcon(icon_path), button_text, self)
        shape_button.setStatusTip(status_tip)
        #shape_button.setCheckable(True)
        shape_button.triggered.connect(lambda: self.setMode(shape_mode))
        return shape_button

    def createCursorButton(self):
        square_button = self.createShapeButton("icons/cursor.png", "Cursor button", "This is the cursor button", ShapeMode.Cursor)
        return square_button

    def createSquareButton(self):
        square_button = self.createShapeButton("icons/square.png", "Square button", "This is the square button", ShapeMode.Square)
        return square_button

    def createCircleButton(self):
        circle_button = self.createShapeButton("icons/circle.png", "Circle button", "This is the circle button", ShapeMode.Circle)
        return circle_button

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

    # Checks whether a shape button is clicked, if it is then that drawing mode is enabled
    def setMode(self, shape_mode):
        if shape_mode == ShapeMode.Cursor:
            self.drawing_widget.set_drawing_mode(False)
        else:
            self.drawing_widget.set_drawing_mode(True)
        global SHAPE_MODE
        SHAPE_MODE = shape_mode


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()