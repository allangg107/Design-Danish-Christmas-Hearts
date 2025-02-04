import sys

from PyQt6.QtCore import (
    QSize,
    Qt,
    QRectF,
    QRect,
    QPoint
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

class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.squares = [] #Stores squares
        self.setGeometry(30,30,600,400)
        self.begin = QPoint()
        self.end = QPoint()
        self.drawing_mode = False
        self.show()

    def paintEvent(self, event):
        qp = QPainter(self)
        for square in self.squares:
            br = QBrush(QColor(100, 10, 10, 40))  # Set color and transparency
            qp.setBrush(br)
            qp.drawRect(QRect(square[0], square[1]))  # Draw the square

        # Draw the current square being created
        if self.begin != self.end:
            br = QBrush(QColor(100, 10, 10, 40))
            qp.setBrush(br)
            qp.drawRect(QRect(self.begin, self.end))

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
            self.squares.append((self.begin, self.end))
            self.begin = event.pos()
            self.end = event.pos()
            self.update()

    def set_drawing_mode(self, enabled):
        self.drawing_mode = enabled
        self.update()



# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
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

        # Square Button
        square_button = QAction(QIcon("icons/square.png"), "Square button", self)
        square_button.setStatusTip("This is the square button")
        #button_action.triggered.connect(self.onMyToolBarButtonClick)
        square_button.setCheckable(True)
        square_button.toggled.connect(self.isToggled)
        shapes_toolbar.addAction(square_button)

        shapes_toolbar.addSeparator()

        # Circle Button
        circle_button = QAction(QIcon("icons/circle.png"), "Circle button", self)
        circle_button.setStatusTip("This is the circle button")
        #button_action.triggered.connect(self.onMyToolBarButtonClick)
        circle_button.setCheckable(True)
        circle_button.toggled.connect(self.isToggled)
        shapes_toolbar.addAction(circle_button)

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
        

    def isToggled(self, checked):
        self.drawing_widget.set_drawing_mode(checked)

    #def onMyToolBarButtonClick(self, s):
     #   self.setCentralWidget(self.drawing_widget)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()