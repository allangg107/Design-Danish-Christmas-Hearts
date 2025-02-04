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
    QToolBar
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

        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)


        # Square Button
        button_action = QAction(QIcon("icons/square.png"), "Square button", self)
        button_action.setStatusTip("This is the square button")
        #button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action.setCheckable(True)
        button_action.toggled.connect(self.isToggled)
        toolbar.addAction(button_action)

        toolbar.addSeparator()

        # Circle Button
        button_action1 = QAction(QIcon("icons/circle.png"), "Circle button", self)
        button_action1.setStatusTip("This is the circle button")
        #button_action.triggered.connect(self.onMyToolBarButtonClick)
        button_action1.setCheckable(True)
        button_action1.toggled.connect(self.isToggled)
        toolbar.addAction(button_action1)

        # string currentMode = cursorMode; # cursorMode, rectMode, circMode, lineMode, etc.
        # in each method, just check       if (currentMode != exampleMode) currentMode = exampleMode; else currentMode = cursorMode;

        #main_layout = QHBoxLayout()
        #main_layout = QVBoxLayout()

        #main_layout.addStretch()

        #layout = QHBoxLayout()
        #layout.addWidget(QPushButton("Red", self, styleSheet="background-color: red"))
        #layout.addWidget(QPushButton("Green", self, styleSheet="background-color: green"))
        #layout.addWidget(QPushButton("Orange", self, styleSheet="background-color: orange"))
        #layout.addWidget(QPushButton("Blue", self, styleSheet="background-color: blue"))

        #main_layout.addLayout(layout)


        widget = QWidget()
        #widget.setLayout(main_layout)
        self.drawing_widget = DrawingWidget()
        self.setCentralWidget(self.drawing_widget)
        self.setFixedSize(QSize(1200, 700))

    def isToggled(self, checked):
        self.drawing_widget.set_drawing_mode(checked)

    #def onMyToolBarButtonClick(self, s):
     #   self.setCentralWidget(self.drawing_widget)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()