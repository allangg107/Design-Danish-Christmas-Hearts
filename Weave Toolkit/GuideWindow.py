from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QSize



class GuideWindow(QDialog):
    def __init__(self, pattern_type):
        super().__init__()
        self.setWindowTitle("Guide Window")

        layout = QVBoxLayout()

        # Add an image to the layout


        # Add some text to the layout
        text_label = QLabel(f"This is some text displayed in the popup window.\nPattern Type: {pattern_type.name}")
        layout.addWidget(text_label)

        self.setLayout(layout)

        self.setFixedSize(QSize(1200, 700))