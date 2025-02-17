from PyQt6.QtWidgets import (
    QMessageBox
)

# Message when user draws out of bounds
def outOfBoundDrawingMessage():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle("Warning")
    msg.setText("This is a warning message.")
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()