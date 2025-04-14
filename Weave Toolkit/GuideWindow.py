from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QWidget, QApplication, QPushButton
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import QSize
from PatternType import PatternType
from GlobalVariables import(
    setHasImage,
    getHasImage,
    getUserOutputSVGFileName
)


class GuideWindow(QDialog):
    def __init__(self, pattern_type):
        super().__init__()
        self.setWindowTitle("Guide Window")
        self.guide_content = """
        <h1 style="color: red;">Danish Christmas Hearts Weaving Guide</h1>

        <h2>Step 1 </h2>
        <div>
            <img src={} width="300">
            <br>
            <p>Place your 2 stencils down side by side in front of each hand, such that the stencil with the circle with a cross within
                is in front of your left hand and the stencil with just a cross within is placed in front of your right hand and
                the crosses from each stencil overlap, such that the circle with the cross is not visible inside the heart.
                As seen in the image above.</p>
        </div>
    
        <h2>Step 2</h2>
        <div>
            <img src="step2.svg" width="300">
            <br>
            <p> Now take the left stencil and its right half and fold it inwards, such that the right half is
                on top of the left half and the cicle with a cross is inside the fold. Now take the right stencil and its left half and fold it inwards, such that the left half
                is on top of the right half and the cross is inside the fold as seen in the image above. You are now ready to start the weaving.</p>
        </div>

        <h2>Step 3</h2>
        <div>
            <img src="step3.svg" width="300">
            <br>
            <p> Start the weaving by picking up the folded stencil closest to your dominant hand. Then take hold of the strip
                closest to your own body. Now pickup the other folded stencil with your other hand and grab hold of the strip closest
                to your own body again. Now take the strip in your dominant hand and weave it {} the strip in your non-dominant hand
                as seen on the image above</p>
        </div>

        <h2>Step 4</h2>
        <div>
            <img src="step4.svg" width="300">
            <br>
            <p> You then continue the weaving with the same strip in your dominant hand and weave it {} the next strip in your non-dominant hand.
            <strong>You then continue doing step 3 and step 4 until the strip in your dominant hand has been woven all the way through.</strong></p>
        </div>

        <h2>Step 5</h2>
        <div>
            <img src="step5.svg" width="300">
            <br>
            <p> You then take the next strip from the stencil closest to your non-dominant hand and weave it {} the next strip in your non-dominant hand.
                You then weave it {} the next strip in your non-dominant hand. <strong>You then continue to do step 3 through 5 until the heart is fully woven.</strong></p>
        </div>
        """
        self.third_step_text = ""
        self.fourth_step_text = ""
        # QTextEdit for displaying the instructions
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True) # Sets the guide to readonly to prevent editing
        palette = self.text_edit.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorRole.Text, QColor("black"))
        self.text_edit.setPalette(palette)
        #self.text_edit.setStyleSheet("background.color: white; color: black;") 
        self.update_text_edit() # Load content
        

        ## Add some text to the layout
        ##text_label = QLabel(f"This is some text displayed in the popup window.\nPattern Type: {pattern_type.name}")
        ##layout.addWidget(text_label)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

        ##self.setFixedSize(QSize(1200, 700))
        self.resize(QSize(1200, 700))
        
        # Checks for pattern type and if an image was drawn to make the guide
        #  fit the users need on the first step of the weaving
        if pattern_type == PatternType.Classic and not(getHasImage()):
            self.update_step("over", "in between")
        
        else:
            self.update_step( "in between", "over")
    def update_text_edit(self):
        """Updates the text edit with the current third step text."""
        self.text_edit.setHtml(self.guide_content.format(getUserOutputSVGFileName(),self.third_step_text, self.fourth_step_text, self.fourth_step_text, self.third_step_text))


    def update_step(self, step_text1, step_text2):
        """Changes the text of the third and fourth step and updates the display."""
        self.third_step_text = step_text1
        self.fourth_step_text = step_text2

        self.update_text_edit()