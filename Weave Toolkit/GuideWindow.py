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

        <h2>Pre-Weaving/Cricut Design Guide</h2>
        <div>
            <p>
            You will have two svg files once you have exported your project and, you may decide which color paper you wish to have for each file. 
            The best results occur when using two different colors for each stencil. You need to upload each svg file directly to Cricut Design Space, 
            and choose your preferred dimensions for the stencil size. It is important to ensure that for each stencil file, you should maintain the same
            size stencil for both files. It is also important that there is a limit on the design size, you cannot make a stencil go out
            of the bounds of the Cricut Design Board, if you do, it will not cut the stencils properly. You must also ensure that
            you left click on your stencil, once it is laoded into Cricut Design Space, and select Attach. This will ensure that your
            stencil prints out exactly as it is loaded in on the program. You also have the freedom to choose
            your own paper types, and choose blade length in accordance to the paper type of your choosing to ensure a proper cut.
            After you have cut out the stencils using the Cricut machine and Cricut Design Space, you will have two stencils,
            one with a circle and one with a cross. In Step 1, we will go over how to orient the stencils properly to begin weaving.
            </p>
        </div>

        <h2>Step 1 </h2>
        <div>
            <img src="step2.svg" width="300">
            <br>
            <p>
            Place the stencil with the circle, on the flat surface with the circle on the bottom left side of the stencil.
            Place the other stencil, the one with the cross, on the flat surface with the cross on the bottom right side of the stencil.
            The stencil with the circle is on the left hand side of the flat surface the stencil with the cross is on the right hand side of the flat surface.
            </p>
        </div>
    
        <h2>Step 2</h2>
        <div>
            <img src="step2.svg" width="300">
            <br>
            <p>
            Before weaving, we must fold the stencils. The left stencil is the one with the circle and the right stencil is the one with the cross.
            Take the left stencil and fold it in half, such that the left half is on top of the right half. 
            Take the right stencil and fold it in half, such that the right half is on top of the left half.
            Follow the images below to see how the stencils should be folded.
            </p>
        </div>

        <h2>Step 3</h2>
        <div>
            <img src="step3.svg" width="300">
            <br>
            <p> 
            Before we start weaving, we must ensure that the stencils are folded properly and oriented in the correct starting position.
            This will ensure that the weaving process is smooth and easy to follow, and will always weave correctly if oriented properly.
            As per the image in this section, we want the stencil with the circle to have the circle in the top right corner and the
            stencil with the cross to have the cross in the top left corner.
            </p>
        </div>

        <h2>Step 4</h2>
        <div>
            <img src="step4.svg" width="300">
            <br>
            <p>
            Before we weave, we want to show the correct orientation that will occur after the first weave is completed. We want to rotate the stencil with
            the circle 90 degrees counter-clockwise and the stencil with the cross 90 degrees clockwise. Now the stencils are in the correct starting position.
            </p>
        </div>

        <h2>Step 5</h2>
        <div>
            <img src="step5.svg" width="300">
            <br>
            <p> 
            The very first step of the weaving process is the most important because it will affect how the weave will look in the end. There is a gif that is playing
            to help process the following instructions below.
            Take the first strip of the stencil with the circle, this strip should have the circle on it and place it in one hand. Take the 
            first strip of the stencil with the cross, this strip should have the cross on it and place it in the other hand. We now want to take the 
            strip with the cross on it and put it through the strip with the circle on it. If done correctly, the circle should be on the outer side of the weave
            and the cross should be visible inside the circle. You will know you have done it correctly if it looks like the image below. 
            Now continue the weaving process for this strip, next the strip with the circle should be put inside the other stencil strip, in that direction.
            This is typically described as over and under. However, this weave is inside a strip and outside a strip. (Explain this better)
            To continue the weave, finish each strip in order, maintaining the weave pattern. At the end of the weave, you should have the image shown below.
            Congratulations! You have completed weaving your customized Danish Christmas heart. 



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
        self.text_edit.setHtml(self.guide_content.format(getUserOutputSVGFileName()))


    def update_step(self, step_text1, step_text2):
        """Changes the text of the third and fourth step and updates the display."""
        self.third_step_text = step_text1
        self.fourth_step_text = step_text2

        self.update_text_edit()

        #  This is Peters old thing -> going to need you to do these braces. Start the weaving by picking up the folded stencil closest to your dominant hand. Then take hold of the strip
        #         closest to your own body. Now pickup the other folded stencil with your other hand and grab hold of the strip closest
        #         to your own body again. Now take the strip in your dominant hand and weave it {} the strip in your non-dominant hand
        #         as seen on the image above You then continue the weaving with the same strip in your dominant hand and weave it {} the next strip in your non-dominant hand.
        #     <strong>You then continue doing step 3 and step 4 until the strip in your dominant hand has been woven all the way through.</strong>You then take the next strip from the stencil closest to your non-dominant hand and weave it {} the next strip in your non-dominant hand.
        #         You then weave it {} the next strip in your non-dominant hand. <strong>You then continue to do step 3 through 5 until the heart is fully woven.</strong></p>
