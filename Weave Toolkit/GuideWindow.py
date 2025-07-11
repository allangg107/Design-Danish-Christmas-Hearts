from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QWidget, QApplication, QPushButton, QTextBrowser, QScrollArea
from PyQt6.QtGui import QPalette, QColor, QMovie
from PyQt6.QtCore import QSize, Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PatternType import PatternType
from GlobalVariables import(
    getUserOutputSVGFileName
)
from PatternType import PatternType


class GuideWindow(QDialog):
    def __init__(self, pattern_type):
        super().__init__()
        self.setWindowTitle("Guide Window")
        self.text_list = self.saveTextInList(pattern_type)
        self.video_list = self.createVideoList(pattern_type)
        self.createLayout(self.text_list, self.video_list)

    def createVideoList(self, pattern_type):
        if pattern_type == PatternType.Simple:
            return ["guide_assets/simple_step_2.mp4","guide_assets/simple_step_4.mp4","guide_assets/simple_step_5.mp4","guide_assets/simple_step_7.mp4"]
        
        elif pattern_type == PatternType.Symmetric or pattern_type == PatternType.Asymmetric:
            return ["guide_assets/sym_step_2.mp4","guide_assets/sym_step_4.mp4","guide_assets/sym_step_5.mp4","guide_assets/sym_step_7.mp4"]
        
        elif pattern_type == PatternType.Classic:
            return ["guide_assets/classic_step_2.mp4","guide_assets/classic_step_4.mp4","guide_assets/classic_step_5.mp4","guide_assets/classic_step_7.mp4"]
        
    def createImageList(self, pattern_type):
        if pattern_type == PatternType.Simple:
            return ["guide_assets/simple_step_1.jpg","guide_assets/simple_step_3.jpg","guide_assets/simple_step_6.jpg"]
        
        elif pattern_type == PatternType.Symmetric or pattern_type == PatternType.Asymmetric:
            return ["guide_assets/sym_step_1.jpg","guide_assets/sym_step_3.jpg","guide_assets/sym_step_6.jpg"]
        
        elif pattern_type == PatternType.Classic:
            return ["guide_assets/classic_step_1.jpg","guide_assets/classic_step_3.jpg","guide_assets/classic_step_6.jpg"]

    def saveTextInList(self, pattern_type):
        image_list = self.createImageList(pattern_type)
        step0_text = f"""
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
        </div>"""

        step1_text = f"""
        <h2>Step 1</h2>
        <div>
            <p>
            Place the stencil with the circle, on the flat surface with the circle on the bottom left side of the stencil.
            Place the other stencil, the one with the cross, on the flat surface with the cross on the bottom right side of the stencil.
            The stencil with the circle is on the left hand side of the flat surface the stencil with the cross is on the right hand side of the flat surface.
            <br>
            </p>
            <img src="{image_list[0]}" width="300">
        </div>"""

        step2_text = """
        <h2>Step 2</h2>
        <div>
            <p>
            Before weaving, we must fold the stencils. The left stencil is the one with the circle and the right stencil is the one with the cross.
            Take the left stencil and fold it in half, such that the left half is on top of the right half. 
            Take the right stencil and fold it in half, such that the right half is on top of the left half.
            Follow the video below to see how the stencils should be folded.
            
        </div>"""

        step3_text = f"""
        <h2>Step 3</h2>
        <div>
            <p> 
            Before we start weaving, we must ensure that the stencils are folded properly and oriented in the correct starting position.
            This will ensure that the weaving process is smooth and easy to follow, and will always weave correctly if oriented properly.
            As per the image in this section, we want the stencil with the circle to have the circle in the top left corner and the
            stencil with the cross to have the cross in the top right corner.
            <br>
            </p>
            <img src="{image_list[1]}" width="300">
        </div>"""

        step4_text = """
        <h2>Step 4</h2>
        <div>
            <p>
            Before we weave, we want to show the correct orientation that will occur after the first weave is completed. We want to rotate the stencil with
            the circle 135 degrees counter-clockwise and the stencil with the cross 135 degrees clockwise. Now the stencils are in the correct starting position.
            </p>
        </div>"""

        step5_text = """
        <h2>Step 5</h2>
        <div>
            <p> 
            The very first step of the weaving process is the most important because it will affect how the weave will look in the end. There is a video that is playing
            to help process the following instructions below.
            Take the first strip of the stencil with the circle, this strip should have the circle on it and place it in one hand. Take the 
            first strip of the stencil with the cross, this strip should have the cross on it and place it in the other hand. We now want to take the 
            strip with the cross on it and put it through the strip with the circle on it. If done correctly, the circle should be on the outer side of the weave
            and the cross should be visible inside the circle.
        </div>
        """
        step6_text = f"""
        <div>
            <p>
             You will know you have done it correctly if it looks like the image below.
            <br>
            </p>
        <img src="{image_list[2]}" width="300">
        """
        step7_text = """
        
        <div>
            <p> 
            Now continue the weaving process for this strip, next the strip with the circle on it should be put through 
            the next strip on the stencil with the cross on it.
            This is typically described as over and under weaving. However, this weave is inside a strip and outside the next strip and is therefore called under over weaving.
            To continue the weave, finish each strip in order, maintaining the weave pattern. At the end of the weave, you should have the image shown below.
        </div>
        """
        step8_text = f"""
        
        <div>
            <p>
            Congratulations! You have completed weaving your customized Danish Christmas heart.
            </p>
        </div>
        """
        
        return [step0_text, step1_text,step2_text,step3_text,step4_text,step5_text,step6_text,step7_text,step8_text]
    
    def createLayout(self, text_lst, video_lst):
        guide_layout = QVBoxLayout()
        video_step_index = 0
        for i in range(len(text_lst)):
            if i in (0,1,3,6,8):
                # Adds text without video
                guide_layout.addWidget(TextWidget(text_lst[i]))
            
            else:
                # Adds text with video
                if video_step_index < len(video_lst):
                    #print(video_lst[i])
                    guide_layout.addWidget(StepWidget(text_lst[i],video_lst[video_step_index]))
                    video_step_index +=1
                else:
                    guide_layout.addWidget(TextWidget(text_lst[i]))
        
        container = QWidget()
        container.setLayout(guide_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)
        self.resize(QSize(1200, 700))

        #  This is Peters old thing -> going to need you to do these braces. Start the weaving by picking up the folded stencil closest to your dominant hand. Then take hold of the strip
        #         closest to your own body. Now pickup the other folded stencil with your other hand and grab hold of the strip closest
        #         to your own body again. Now take the strip in your dominant hand and weave it {} the strip in your non-dominant hand
        #         as seen on the image above You then continue the weaving with the same strip in your dominant hand and weave it {} the next strip in your non-dominant hand.
        #     <strong>You then continue doing step 3 and step 4 until the strip in your dominant hand has been woven all the way through.</strong>You then take the next strip from the stencil closest to your non-dominant hand and weave it {} the next strip in your non-dominant hand.
        #         You then weave it {} the next strip in your non-dominant hand. <strong>You then continue to do step 3 through 5 until the heart is fully woven.</strong></p>

# Used for creating displays text with a video when called
class StepWidget(QWidget):
    def __init__(self, html, video_file, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        text = QTextBrowser()
        text.setHtml(html)
        text.setReadOnly(True) # Sets the guide to readonly to prevent editing
        text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        max_width = 1000
        text.document().setTextWidth(max_width)
        doc_h = text.document().size().height()
        text.setFixedSize(max_width, int(doc_h))
        
        palette = text.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorRole.Text, QColor("black"))
        text.setPalette(palette)
        v.addWidget(text)
        
        video_widget = QVideoWidget()
        
        video_widget.setMinimumSize(300,300)
        
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(0)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(video_widget)
        self.player.setSource(QUrl.fromLocalFile(video_file))
        self.player.setLoops(-1)
        self.player.play()
        v.insertWidget(1,video_widget)

# Used for creating displays text without a video when called
class TextWidget(QWidget):
    def __init__(self, html, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        text = QTextBrowser()
        text.setHtml(html)
        text.setReadOnly(True) # Sets the guide to readonly to prevent editing
        text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        max_width = 1000
        text.document().setTextWidth(max_width)
        doc_h = text.document().size().height()
        text.setFixedSize(max_width, int(doc_h))

        palette = text.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorRole.Text, QColor("black"))
        text.setPalette(palette)
        v.addWidget(text)
        