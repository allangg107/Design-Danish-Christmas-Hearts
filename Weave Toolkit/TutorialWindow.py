from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QWidget, QApplication, QPushButton
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import QSize
from PatternType import PatternType
from GlobalVariables import(
    getUserOutputSVGFileName
)

class TutorialWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tutorial Window")
        self.guide_content = """
        <h1 style="color: red;">Tutorial on How to Use the Toolkit</h1>

        <div>
            <p>
            First, when you launch the program, you will see the interface where you have the drawable canvas on the left side. It will be set to 
            the Simple Pattern Type by default. You can choose the pattern type you want to use from the dropdown menu.
            The pattern types are Simple, Symmetrical, Asymmetrical, and Classic. Each pattern type has its own set of rules and 
            guidelines for creating patterns. In the following sections, we will go over the rules for each pattern type, and other relevant
            information about the toolkit. 
            </p>
        </div>

        <h2>Toolkit: How to Use</h2>

        <div>
            <h3>Drop Down Menu's</h3>
            <p>
            The first drop down is the File Menu, which has the following options:
            <ul>
                <li>New: This will create a new project.</li>
                <li>Open: This will open an existing project.</li>
                <li>Save: This will save the current project.</li>
                <li>Export SVG: This will export the current project as an SVG file.</li>
                <lie> Export Guide: This will open the weaving guide for the existing project.</li>
                <li>Tutorial: This will open the tutorial guide.</li>
                <lie> Undo: This will undo the last action in the drawing window.</li>
                <li>Exit: This will exit the program.</li>
            </ul>
            </p>

            <p>
            The next drop down menu is the Update SVG menu. This menu has one option, which is to update the SVG file.
            Updating the SVG file will allow you to see the changes you have made in the drawing window. Specifically, the changes
            you have made will appear on the right side of the window, where you can see the result of your design.
            </p>

            <p>
            The next drop down menu is the Edit button. This button allows the user to change the view of the right side of the window. 
            After pressing Update SVG, the right side will show the design as an output, pressing Edit here, allows for the user to change the 
            view of the right side of the window to the original backside design.
            </p>

            <p>
            The next drop down menu is the Weaving Pattern menu. This menu allows you to choose from four different weaving patterns:
            <ul>
                <li>Simple: This is the default pattern type for the most simple designs and weaving.</li>
                <li>Symmetrical: This pattern type is for when users wish to make a symmetrical design.</li>
                <li>Asymmetrical: This pattern type is for when users wish to make an asymmetrical design.</li>
                <li>Classic: This pattern type is for when users wish to make a classic design.</li>
            </ul>
            </p>

            <p>
            The last drop down is the Side Type menu. This menu allows you to choose from two different side types:
            <ul>   
                <li>Front Side: This is the default side type for when users want to see their design on one side of the final product.</li>
                <li>Back Side: This side type is for when users want to see their design on the both sides of the final product.</li>
            </ul>
            </p>
        </div>

        <h2>Drawing Tools</h2>
            <div>
            <h3>Free Draw</h3>  
            <p>
            The free draw tool allows you to draw directly on the canvas when you press down on your mouse and drag to draw and upon 
            release will stop drawing. This tool is only used in the Simple Pattern Type. 
            </p>

            <h3>Eraser Tool</h3>
            <p>
            The eraser tool allows you to erase parts of your drawing on the canvas. You can click on drawings to delete them. This includes
            entire free draw lines, squares, circles, and lines. It is important to note that this eraser tool does not erase by 
            holding down to erase and dragging over parts of drawings. This tool is used in all pattern types.
            </p>

            <h3>Line Tool</h3>
            <p>
            The line tool allows you to draw straight lines on the canvas. You can click and drag to draw a line, and release the mouse button 
            to stop drawing. This tool is used in all pattern types.
            </p>
            
            <h3>Square Tool</h3>
            <p>
            The square tool allows you to draw squares on the canvas. You can click and drag to draw a square, and release the mouse button
            to stop drawing. This tool is used in all pattern types.
            </p>

            <h3>Circle Tool</h3>
            <p>
            The circle tool allows you to draw circles on the canvas. You can click and drag to draw a circle, and release the mouse button
            to stop drawing. This tool is used in all pattern types.
            </p>

            <h3>Heart Tool</h3>
            <p>
            The heart tool allows you to draw hearts on the canvas. You can click and drag to draw a heart, and release the mouse button
            to stop drawing. This tool is used in all pattern types.
            </p>

            <h3>Semi-Circle Tool</h3>
            <p>
            The semi-circle tool allows you to draw semi-circles on the canvas. You can click and drag to draw a semi-circle, 
            and release the mouse button to stop drawing. This tool is used in all pattern types, but has special applications 
            for the classic pattern, which will be explained in the classic pattern section of this guide.
            </p>
            </div>

        <h2>Color and Stroke Width Selection</h2>
            <div>
            <p>
            There are options for selecting foreground and background colors, as well as stroke width. The foreground color is the color of the lines 
            you draw, while the background color is the color of the canvas. The stroke width is the thickness of the lines for all shape tools 
            you draw with. The default color of the foreground is black and the default color of the background is white. 
            You can change the foreground and background colors by clicking on the color boxes next to the foreground and background labels.
            The stroke width is set to 1 by default and can be changed to any value between 1 and 20 by using the Stroke Width slider. 
            </p>
            </div>

        <h2>Rules/Constraints for Using Each Pattern Type</h2>

            <div>
            <p>
            If any of the following constraints are violated, the program will not crash, but the outcome of the weave will not be as expected or it will be entirely incorrect and unweaveable.
            </p>
            <h3>Simple Pattern Type</h3>
            <p>
            The simple pattern type is the most basic pattern type. It allows you to draw any shape you want within the drawable canvas.
            This pattern type allows you to use all of the drawing tools. You may draw outside the bounds of the canvas as the back end
            will automatically crop the drawing to weave correctly. The only constraint for this pattern type is that all drawings must be closed.
            This means that all lines must connect to each other and not leave any gaps when using the line or the free draw tool. While
            it is possible to draw un-closed lines and drawings, it is not recommended because the desired outcome will not occur. This pattern
            type works by cutting out the shapes you have drawn and displaying them when the weave is completed. So if the shape is not closed,
            the shape will not be fully cut out and will not display correctly. This is not important for the other shape tools, 
            as they are all closed shapes so they will always cut out correctly. You can have multiple shapes that are not connected to each other,
            but the shapes themselves must be closed shapes.
            </p>
            
            <h3>Symmetrical Pattern Type</h3>
            <p>
            The symmetrical pattern type is a slightly more advanced pattern type. While this pattern type is selected, when you draw on either
            side of the line of symmetry (the line of symmetry is the dotted line that appears in the middle of the canvas), the other side will mirror
            the drawing you have made. For this case, you may only use the square, circle, and semi-circle tools, the free draw and line tools do not work in this case, and drawing with them will not work. 
            In this case, when you draw on the canvas, the drawing must always have one of the figures touching the line of
            symmetry. In this case, all drawings must be attached to each other. You are not allowed to have any gaps between drawings, they must all
            be attached to each other. This means there is one combined figure on the canvas. This figure has bounds. The topmost and bottommost points of the figure
            along the line of symmetry (on the y-axis) must be able to have a 45 degree line to the left (for topmost) and right (for bottommost) 
            without intersecting any other parts of the pattern. If this constraint is violated, then the intersected part(s) of the pattern will get cut off.
            </p>

            <h3>Asymmetrical Pattern Type</h3>
            <div>
            <p>
            The asymmetrical pattern type is the most advanced pattern type. While this pattern type is selected, all of the same constraints from
            the symmetric case apply. The line of symmetry is still present here, but it is not used for mirroring the drawing. There is no mirroring in this pattern type.
            </p>

            <h3>Classic Pattern Type</h3>
            <p>
            The classic pattern type is the most advanced pattern type. This pattern will contain very different rules than the other pattern types because it 
            does not function as a typical drawing canvas. It is not possible to use the free draw or line tools in this pattern type. Instead, you must use the square, circle, heart, and semi-circle 
            shape tools to create your designs. When this pattern type is selected, the canvas will show Classic Lines: 3, and you can choose between 1-9 classic lines using the Classic Lines slider. Note, this slider only appears when the classic pattern is selected in the Weaving Pattern drop down menu. This is the default when you choose the classic pattern.
            This will result in a total of 6 classic lines in the canvas. You will notice that the lines create a grid-like pattern on the canvas. We refer to the squares made by the grid as boxes for reference in the following.
            These classic lines also create intersection points, the points where the classic lines intersect. These are important because they are
            snap points for the shapes you draw. This means that when you draw a shape, it will snap to the intersection points of the classic lines.
            Now, we can explain how to draw in the grid structure. When you selected a shape, specifically the circle, square, or heart shape, you will notice the shape will snap to the intersection points of the classic lines.
            The only valid way to snap a shape is to have the shape inside a box. This means that the shape must be inside a box completely. We do not enforce a strict constraint here, so
            it is possible to have a shape snap outside a box, but this is not recommended as it will break the weave. The shape must be inside a box completely. You may not 
            snap a shape in any of the boxes along the border of the grid. You may not snap shapes that share a border with another shape in a box. But, you may 
            snap a shape in a box that shares an intersection point with another shape.
            </p>
            <p>
            However, the semi-circle tool is special in this case. The semi-circle tool is use to create larger hearts, typically designed for nesting.
            You do not need to snap semi-circle to be in just one box size like the other shapes. You snap a semi-circle to intersection points, along
            the classic lines. If the semi-circle does not snap both ends of the semi-circle to the intersection points, strictly along a classic line, then
            the semi-circle will not be valid. We do not enforce this constraint strcitly, but it is recommended to ensure the weave is correct. 
            You may snap semi-circles along any amount of boxes, this will result in a larger semi-circle. Creating a semi-circle that is larger than 2 box spaces will result in the 
            line above the semi-circle being removed. This is to ensure the weave is correct. This means, however, that the intersection points will change. 
            It may remove a line that is being used as a snap point for another shape. Doing so will result in an incorrect weave. You may not snap a semi-circle along
            the border of the grid. 
            </p>
            <p>
            There is a specific order of operations for using semi-circles and combining with other drawable shapes, including multiple smi-circles.
            If you wish to use semi-circles, you must start with the largst (spanning across the most boxes) semi-circle first, and preferably near the top of the canvas to ensure more space to draw below it.
            You create a heart with the semi-circle by snapping a semi-circle on one side, and then mirroring the semi-circle on the other side. This, combined 
            with the reuslting square below the semi-circle will produce a heart shape. You can draw more designs inside this new heart shape with more semi circle hearts, or other shapes from the shape tool.
            If you wish to draw a design inside the heart, you must ensure that the shape does not share a boarder with a semi-circle that has been snapped, along with the other rules of not sharing borders with any other shapes drawn.
            Using these constraints effectively will allow users to draw shapes inside of boxes, and create semi-circle hearts that allow for nesting.
            </p>
            </div>

        """

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True) # Sets the guide to readonly to prevent editing
        palette = self.text_edit.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("white"))
        palette.setColor(QPalette.ColorRole.Text, QColor("black"))
        self.text_edit.setPalette(palette)
        self.update_text_edit()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        self.resize(QSize(1200, 700))

    def update_text_edit(self):
        self.text_edit.setHtml(self.guide_content.format(getUserOutputSVGFileName()))