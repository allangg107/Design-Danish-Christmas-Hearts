import cv2 as cv

import numpy as np

import math

import tempfile
import svgwrite
import aspose.svg as svg

from svgpathtools import svg2paths, svg2paths2, wsvg, Path, Line, Arc

from PyQt6.QtSvg import QSvgRenderer, QSvgGenerator
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPainter, QTransform, QPixmap, QColor
from PyQt6.QtCore import QSize, QByteArray, QRectF

# Will be called when the user presses the "Update SVG" button

# The algorithm will be given a image, of SVG type, of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting), and user drawing
#   - this PNG that is returned is the final image once the algorithm has processesed the user drawing and weave pattern
# 2. Assembly instructions to guide the user in weaving the output image

# QGraphicScene for scaling the svg
# QSvgRenderer for turning into QPainter, then PNG image for final output
# svgwrite for full vector control and modifying svg elements

MARGIN = 31

def createSvgGenerator(input_svg, output_svg):
    # Create a QSvgRenderer to load the SVG file
    # QSvgRenderer may not support complex actions like animation or certain CSS styling
    renderer = QSvgRenderer(input_svg)

    # Set up the SVG generator to save the output as SVG
    generator = QSvgGenerator()
    generator.setFileName(output_svg)
    generator.setSize(renderer.defaultSize())
    generator.setViewBox(QRectF(0, 0, renderer.defaultSize().width(), renderer.defaultSize().height()))
    generator.setTitle("Rotated SVG")
    generator.setDescription("SVG with rotation applied using QPainter")
    return generator

def createSvgGeneratorNoRender(output_svg, target_size: QSize):
    generator = QSvgGenerator()
    generator.setFileName(output_svg)
    generator.setSize(target_size)
    generator.setViewBox(QRectF(0, 0, target_size.width(), target_size.height()))
    generator.setTitle("Resized SVG")
    generator.setDescription("SVG resized using QPainter")
    return generator

def saveSVG(svg_string, filename):
    with open(filename, "w") as f:
        f.write(svg_string)

def shiftSvg(input_svg: str, output_svg: str, shift_x: float, shift_y: float):
    """
    Shifts the drawings in an SVG file to the right by a given amount.

    Parameters:
    - input_svg (str): Path to the input SVG file.
    - output_svg (str): Path to save the shifted SVG file.
    - shift_x (float): Amount to shift to the right (in SVG coordinate units).
    """
    # Load SVG paths and attributes
    paths, attributes = svg2paths(input_svg)

    # Apply translation to shift paths
    shifted_paths = [path.translated(complex(shift_x, shift_y)) for path in paths]

    # Save the modified SVG
    wsvg(shifted_paths, attributes=attributes, filename=output_svg)

def upscaleImage(image, scale_factor):
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Use INTER_NEAREST to maintain sharp lines
    upscaled = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_NEAREST)

    return upscaled

def downscaleImage(image, scale_factor):
    height, width = image.shape[:2]
    new_width = int(width // scale_factor)
    new_height = int(height // scale_factor)

    # Use INTER_NEAREST to maintain sharp lines
    downscaled = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_NEAREST)

    return downscaled

def rotateSvgWithQPainter(input_svg, output_svg, angle_degrees, center_x=0, center_y=0):
    renderer = QSvgRenderer(input_svg)
    generator = createSvgGenerator(input_svg, output_svg)

    painter = QPainter(generator)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Apply rotation around the specified center
    painter.translate(center_x, center_y)
    painter.rotate(angle_degrees)
    painter.translate(-center_x, -center_y)

    # Render the original SVG onto the new one
    renderer.render(painter)

    # Finish painting
    painter.end()

def resizeSvg(input_svg, output_svg, target_size: int):
    renderer = QSvgRenderer(input_svg)

    # Get original size
    original_size = renderer.defaultSize()

    # Compute scale factors separately
    scale_x = target_size / original_size.width()
    scale_y = target_size / original_size.height()
    scale_factor = float(min(scale_x, scale_y))  # Ensure it's a float

    # Compute the new size
    new_size = QSize(int(original_size.width() * scale_factor), int(original_size.height() * scale_factor))

    # Create the SVG generator
    generator = createSvgGeneratorNoRender(output_svg, new_size)

    # Use QPainter to render the resized SVG
    painter = QPainter()
    painter.begin(generator)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Apply scaling (must use two floats, not QSize)
    painter.scale(scale_factor, scale_factor)

    # Render the SVG onto the new scaled canvas
    renderer.render(painter)

    # Finish painting
    painter.end()

    #paths, attributes = svg2paths(input_svg)
    ## Get bounding box (min_x, max_x, min_y, max_y)
    #min_x = min(path.bbox()[0] for path in paths)
    #max_x = max(path.bbox()[1] for path in paths)
    #min_y = min(path.bbox()[2] for path in paths)
    #max_y = max(path.bbox()[3] for path in paths)
#
    ## Calculate current width and height
    #current_width = max_x - min_x
    #current_height = max_y - min_y
#
    ## Compute scaling factor to maintain aspect ratio
    #scale_factor = min(target_size / current_width, target_size / current_height)
#
    ## Apply scaling
    #resized_paths = [path.scaled(scale_factor) for path in paths]
#
    ##wsvg(resized_paths, attributes=attributes, filename=output_svg)

def rotateImageQimage(image, angle=-90):
    transform = QTransform().rotate(angle)  # Rotate by the specified angle
    return image.transformed(transform)

def rotateImage(matrix, angle=-60):
    height, width = matrix.shape[:2]

    # Get the center of the image
    center = (width // 2, height // 2)

    # Compute the rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new bounding dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Adjust the rotation matrix to keep the image centered
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Rotate the image
    rotated_matrix = cv.warpAffine(matrix, rotation_matrix, (new_width, new_height), borderValue=(255, 255, 255))

    return rotated_matrix

def createFinalHeartDisplay(image):
    image = rotateImage(image, angle=-45)
    line_color = (0, 0, 0)  # Black color
    height, width, _ = image.shape # isolated_pattern's shape is a square
    square_size =   height // 2  # Ensuring a balanced square
    center = (height // 2, width // 2)

    # Create a blank mask
    mask = np.full_like(image, 255)

    # Define square corners
    half_size = square_size // 2
    points = np.array([
        [center[0] - half_size, center[1]],
        [center[0], center[1] - half_size],
        [center[0] + half_size, center[1]],
        [center[0], center[1] + half_size]
    ])

    # Draw the rotated square (diamond)
    cv.polylines(mask, [points], isClosed=True, color=line_color, thickness=3)

    # Draw a circle at the center of the square
    point1 = [center[0] - half_size, center[1]]
    point2 = [center[0], center[1] + half_size]

    square_width = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    radius = math.floor(square_width // 2)
    halfway_point_upper_left = [(center[0] - half_size + center[0]) // 2, (center[1] + center[1] - half_size) //2]
    halfway_point_upper_right = [(center[0] + center[0] + half_size) // 2, (center[1] - half_size + center[1]) //2]
    cv.ellipse(mask, halfway_point_upper_left, (radius, radius), 0, -225, -45, line_color, thickness=3)
    cv.ellipse(mask, halfway_point_upper_right, (radius, radius), 0, -135, 45, line_color, 3)

    rotated_mask = rotateImage(mask, -45)

    square_width_rounded = math.floor(square_width) - 15
    scaled_pattern = cv.resize(image, (square_width_rounded, square_width_rounded), interpolation=cv.INTER_LANCZOS4) # scaled to fit inside the square of the heart

    # Calculate coordinates to overlay scaled_pattern on the square portion of the heart
    x_center = (rotated_mask.shape[1] - square_width_rounded) // 2
    y_center = (rotated_mask.shape[0] - square_width_rounded) // 2

    # Overlay scaled_pattern onto the square portion of the heart
    rotated_mask[y_center:y_center + square_width_rounded, x_center:x_center + square_width_rounded] = scaled_pattern

    reverse_rotated_mask = rotateImage(rotated_mask, 45)

    return reverse_rotated_mask

def svg_to_cv_image(svg_filepath):
    # Create an SVG renderer
    svg_renderer = QSvgRenderer(svg_filepath)
    width = svg_renderer.defaultSize().width()
    height = svg_renderer.defaultSize().height()

    # Create an image with the specified size
    image = QImage(QSize(width, height), QImage.Format.Format_ARGB32)
    image.fill(0)  # Fill the image with transparent color

    # Create a painter to render the SVG onto the image
    painter = QPainter(image)
    svg_renderer.render(painter)
    painter.end()

    # Convert QImage to NumPy array
    buffer = image.bits().asstring(image.sizeInBytes())
    cv_image = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))

    # Convert from ARGB to BGR (OpenCV format)
    cv_image = cv.cvtColor(cv_image, cv.COLOR_BGRA2BGR)

    return cv_image

def savePixmapToCvImage(pixmap):
     # Convert QPixmap to QImage
    image = pixmap.toImage()

    # Convert QImage to raw data
    width = image.width()
    height = image.height()

    # Use sizeInBytes() instead of byteCount()
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())

    # Create a NumPy array from the raw data, treating it as 8-bit unsigned integers
    img_array = np.array(ptr).reshape((height, width, 4))  # 4 channels (RGBA)

    # Convert from RGBA to BGR (OpenCV format)
    cv_image = cv.cvtColor(img_array, cv.COLOR_BGRA2BGR)

    return cv_image

def saveSvgToMatrix(filepath):
    # Create an SVG renderer
    renderer = QSvgRenderer(filepath)

    # Set the size for the output pixmap (adjust as needed)
    size = QSize(600, 600)  # You can adjust this based on your SVG's dimensions

    # Create a QPixmap to render the SVG onto
    pixmap = QPixmap(size)
    pixmap.fill()  # Fill with transparent background

    # Render the SVG to the pixmap
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    # Convert QPixmap to numpy array (matrix)
    # Extract the pixel data from QPixmap
    image = pixmap.toImage()  # Convert QPixmap to QImage
    width, height = image.width(), image.height()

    # Create a numpy array from the pixel data
    matrix = np.array([[image.pixelColor(x, y).toRgb() for x in range(width)] for y in range(height)])

    return matrix

def saveSvgToPixmap(dwg, size=QSize(600, 600)):
    # Convert the svgwrite.Drawing to a string (SVG content)
    svg_string = dwg.tostring()

    # Convert the string to a QByteArray
    byte_array = QByteArray(svg_string.encode('utf-8'))

    # Create a QSvgRenderer from the QByteArray
    renderer = QSvgRenderer(byte_array)

    # Create a QPixmap to render the SVG onto
    pixmap = QPixmap(size)
    pixmap.fill()  # Fill with transparent background

    # Create a QPainter to paint the SVG onto the pixmap
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    return pixmap

def saveSvgFileAsPixmap(filepath, size=QSize(600, 600)):
    renderer = QSvgRenderer(filepath)

    # Create a QPixmap to render the SVG onto
    pixmap = QPixmap(size)
    pixmap.fill()  # Fill with transparent background

    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    return pixmap

# Using QSvgRenderer, save final svg as QImage
def saveSvgFileAsQimage(filepath):
    # Load the SVG
    renderer = QSvgRenderer(filepath)

    # Create an image with the SVG's default size
    size = renderer.defaultSize()
    image = QImage(size, QImage.Format.Format_ARGB32)
    image.fill(0)  # Transparent background

    # Render SVG onto the QImage
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()

    return image

def saveSvgFileAsQLabel(filepath):
    widget = QLabel()
    widget.setGeometry(50,200,500,500)
    renderer = QSvgRenderer(filepath)
    widget.resize(renderer.defaultSize())
    painter = QPainter(widget)
    painter.restore()
    renderer.render(painter)

    return painter

def mainAlgorithmSvg(img, function = 'create'):

    match function:
        case 'create':
            createHeartCutoutSimplestPattern(1200)

        case 'show':
            # We start with a filepath to an svg image. But, we want to give createFinalHeartDisplay a CV Image
            heartPixmap = saveSvgFileAsPixmap(img)
            heartCvImage = savePixmapToCvImage(heartPixmap)

            return createFinalHeartDisplay(heartCvImage)

        case _:
            return 'error'

def drawEmptyStencil(width, height, starting_y, margin_x=MARGIN, line_color='black', file_name="allan is a miracle.svg"):
    dwg = svgwrite.Drawing(file_name, size=(width,height+starting_y))
    # define the square size
    square_size = (height // 1.5) - margin_x
    margin_y = margin_x + starting_y

    # x_offset = size // 2 // 1.5

    # draw the left square
    left_top_line_start = (margin_x + square_size // 2, margin_y)
    left_top_line_end = (left_top_line_start[0] + square_size, left_top_line_start[1])
    left_bottom_line_start = (left_top_line_start[0], left_top_line_start[1] + square_size)
    left_bottom_line_end = (left_bottom_line_start[0] + square_size, left_bottom_line_start[1])

    dwg.add(dwg.line(start=(left_top_line_start), end=(left_top_line_end), stroke="red", stroke_width=3))
    dwg.add(dwg.line(start=(left_bottom_line_start), end=(left_bottom_line_end), stroke="red", stroke_width=3))

    # left_arc
    # draw the left arc
    radius_x = square_size / 2
    radius_y = square_size / 2
    arc_start = (left_bottom_line_start[0], left_bottom_line_start[1])
    arc_end = (left_top_line_start[0], left_top_line_start[1])
    left_arc_path = f"M {arc_start[0]},{arc_start[1]} A {radius_x},{radius_y} 0 0,1 {arc_end[0]},{arc_end[1]}"

    dwg.add(dwg.path(d=left_arc_path, stroke="purple", fill="none", stroke_width=3))

    # draw the right square
    right_top_line_start = left_top_line_end
    right_top_line_end = (right_top_line_start[0] + square_size, right_top_line_start[1])
    right_bottom_line_start = left_bottom_line_end
    right_bottom_line_end = (right_bottom_line_start[0] + square_size, right_bottom_line_start[1])

    dwg.add(dwg.line(start=(right_top_line_start), end=(right_top_line_end), stroke="blue", stroke_width=3))
    dwg.add(dwg.line(start=(right_bottom_line_start), end=(right_bottom_line_end), stroke="blue", stroke_width=3))

    # draw the right arc
    arc_start = (right_top_line_end[0], right_top_line_end[1])
    arc_end = (right_bottom_line_end[0], right_bottom_line_end[1])
    right_arc_path = f"M {arc_start[0]},{arc_start[1]} A {radius_x},{radius_y} 0 0,1 {arc_end[0]},{arc_end[1]}"

    dwg.add(dwg.path(d=right_arc_path, stroke="yellow", fill="none", stroke_width=3))

    # draw the inner line cuts
    dwg.add(dwg.line(start=((left_top_line_start[0], left_top_line_start[1] + margin_x)), end=((right_top_line_end[0], right_top_line_end[1] + margin_x)), stroke="brown", stroke_width=3))
    dwg.add(dwg.line(start=((left_bottom_line_start[0], left_bottom_line_start[1] - margin_x)), end=((right_bottom_line_end[0], right_bottom_line_end[1] - margin_x)), stroke="brown", stroke_width=3))

    dwg.save()

    return file_name

def combineStencils(first_half, second_half, filename='combined.svg'):
    # combine the 2 stencil halves, where the first half is the left side and the second is the right

    # Load paths and attributes of the first SVG file
    paths1, attributes1 = svg2paths(first_half)

    # Load paths and attributes of the second SVG file
    paths2, attributes2 = svg2paths(second_half)

    # Combine the paths and attributes
    combined_paths = paths1 + paths2
    combined_attributes = attributes1 + attributes2

    # Save the combined SVG
    wsvg(combined_paths, attributes=combined_attributes, filename=filename)

def getPattern(original_pattern):
    match original_pattern:
        case 'front':
            return 'svg_file.svg'

        case 'back':
            return 'svg_file_2.svg'

        case _:
            return 'error'

def combinePatterns(first_half, second_half):
    # combine the 2 pattern halves, where the first half is the left side and the second is the right
    pass

def overlayPatternOnStencil(pattern, stencil, size, stencil_number, pattern_type, margin=MARGIN):
    # overlaying a pattern on a stencil will involve:
    # 1. rotate clockwise 45 degrees
    rotateSvgWithQPainter(pattern, "rotated_pattern.svg", 45, 200, 200)

    # 2. scale it to fit the inner line cuts
    scaled_pattern = resizeSvg("rotated_pattern.svg", "scaled_pattern.svg", 100)

    # 3. shift it right and down
    # shiftSvg(scaled_pattern, stencil, size)

def svgToDrawing(input_svg, output_drawing):
    with open(input_svg, 'r') as file:
        svg_content = file.read()
    # Create a new svgwrite Drawing
    dwg = svgwrite.Drawing(output_drawing)
    dwg.save()
    return dwg

def determinePatternType():
    return "simple"

def createHeartCutoutSimplestPattern(size, line_start=0, sides='onesided', line_color='black', background_color='white'):
    if sides=='onesided':
        width = size
        height = size // 2
        empty_stencil_1 = drawEmptyStencil(width, height, 0, file_name="stencil1.svg")
        empty_stencil_2 = drawEmptyStencil(width, height, height, file_name="stencil2.svg")

        pattern_type = determinePatternType()

        # if pattern 1 == symetrical:
            # stencil_1_pattern = getSymetricalPattern(1)
            # stencil_2_pattern = getSymetricalPattern(2)
        # elif asymetrical:
            # stencil_1_pattern = getAsymtricalPattern(1)
            # stencil_2_pattern = getAsymtricalPattern(2)
        # else:
        stencil_1_pattern = getPattern("front")
        stencil_2_pattern = getPattern("back")

        overlayed_pattern_1 = overlayPatternOnStencil(stencil_1_pattern, empty_stencil_1, size, 1, pattern_type)
        # overlayed_pattern_2 = overlayPatternOnStencil(stencil_2_pattern, empty_stencil_2, size, 2, pattern_type)

        # combined_stencil = combineStencils(overlayed_pattern_1, overlayed_pattern_2)

        # return combined_stencil

    # do the same for the mirrored version
    if sides =='twosided':
        return None
