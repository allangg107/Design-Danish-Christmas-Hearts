import cv2 as cv

import numpy as np

import math

import tempfile
import svgwrite

from svgwrite.container import Group
import xml.etree.ElementTree as ET


from svgpathtools import svg2paths, svg2paths2, wsvg, Path, Line, Arc

from PyQt6.QtSvg import QSvgRenderer, QSvgGenerator
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPainter, QTransform, QPixmap, QColor
from PyQt6.QtCore import QSize, QByteArray, QRectF
from collections import namedtuple
# Will be called when the user presses the "Update SVG" button

# The algorithm will be given a image, of SVG type, of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting), and user drawing
#   - this PNG that is returned is the final image once the algorithm has processesed the user drawing and weave pattern
# 2. Assembly instructions to guide the user in weaving the output image

# QGraphicScene for scaling the svg
# QSvgRenderer for turning into QPainter, then PNG image for final output
# svgwrite for full vector control and modifying svg elements

MARGIN = 31

def removeOutOfBoundsDrawing(img):
    new_image= rotateImage(img, angle=45)

    # Removes the canvas lines and non-draw zones from the image
    matrix = new_image[180:-180,180:-180]
    return matrix

def createSvgGenerator(input_svg, output_svg, width = None, height = None):
    # Create a QSvgRenderer to load the SVG file
    # QSvgRenderer may not support complex actions like animation or certain CSS styling
    renderer = QSvgRenderer(input_svg)

    # if width or height are not specified, then the input_svg's size is used
    if width == None or height == None:
        bounds = renderer.viewBoxF()
        width = int(bounds.width())
        height = int(bounds.height())

    bounds = QRectF(0, 0, width, height)
    generator = QSvgGenerator()
    generator.setFileName(output_svg)
    generator.setSize(QSize(width, height))  # Ensure the size matches the viewBox
    generator.setViewBox(bounds)  # Match the original viewBox exactly

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

def rotateSvgFileByAngle(paths, angle, size):
    cx, cy = size.width() / 2, size.height() / 2  # Center of canvas
    angle_rad = math.radians(angle)

    def rotate_point(x, y):
        """ Rotates a point (x, y) around (cx, cy) by angle_rad """
        dx, dy = x - cx, y - cy
        new_x = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        new_y = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        return new_x, new_y

    rotated_paths = []
    for path in paths:
        new_segments = []
        for segment in path:
            if isinstance(segment, Line):
                start_x, start_y = rotate_point(segment.start.real, segment.start.imag)
                end_x, end_y = rotate_point(segment.end.real, segment.end.imag)
                new_segments.append(Line(start=complex(start_x, start_y), end=complex(end_x, end_y)))
            else:
                # Handle other types of segments if necessary (CubicBezier, QuadraticBezier, Arc)
                new_segments.append(segment)

        rotated_paths.append(Path(*new_segments))

    return rotated_paths

# Define a simple Line structure
Line = namedtuple("Line", ["start", "end"])
Path = list  # A path is a list of Line objects
def has_start_attribute(segment):
    """Checks if a segment has the 'start' attribute."""
    return hasattr(segment, 'start')

def rotate_point_complex(point, angle, center):
    """Rotates a complex point around a center by a given angle in degrees."""
    angle_rad = math.radians(angle)
    rotated = (point - center) * complex(math.cos(angle_rad), math.sin(angle_rad)) + center
    return rotated

def rotate_paths(paths, angle, size):
    """Rotates a list of SVG paths around the center of a canvas with given size."""
    center = (size.width() / 2, size.height() / 2)
    rotated_paths = []

    for path in paths:
        rotated_segments = []
        # Iterate over each segment in the path
        for segment in path:
            if has_start_attribute(segment):  # Only process Line segments
                # Rotate the start and end points
                new_start = rotate_point_complex(segment.start, angle, center)
                new_end = rotate_point_complex(segment.end, angle, center)
                rotated_segments.append(Line(new_start, new_end))
            else:
                # Handle non-Line segments (optional, depending on your needs)
                rotated_segments.append(segment)

        # Wrap the list of rotated segments in a Path object
        rotated_paths.append(Path(*rotated_segments))

    return rotated_paths

def rotateSvgWithQPainter(input_svg, output_svg, angle_degrees, center_x=0, center_y=0, attributes=[]):
    renderer = QSvgRenderer(input_svg)
    generator = createSvgGenerator(input_svg, output_svg)

    painter = QPainter(generator)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # print(generator.size())

    # Apply rotation around the specified center
    painter.translate(center_x, center_y)
    painter.rotate(angle_degrees)
    painter.translate(-center_x, -center_y)

    # Render the original SVG onto the new one
    renderer.render(painter)

    # Finish painting
    painter.end()
    return output_svg

def resizeSvg(input_svg, output_svg, target_size: int):
    renderer = QSvgRenderer(input_svg)

    # Get original size
    original_size = renderer.defaultSize()

    # Compute scale factors separately
    scale_x = target_size / original_size.width()
    scale_y = target_size / original_size.height()
    scale_factor = float(min(scale_x, scale_y))  # Ensure it's a float

    # Makes the old size a QSize object
    old_size = QSize(int(original_size.width()), int(original_size.height()))
    # Compute the new size
    new_size = QSize(int(original_size.width() * scale_x), int(original_size.height() * scale_x))

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

def resizeSVGNoGen(input_svg, output_svg, target_size: int):
    paths, atrributes = svg2paths(input_svg)
    return None

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
    # image = rotateImage(image, angle=-45)
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

def mainAlgorithmSvg(img, function = 'create', shape_attributes=[]):

    match function:
        case 'create':
            createFinalHeartCutoutPatternExport(1200, attributes=shape_attributes)

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

    # draw the left square
    left_top_line_start = (margin_x + square_size // 2, margin_y)
    left_top_line_end = (left_top_line_start[0] + square_size, left_top_line_start[1])
    left_bottom_line_start = (left_top_line_start[0], left_top_line_start[1] + square_size)
    left_bottom_line_end = (left_bottom_line_start[0] + square_size, left_bottom_line_start[1])

    dwg.add(dwg.line(start=(left_top_line_start), end=(left_top_line_end), stroke="red", stroke_width=3))
    dwg.add(dwg.line(start=(left_bottom_line_start), end=(left_bottom_line_end), stroke="red", stroke_width=3))

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

def combineStencils(first_stencil, second_stencil, filename='combined.svg'):
    # Load paths and attributes of the first SVG file
    paths1, attributes1 = svg2paths(first_stencil)

    # Load paths and attributes of the second SVG file
    paths2, attributes2 = svg2paths(second_stencil)

    # Combine the paths and attributes
    combined_paths = paths1 + paths2
    combined_attributes = attributes1 + attributes2

    # Save the combined SVG
    wsvg(combined_paths, attributes=combined_attributes, filename=filename)

def getPattern(original_pattern):
    match original_pattern:
        case 'front':
            return 'final_output_svg.svg'

        case 'back':
            return 'svg_file_2.svg'

        case _:
            return 'error'

def overlayDrawingOnStencil(stencil_file, user_drawing_file, size, filename='combined_output.svg'):
        # Load paths from the stencil
        paths1, attributes1 = svg2paths(stencil_file)

        # Load paths from the user’s drawing
        paths2, attributes2 = svg2paths(user_drawing_file)

        # Combine paths from both the stencil and the user’s drawing
        combined_paths = paths1 + paths2
        combined_attributes = attributes1 + attributes2

        # Create a new SVG drawing to store the combined result
        dwg = svgwrite.Drawing(filename, size=(size, size))

        # Add each path from the combined paths to the new SVG
        for path, attr in zip(combined_paths, combined_attributes):
            # Extract stroke, fill, and stroke-width attributes (if they exist)
            stroke = attr.get('stroke', 'black')  # Default to black if no stroke is defined
            fill = attr.get('fill', 'none')      # Default to 'none' if no fill is defined
            stroke_width = attr.get('stroke-width', 1)  # Default to 1 if no stroke width is defined


            # Add the path with its attributes to the new SVG
            dwg.add(dwg.path(d=path.d(), stroke=stroke, fill=fill, stroke_width=stroke_width))

        # for path in combined_paths:
          #  dwg.add(dwg.path(d=path.d(), stroke="black", fill="none"))

        # Save the combined SVG to the file
        dwg.save()
        return filename

def overlayPatternOnStencil(pattern, stencil, size, stencil_number, pattern_type, margin=MARGIN, shape_attributes=[{}]):
    # 1. Rotate and scale the pattern
    paths, attributes = svg2paths(pattern)
    print("paths now: ", len(paths))
    print("allans nuts: ", attributes)
    #resizeSvg(pattern, "resized.svg", size // 3 - margin * 3)
    #rotate that bitch
    paths, _ = svg2paths(pattern)
    print("paths nlater: ", len(paths))

    # Debug: Print the number of paths
    print("Number of paths:", len(paths))


    # Ensure shape_attributes list has the same length as paths
    # if len(shape_attributes) < len(paths):
    #     # Instead of replicating the first element, extend with default attribute dictionaries.
    #     missing = len(paths) - len(shape_attributes)
    #     shape_attributes.extend([{} for _ in range(missing)])
    # elif len(shape_attributes) > len(paths):
    #     shape_attributes = shape_attributes[:len(paths)]

    # Debug: Print the final attributes that will be passed
    print("Final shape attributes:", shape_attributes)

    # Save the SVG using wsvg with the overwritten attributes
    wsvg(paths, attributes=shape_attributes, filename="newfile.svg", dimensions=(size, size))


    # 3. Shift it right and down (overlay on stencil)
    combined_output = overlayDrawingOnStencil(stencil, "newfile.svg", size, "overlayed_test.svg")
    return combined_output


def svgToDrawing(input_svg, output_drawing):
    with open(input_svg, 'r') as file:
        svg_content = file.read()
    # Create a new svgwrite Drawing
    dwg = svgwrite.Drawing(output_drawing)
    dwg.save()
    return dwg

def determinePatternType():
    return "simple"

def createFinalHeartCutoutPatternExport(size, line_start=0, sides='onesided', line_color='black', background_color='white', attributes=[]):
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

        overlayed_pattern_1 = overlayPatternOnStencil(stencil_1_pattern, empty_stencil_1, size, 1, pattern_type, shape_attributes=attributes)
        # overlayed_pattern_2 = overlayPatternOnStencil(stencil_2_pattern, empty_stencil_2, size, 2, pattern_type)

        # combined_stencil = combineStencils(overlayed_pattern_1, overlayed_pattern_2)

        # resizeSvg(combined_stencil, user_decided_export_size)

        # return combined_stencil

    # do the same for the mirrored version
    if sides =='twosided':
        return None
