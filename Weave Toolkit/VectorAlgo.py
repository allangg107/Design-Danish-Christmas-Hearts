import cv2 as cv

import numpy as np

import svgwrite

import math


from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPainter, QTransform, QPixmap
from PyQt6.QtCore import QSize, QByteArray

# Will be called when the user presses the "Update SVG" button

# The algorithm will be given a image, of SVG type, of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting), and user drawing
#   - this PNG that is returned is the final image once the algorithm has processesed the user drawing and weave pattern
# 2. Assembly instructions to guide the user in weaving the output image

# QGraphicScene for scaling the svg
# QSvgRenderer for turning into QPainter, then PNG image for final output
# svgwrite for full vector control and modifying svg elements

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

# def createFinalHeartDisplay(filepath):
#     # Load existing SVG
#     dwg = svgwrite.Drawing(size=("500px", "500px"))
#     #dwg.add(dwg.image(filepath, insert=(0, 0), size=("600px", "600px")))
#     # Define center and size for transformations
#     width, height = 500, 500
#     center = (width // 2, height // 2)
#     square_size = height // 2
#     half_size = square_size // 2
#     # Create square
#     square_points = [
#         (center[0] - half_size, center[1] - half_size),  # Top-left
#         (center[0] + half_size, center[1] - half_size),  # Top-right
#         (center[0] + half_size, center[1] + half_size),  # Bottom-right
#         (center[0] - half_size, center[1] + half_size)   # Bottom-left
#     ]

#     # Add the square
#     dwg.add(dwg.polygon(points=square_points, fill='none', stroke='black', stroke_width=3))

#     # Draw heart-like curves (two ellipses)
#     radius = math.floor(square_size / 2)
#     halfway_left = (center[0] - half_size // 2, center[1] - half_size // 2)
#     halfway_right = (center[0] + half_size // 2, center[1] - half_size // 2)

#     # Left semi-circle (half of a circle, clockwise direction)
#     dwg.add(dwg.path(d="M {},{} A {},{} 0 0,1 {},{}"
#                      .format(center[0] - half_size - radius, center[1],
#                              radius, radius,
#                              center[0] - half_size + radius, center[1]),
#                      fill='none', stroke='black', stroke_width=3))

#     # Right semi-circle (half of a circle, counter-clockwise direction)
#     dwg.add(dwg.path(d="M {},{} A {},{} 0 0,1 {},{}"
#                      .format(center[0] + half_size - radius, center[1],
#                              radius, radius,
#                              center[0] + half_size + radius, center[1]),
#                      fill='none', stroke='black', stroke_width=3))

#     # Scale and center the SVG pattern inside the diamond
#     dwg.add(dwg.image(filepath, insert=(center[0] - half_size, center[1] - half_size),
#                       size=(f"{square_size}px", f"{square_size}px"),
#                       transform=f"rotate(-45,{center[0]},{center[1]})"))
#     finalDwg = saveSvgToPixmap(dwg)

#     return finalDwg

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

    case 'show':
        # We start with a filepath to an svg image. But, we want to give createFinalHeartDisplay a CV Image
        heartPixmap = saveSvgFileAsPixmap(img)
        heartCvImage = savePixmapToCvImage(heartPixmap)

        return createFinalHeartDisplay(heartCvImage)

    case _:
      return 'error'

def baseStencilSvg(width, height, margin=31, line_start=0, line_color='black'):
  dwg = svgwrite.Drawing(size=(width,height))
  line_y1 = margin*4
  line_y2 = height-margin*4
  line_y3 = margin
  line_y4 = height-margin

  dwg.add(dwg.line(start=(line_start, line_y1), end=(width, line_y1), stroke=line_color, stroke_width=3))
  dwg.add(dwg.line(start=(line_start, line_y2), end=(width, line_y2), stroke=line_color, stroke_width=3))
  dwg.add(dwg.line(start=(line_start, line_y3), end=(width, line_y3), stroke=line_color, stroke_width=3))
  dwg.add(dwg.line(start=(line_start, line_y4), end=(width, line_y4), stroke=line_color, stroke_width=3))

  center_x = line_start
  center_y = (line_y3 + line_y4) // 2
  radius_x = 330
  radius_y = (line_y4 - line_y3)  // 2
  path_d = f"M {center_x}, {line_y3} A {radius_x},{radius_y} 0 1, 0 {center_x}, {line_y4}"
  dwg.add(dwg.path(d=path_d, stroke=line_color, fill='none', stroke_width=3))
  return dwg

def createHeartCutoutSimplestPattern(width, height, line_start=0, sides='oneisided', line_color='black', background_color='white'):
    dwg = baseStencilSvg(width, height, 31, line_start, line_color)

    if sides=='twosided':
       mirrored_dwg = baseStencilSvg(width,height, 31, line_start, background_color)
       return None
    return None
def saveSVG(svg_string, filename):
    with open(filename, "w") as f:
        f.write(svg_string)