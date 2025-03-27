import sys
from PyQt6.QtSvg import QSvgRenderer, QSvgGenerator
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPainter, QTransform, QPixmap, QColor
from PyQt6.QtCore import QSize, QByteArray, QRectF
import xml.etree.ElementTree as ET
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier, QuadraticBezier, parse_path
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon
import math
import numpy as np
import cv2 as cv

FILE_STEP_COUNTER = 1

"""Preprocessing"""
def pre_process_user_input(original_pattern, shape_types, width, height, square_size):
    global FILE_STEP_COUNTER
    global DRAWING_SQUARE_SIZE
    DRAWING_SQUARE_SIZE = square_size

    rotated_path_name = f"{FILE_STEP_COUNTER}_rotated_pattern_step.svg"
    FILE_STEP_COUNTER += 1
    rotateSVG(original_pattern, rotated_path_name, 45)

    # crop to the designated drawing space
    cropped_size = int((width - square_size) // 2)
    translated_path_name = f"{FILE_STEP_COUNTER}_translated_pattern_step.svg"
    FILE_STEP_COUNTER += 1
    translateSVGBy(rotated_path_name, translated_path_name, -cropped_size, -cropped_size)

    paths, attributes = svg2paths(translated_path_name)

    final_output_path_name = "preprocessed_pattern.svg"
    wsvg(paths, attributes=attributes, filename=final_output_path_name, dimensions=(square_size, square_size))

    # print("pre-processed attributes: ", attributes)

    # print(f"Original path ({len(paths)} segments):", paths)
    clipped_paths = crop_svg(paths, square_size, square_size)
    # print(f"Original path ({len(paths)} segments):", paths)

    print(f"Number of paths after translation: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"Path {i} has {len(path)} segments")

    clipped_paths = crop_svg(paths, square_size, square_size)

    # Print the number of clipped paths and segments
    print(f"Number of clipped paths: {len(clipped_paths)}")
    for i, path in enumerate(clipped_paths):
        print(f"Clipped path {i} has {len(path)} segments")

    if len(clipped_paths) > len(attributes):
        attributes = attributes * len(clipped_paths)

    wsvg(clipped_paths, attributes=attributes, filename=final_output_path_name, dimensions=(square_size, square_size))

"""Utility functions for rotating an SVG file"""

def rotateImage(matrix, angle=-45):
    height, width = matrix.shape[:2]

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

def rotateSVG(input_svg, output_svg, angle, center_x=None, center_y=None):
    paths, attributes = svg2paths(input_svg)

    tree = ET.parse(input_svg)
    root = tree.getroot()
    width = float(root.get("width", "500"))  # Default to 500 if missing
    height = float(root.get("height", "500"))

    # Calculate the center of the SVG
    if center_x is None or center_y is None:
        center_x = width / 2
        center_y = height / 2
    center = complex(center_x, center_y)  # Convert to complex for easier rotation

    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Rotation function
    def rotate_point(point):
        return (point - center) * complex(math.cos(angle_rad), math.sin(angle_rad)) + center

    # Rotate all paths
    for i in range(len(paths)):
        new_segments = []
        for segment in paths[i]:
            if isinstance(segment, Line):
                new_segments.append(Line(rotate_point(segment.start), rotate_point(segment.end)))
            elif isinstance(segment, CubicBezier):
                new_segments.append(CubicBezier(
                    rotate_point(segment.start),
                    rotate_point(segment.control1),
                    rotate_point(segment.control2),
                    rotate_point(segment.end)
                ))
            elif isinstance(segment, QuadraticBezier):
                new_segments.append(QuadraticBezier(
                    rotate_point(segment.start),
                    rotate_point(segment.control),
                    rotate_point(segment.end)
                ))
            elif isinstance(segment, CubicBezier):
                new_segments.append(CubicBezier(
                    rotate_point(segment.start),
                    rotate_point(segment.control1),
                    rotate_point(segment.control2),
                    rotate_point(segment.end)
                ))
            elif isinstance(segment, QuadraticBezier):
                new_segments.append(QuadraticBezier(
                    rotate_point(segment.start),
                    rotate_point(segment.control),
                    rotate_point(segment.end)
                ))
            else:
                print("rotateSVG: unsupported line detected")
        paths[i] = Path(*new_segments)

    wsvg(paths,
         attributes=attributes,
         filename=output_svg,
         dimensions=(center_x*2, center_y*2))

"""
Utility functions for resizing an SVG file
"""
def resizeSVG(input_svg, output_svg, target_width):
    paths, attributes = svg2paths(input_svg)

    tree = ET.parse(input_svg)
    root = tree.getroot()
    original_width = float(root.get("width", "500"))
    original_height = float(root.get("height", "500"))

    # Compute scale factor based on width
    scale_factor = target_width / original_width
    target_height = original_height * scale_factor  # Maintain aspect ratio

    # Function to scale a point (complex number)
    def scale_point(point):
        return point * scale_factor

    # Scale each segment in each path
    for i in range(len(paths)):
        new_segments = []
        for segment in paths[i]:
            if isinstance(segment, Line):
                new_segments.append(
                    Line(scale_point(segment.start), scale_point(segment.end))
                )
            elif isinstance(segment, CubicBezier):
                new_segments.append(
                    CubicBezier(
                        scale_point(segment.start),
                        scale_point(segment.control1),
                        scale_point(segment.control2),
                        scale_point(segment.end)
                    )
                )
            elif isinstance(segment, QuadraticBezier):
                new_segments.append(
                    QuadraticBezier(
                        scale_point(segment.start),
                        scale_point(segment.control),
                        scale_point(segment.end)
                    )
                )
            else:
                new_segments.append(segment)
        paths[i] = Path(*new_segments)

    wsvg(paths,
         attributes=attributes,
         filename=output_svg,
         dimensions=(int(target_width), int(target_height)),
         svg_attributes={'viewBox': f'0 0 {int(target_width)} {int(target_height)}'})


"""Moves SVG files by a specific values"""
    
def translateSVGBy(input_svg, output_svg, x_shift, y_shift):
    """
    Translates an SVG file by a set amount depending on the values of the x_shift and y_shift
    """
    paths, attributes = svg2paths(input_svg)

    translation = complex(x_shift, y_shift)

    tree = ET.parse(input_svg)
    root = tree.getroot()
    width = float(root.get("width", "500"))  # Default to 500 if missing
    height = float(root.get("height", "500"))

    # Apply translation to all paths
    for i in range(len(paths)):
        new_segments = []
        for segment in paths[i]:
            if isinstance(segment, Line):
                new_segments.append(Line(segment.start + translation, segment.end + translation))
            elif isinstance(segment, CubicBezier):
                new_segments.append(CubicBezier(
                    segment.start + translation,
                    segment.control1 + translation,
                    segment.control2 + translation,
                    segment.end + translation
                ))
            elif isinstance(segment, QuadraticBezier):
                new_segments.append(QuadraticBezier(
                    segment.start + translation,
                    segment.control + translation,
                    segment.end + translation
                ))
            else:
                print("translateSVG: unsupported line detected")
        paths[i] = Path(*new_segments)

    wsvg(paths,
         attributes=attributes,
         filename=output_svg,
         dimensions=(height, width))
    
def translateSVGTo(input_svg, output_svg, target_x, target_y):
    """
    Translates an SVG to a specific target position (target_x, target_y).
    The top-left corner of the SVG will be positioned at these coordinates.
    """
    paths, attributes = svg2paths(input_svg)

    # Get the current position of the top-left corner
    min_x = float('inf')
    min_y = float('inf')
    
    for path in paths:
        for segment in path:
            # Sample points to find the minimum x and y coordinates
            for t in np.linspace(0, 1, 10):
                pt = segment.point(t)
                min_x = min(min_x, pt.real)
                min_y = min(min_y, pt.imag)
    
    # Calculate the translation needed to move to target position
    x_shift = target_x - min_x
    y_shift = target_y - min_y
    
    # Apply the translation
    translation = complex(x_shift, y_shift)

    tree = ET.parse(input_svg)
    root = tree.getroot()
    width = float(root.get("width", "500"))  # Default to 500 if missing
    height = float(root.get("height", "500"))

    # Apply translation to all paths
    for i in range(len(paths)):
        new_segments = []
        for segment in paths[i]:
            if isinstance(segment, Line):
                new_segments.append(Line(segment.start + translation, segment.end + translation))
            elif isinstance(segment, CubicBezier):
                new_segments.append(CubicBezier(
                    segment.start + translation,
                    segment.control1 + translation,
                    segment.control2 + translation,
                    segment.end + translation
                ))
            elif isinstance(segment, QuadraticBezier):
                new_segments.append(QuadraticBezier(
                    segment.start + translation,
                    segment.control + translation,
                    segment.end + translation
                ))
            else:
                print("translateSVGTo: unsupported segment type detected")
        paths[i] = Path(*new_segments)

    wsvg(paths,
         attributes=attributes,
         filename=output_svg,
         dimensions=(width, height))


"""Mirrors SVG files over different axises"""

def mirrorSVGOverYAxis(input_svg, output_svg, width, height):
    paths, attributes = svg2paths(input_svg)

    # Mirror each path over the Y-axis by negating the x coordinates
    mirrored_paths = []
    for path in paths:
        mirrored_segments = []
        for segment in path:
            if isinstance(segment, Line):
                mirrored_segments.append(
                    Line(
                        complex(width - segment.start.real, segment.start.imag),
                        complex(width - segment.end.real, segment.end.imag)
                    )
                )
            elif isinstance(segment, CubicBezier):
                mirrored_segments.append(
                    CubicBezier(
                        complex(width - segment.start.real, segment.start.imag),
                        complex(width - segment.control1.real, segment.control1.imag),
                        complex(width - segment.control2.real, segment.control2.imag),
                        complex(width - segment.end.real, segment.end.imag)
                    )
                )
            elif isinstance(segment, QuadraticBezier):
                mirrored_segments.append(
                    QuadraticBezier(
                        complex(width - segment.start.real, segment.start.imag),
                        complex(width - segment.control.real, segment.control.imag),
                        complex(width - segment.end.real, segment.end.imag)
                    )
                )
        mirrored_paths.append(Path(*mirrored_segments))

    # Write the mirrored paths to the output file
    wsvg(mirrored_paths, attributes=attributes, filename=output_svg, dimensions=(width, height))

def mirrorSVGOverYAxisWithX(input_svg, output_svg, width, height, x_mirror):
    paths, attributes = svg2paths(input_svg)

    # Mirror each path over the Y-axis by negating the x coordinates
    mirrored_paths = []
    for path in paths:
        mirrored_segments = []
        for segment in path:
            if isinstance(segment, Line):
                mirrored_segments.append(
                    Line(
                        complex(2*x_mirror - segment.start.real, segment.start.imag),
                        complex(2*x_mirror - segment.end.real, segment.end.imag)
                    )
                )
            elif isinstance(segment, CubicBezier):
                mirrored_segments.append(
                    CubicBezier(
                        complex(2*x_mirror - segment.start.real, segment.start.imag),
                        complex(2*x_mirror - segment.control1.real, segment.control1.imag),
                        complex(2*x_mirror - segment.control2.real, segment.control2.imag),
                        complex(2*x_mirror - segment.end.real, segment.end.imag)
                    )
                )
            elif isinstance(segment, QuadraticBezier):
                mirrored_segments.append(
                    QuadraticBezier(
                        complex(2*x_mirror - segment.start.real, segment.start.imag),
                        complex(2*x_mirror - segment.control.real, segment.control.imag),
                        complex(2*x_mirror - segment.end.real, segment.end.imag)
                    )
                )
        mirrored_paths.append(Path(*mirrored_segments))

    # Write the mirrored paths to the output file
    wsvg(mirrored_paths, attributes=attributes, filename=output_svg, dimensions=(width, height))

def mirrorSVGOverXAxis(input_svg, output_svg, width, height):
    paths, attributes = svg2paths(input_svg)
    
    # Mirror each path over the X-axis by negating the y-coordinates
    mirrored_paths = []
    for path in paths:
        mirrored_segments = []
        for segment in path:
            if isinstance(segment, Line):
                mirrored_segments.append(
                    Line(
                        complex(segment.start.real, height - segment.start.imag),
                        complex(segment.end.real, height - segment.end.imag)
                    )
                )
            elif isinstance(segment, CubicBezier):
                mirrored_segments.append(
                    CubicBezier(
                        complex(segment.start.real, height - segment.start.imag),
                        complex(segment.control1.real, height - segment.control1.imag),
                        complex(segment.control2.real, height - segment.control2.imag),
                        complex(segment.end.real, height - segment.end.imag)
                    )
                )
            elif isinstance(segment, QuadraticBezier):
                mirrored_segments.append(
                    QuadraticBezier(
                        complex(segment.start.real, height - segment.start.imag),
                        complex(segment.control.real, height - segment.control.imag),
                        complex(segment.end.real, height - segment.end.imag)
                    )
                )
        mirrored_paths.append(Path(*mirrored_segments))
    
    # Write the mirrored paths to the output file
    wsvg(mirrored_paths, attributes=attributes, filename=output_svg, dimensions=(width, height))

def mirrorSVGOverXAxisWithY(input_svg, output_svg, width, height, y_mirror):
    paths, attributes = svg2paths(input_svg)
    
    # Mirror each path over the X-axis by negating the y-coordinates
    mirrored_paths = []
    for path in paths:
        mirrored_segments = []
        for segment in path:
            if isinstance(segment, Line):
                mirrored_segments.append(
                    Line(
                        complex(segment.start.real, 2 * y_mirror - segment.start.imag),
                        complex(segment.end.real, 2 * y_mirror - segment.end.imag)
                    )
                )
            elif isinstance(segment, CubicBezier):
                mirrored_segments.append(
                    CubicBezier(
                        complex(segment.start.real, 2 * y_mirror - segment.start.imag),
                        complex(segment.control1.real, 2 * y_mirror - segment.control1.imag),
                        complex(segment.control2.real, 2 * y_mirror - segment.control2.imag),
                        complex(segment.end.real, 2 * y_mirror - segment.end.imag)
                    )
                )
            elif isinstance(segment, QuadraticBezier):
                mirrored_segments.append(
                    QuadraticBezier(
                        complex(segment.start.real, 2 * y_mirror - segment.start.imag),
                        complex(segment.control.real, 2 * y_mirror - segment.control.imag),
                        complex(segment.end.real, 2 * y_mirror - segment.end.imag)
                    )
                )
        mirrored_paths.append(Path(*mirrored_segments))
    
    # Write the mirrored paths to the output file
    wsvg(mirrored_paths, attributes=attributes, filename=output_svg, dimensions=(width, height))


def removeDuplicateLinesFromSVG(svg_with_pattern, svg_without_pattern, output_filename=None):
    """
    Extracts paths that exist in svg_with_pattern but not in svg_without_pattern.
    
    Args:
        svg_with_pattern: Path to SVG file containing pattern and stencil
        svg_without_pattern: Path to SVG file containing only the stencil
        output_filename: Optional filename for output SVG, defaults to auto-generated name
        
    Returns:
        Path to the output SVG file containing only the pattern elements
    """
    if output_filename is None:
        output_filename = f"{getFileStepCounter()}_pattern_only.svg"
        incrementFileStepCounter()
    
    # Extract paths from both SVGs
    with_paths, with_attrs = svg2paths(svg_with_pattern)
    without_paths, without_attrs = svg2paths(svg_without_pattern)
    
    # Convert paths to strings for comparison
    without_path_strings = [path.d() for path in without_paths]
    
    # Find paths that are in with_paths but not in without_paths
    pattern_paths = []
    pattern_attrs = []
    
    for i, path in enumerate(with_paths):
        path_str = path.d()
        if path_str not in without_path_strings:
            pattern_paths.append(path)
            pattern_attrs.append(with_attrs[i])
    
    # Save the pattern-only paths to a new SVG
    wsvg(pattern_paths, attributes=pattern_attrs, filename=output_filename)
    
    return output_filename


"""Cropping of SVG files"""

def clip_path_to_boundary(path, boundary, width, height, num_samples_line=1, num_samples_bezier=20):
    """
    Clips a given path to the boundary using Shapely geometric operations.
    Samples each segment to better approximate curves, then closes the resulting
    path by connecting the entry and exit points along the boundary, including any
    missing corner points.
    """
    try:
        # Sample points along each segment.
        sampled_coords = []
        for seg in path:
            if isinstance(seg, CubicBezier):
                num_samples = num_samples_bezier
            else:
                num_samples = num_samples_line

            for t in np.linspace(0, 1, num_samples, endpoint=False):
                pt = seg.point(t)
                sampled_coords.append((pt.real, pt.imag))
        # Ensure the final point is included.
        last_pt = path[-1].point(1.0)
        sampled_coords.append((last_pt.real, last_pt.imag))

        if len(sampled_coords) < 2:
            print("Skipping path: Not enough coordinates.")
            return None

        path_shape = LineString(sampled_coords)
        clipped_shape = path_shape.intersection(boundary)

        if clipped_shape.is_empty:
            print("Warning: Clipped shape is empty!")
            return None

        # Process a single LineString result.
        if isinstance(clipped_shape, LineString):
            coords = list(clipped_shape.coords)
            new_path = Path(
                *[Line(complex(x, y), complex(x2, y2))
                  for (x, y), (x2, y2) in zip(coords[:-1], coords[1:])]
            )
            return new_path

        # Process MultiLineString by merging segments.
        elif isinstance(clipped_shape, MultiLineString):
            all_coords = []
            for line in clipped_shape.geoms:
                line_coords = list(line.coords)
                if not all_coords:
                    all_coords.extend(line_coords)
                else:
                    # If the end of the last segment isn't the start of the next,
                    # insert a connecting segment.
                    if (abs(all_coords[-1][0] - line_coords[0][0]) > 1e-6 or
                        abs(all_coords[-1][1] - line_coords[0][1]) > 1e-6):
                        all_coords.append(line_coords[0])
                    all_coords.extend(line_coords)
            new_path = Path(
                *[Line(complex(x, y), complex(x2, y2))
                  for (x, y), (x2, y2) in zip(all_coords[:-1], all_coords[1:])]
            )
            return new_path

        print("Warning: Unexpected geometry type from intersection:", type(clipped_shape))
        return None

    except Exception as e:
        print("Error while clipping path:", e)
        return None
    
def crop_svg(paths, width, height):
    """
    Crops all paths to fit within the given square_size.
    """
    boundary = Polygon([(0, 0), (width, 0), (width,height), (0, height)])

    #print("\nBoundary Polygon:", boundary)
    #print("Total Paths Received for Clipping:", len(paths))

    clipped_paths = []
    for path in paths:
        clipped = clip_path_to_boundary(path, boundary, width, height)
        if clipped:
            if isinstance(clipped, list):  # Handle MultiLineString cases
                clipped_paths.extend(clipped)
            else:
                clipped_paths.append(clipped)

    #print("Final Clipped Paths:", clipped_paths)

    return clipped_paths

def cropPrep(pattern, output_name, cropped_size, angle):
    global FILE_STEP_COUNTER

    # Step 1: Rotate the pattern 45 degrees clockwise
    rotated_path_name = f"{FILE_STEP_COUNTER}_rotated_pattern_step.svg"
    FILE_STEP_COUNTER += 1
    rotateSVG(pattern, rotated_path_name, angle)

    # Step 2: Translate to correct position after rotation
    translateSVGBy(rotated_path_name, output_name, cropped_size, cropped_size)


def cropToTopHalf(input_svg, output_svg):
    # Crop to only the top half of the SVG
        paths, attributes = svg2paths(input_svg)
        stencil_1_clipped_paths = crop_svg(paths, 500, 500 // 2)

        # Adjust attributes if needed
        if len(stencil_1_clipped_paths) > len(attributes):
            attributes = attributes * len(stencil_1_clipped_paths)

        # Save the cropped half
        wsvg(stencil_1_clipped_paths, attributes=attributes, filename=output_svg,
                dimensions=(DRAWING_SQUARE_SIZE, DRAWING_SQUARE_SIZE))


"""Conversion of SVG files into different formats"""

def convertSvgToPng(svg_file, width, height, output_file):
    cvImage = savePixmapToCvImage(saveSvgFileAsPixmap(svg_file, QSize(height, width)))

    transparentImage = makeTrans(cvImage, [255, 255, 255])
    cv.imwrite(output_file, transparentImage)


def makeTrans(final_output_array, color):
    # Create an empty alpha channel (fully opaque)
    alpha_channel = np.ones((final_output_array.shape[0], final_output_array.shape[1]), dtype=np.uint8) * 255  # Fully opaque
    # Find where the matrix is white (background)
    colored_pixels = np.all(final_output_array == color, axis=-1)
    # Set alpha channel to 0 (fully transparent) where the pixels are white
    alpha_channel[colored_pixels] = 0
    # Create an RGBA image by adding the alpha channel to the original matrix
    rgba_image = np.dstack((final_output_array, alpha_channel))
    return rgba_image


def saveSvgFileAsPixmap(filepath, size=QSize(600, 600)):
    renderer = QSvgRenderer(filepath)

    pixmap = QPixmap(size)
    pixmap.fill()  # Fill with transparent background

    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    return pixmap


def savePixmapToCvImage(pixmap):
    image = pixmap.toImage()

    width = image.width()
    height = image.height()

    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())

    # Create a NumPy array from the raw data, treating it as 8-bit unsigned integers
    img_array = np.array(ptr).reshape((height, width, 4))  # 4 channels (RGBA)

    # Convert from RGBA to BGR (OpenCV format)
    cv_image = cv.cvtColor(img_array, cv.COLOR_BGRA2BGR)

    return cv_image

"""Functions to get the global values"""

def getFileStepCounter():
    return FILE_STEP_COUNTER

def getDrawingSquareSize():
    return DRAWING_SQUARE_SIZE

"""Functions to set global values"""

def incrementFileStepCounter():
    global FILE_STEP_COUNTER
    FILE_STEP_COUNTER += 1

def setDrawingSquareSize(value):
    global DRAWING_SQUARE_SIZE
    DRAWING_SQUARE_SIZE = value
