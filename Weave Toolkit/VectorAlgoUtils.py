import sys
from PyQt6.QtSvg import QSvgRenderer, QSvgGenerator
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPainter, QTransform, QPixmap, QColor, QPen, QBrush, QPolygonF
from PyQt6.QtCore import QSize, QByteArray, QRectF, Qt, QPointF, QRect
import xml.etree.ElementTree as ET
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier, QuadraticBezier, parse_path
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon
import math
import numpy as np
import cv2 as cv
import svgwrite

from GlobalVariables import (
    getFileStepCounter,
    getDrawingSquareSize,
    incrementFileStepCounter,
    setDrawingSquareSize,
    getMargin,
    getLineThicknessAndExtension,
    getBackgroundColor,
    getShapeColor,
    getCurrentPatternType,
    getNumClassicLines,
    getPenWidth,
    getClassicIndicesLineDeleteList,
    setClassicIndicesLineDeleteList,
    getClassicPatternSnapPoints,
    getClassicPatternClassicLines
)

from SideType import (
    SideType
)

from PatternType import (
    PatternType
)

"""Preprocessing"""
def pre_process_user_input(original_pattern, shape_types, width, height, square_size):
    setDrawingSquareSize(square_size)
    if original_pattern is None:
        # create a blank SVG file with the specified width and height
        original_pattern = "preprocessed_pattern.svg"
        dwg = svgwrite.Drawing(original_pattern, profile='tiny', size=(square_size, square_size))
        dwg.save()
        return

    rotated_path_name = f"{getFileStepCounter()}_rotated_pattern_step.svg"
    incrementFileStepCounter()
    rotateSVG(original_pattern, rotated_path_name, 45)

    # crop to the designated drawing space
    cropped_size = int((width - square_size) // 2)
    translated_path_name = f"{getFileStepCounter()}_translated_pattern_step.svg"
    incrementFileStepCounter()
    translateSVGBy(rotated_path_name, translated_path_name, -cropped_size, -cropped_size)

    paths, attributes = svg2paths(translated_path_name)

    final_output_path_name = "preprocessed_pattern.svg"
    wsvg(paths, attributes=attributes, filename=final_output_path_name, dimensions=(square_size, square_size))

    # print("pre-processed attributes: ", attributes)

    # print(f"Original path ({len(paths)} segments):", paths)
    clipped_paths = crop_svg(paths, 0, 0, square_size, square_size)
    clipped_paths = crop_svg(paths, 0, 0, square_size, square_size)
    # print(f"Original path ({len(paths)} segments):", paths)

    print(f"Number of paths after translation: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"Path {i} has {len(path)} segments")

    clipped_paths = crop_svg(paths, 0, 0, square_size, square_size)
    clipped_paths = crop_svg(paths, 0, 0, square_size, square_size)

    # Print the number of clipped paths and segments
    print(f"Number of clipped paths: {len(clipped_paths)}")
    for i, path in enumerate(clipped_paths):
        print(f"Clipped path {i} has {len(path)} segments")

    if len(clipped_paths) > len(attributes):
        attributes = attributes * len(clipped_paths)

    wsvg(clipped_paths, attributes=attributes, filename=final_output_path_name, dimensions=(square_size, square_size))

    print("finished pre-processing")

"""Utility functions for rotating an SVG file"""

def combineStencils(first_stencil, second_stencil, filename='combined.svg'):
    """Combines two SVG files together into one"""
    # Initialize empty paths and attributes
    paths1, attributes1 = [], []
    paths2, attributes2 = [], []

    # Try to read the first stencil
    try:
        paths1, attributes1 = svg2paths(first_stencil)
    except Exception as e:
        print(f"Could not read {first_stencil}: {e}")

    # Try to read the second stencil
    try:
        paths2, attributes2 = svg2paths(second_stencil)
    except Exception as e:
        print(f"Could not read {second_stencil}: {e}")

    # Check if both files are empty or non-existent
    if not paths1 and not paths2:
        print("Both SVG files are empty or non-existent.")
        return

    # If only one file has content, use that one
    if not paths1:
        wsvg(paths2, attributes=attributes2, filename=filename)
        return
    if not paths2:
        wsvg(paths1, attributes=attributes1, filename=filename)
        return

    # If both files have content, combine them
    combined_paths = paths1 + paths2
    combined_attributes = attributes1 + attributes2
    wsvg(combined_paths, attributes=combined_attributes, filename=filename)



def grabLeftMostPointOfPaths(paths):
    """Grab the left most point from a path or a list of paths"""
    min_x = float('inf')
    min_point = None

    # Convert single path to a list for consistent processing
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        for segment in path:
            # Sample points along the segment
            for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
                pt = segment.point(t)
                if pt.real < min_x:
                    min_x = pt.real
                    min_point = pt

    return min_point


def grabRightMostPointOfPaths(paths):
    """Grab the right most point from a path or a list of paths"""
    max_x = float('-inf')
    max_point = None

    # Convert single path to a list for consistent processing
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        for segment in path:
            # Sample points along the segment
            for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
                pt = segment.point(t)
                if pt.real > max_x:
                    max_x = pt.real
                    max_point = pt

    return max_point

def grabTopMostPointOfPaths(paths):
    """Grab the top most point from a path or a list of paths"""
    min_y = float('inf')  # Using min_y since in SVG coordinate system, lower y values are higher up
    min_point = None
    # Convert single path to a list for consistent processing
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        for segment in path:
            # Sample points along the segment
            for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
                pt = segment.point(t)
                if pt.imag < min_y:
                    min_y = pt.imag
                    min_point = pt
    return min_point


def grabBottomMostPointOfPaths(paths):
    """Grab the bottom most point from a path or a list of paths"""
    max_y = float('-inf')  # Using max_y since in SVG coordinate system, higher y values are lower down
    max_point = None
    # Convert single path to a list for consistent processing
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        for segment in path:
            # Sample points along the segment
            for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
                pt = segment.point(t)
                if pt.imag > max_y:
                    max_y = pt.imag
                    max_point = pt
    return max_point

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
    width = float(root.get("width", "500")) # Default to 500 if missing
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
         dimensions=(target_width, target_height),
         svg_attributes={'viewBox': f'0 0 {target_width} {target_height}'})


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


def mirrorSVGOver45DegreeLine(input_svg, output_svg, point, width, height):
    """
    Mirrors SVG paths over a 45-degree line that travels downward (y = -x + c) through the given point.
    
    Args:
        input_svg: Path to input SVG file
        output_svg: Path to output SVG file
        point: A complex number representing the point through which the 45-degree line passes
        width: Width of the SVG
        height: Height of the SVG
    """
    paths, attributes = svg2paths(input_svg)

    # Extract coordinates of the point
    a = point.real
    b = point.imag
    const = a + b  # The constant in the line equation y = -x + const

    # Mirror each path over the 45-degree line y = -x + const
    mirrored_paths = []
    for path in paths:
        mirrored_segments = []
        for segment in path:
            if isinstance(segment, Line):
                mirrored_segments.append(
                    Line(
                        complex(const - segment.start.imag, const - segment.start.real),
                        complex(const - segment.end.imag, const - segment.end.real)
                    )
                )
            elif isinstance(segment, CubicBezier):
                mirrored_segments.append(
                    CubicBezier(
                        complex(const - segment.start.imag, const - segment.start.real),
                        complex(const - segment.control1.imag, const - segment.control1.real),
                        complex(const - segment.control2.imag, const - segment.control2.real),
                        complex(const - segment.end.imag, const - segment.end.real)
                    )
                )
            elif isinstance(segment, QuadraticBezier):
                mirrored_segments.append(
                    QuadraticBezier(
                        complex(const - segment.start.imag, const - segment.start.real),
                        complex(const - segment.control.imag, const - segment.control.real),
                        complex(const - segment.end.imag, const - segment.end.real)
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

    if pattern_paths == []:
        return None

    # Save the pattern-only paths to a new SVG
    wsvg(pattern_paths, attributes=pattern_attrs, filename=output_filename)

    return output_filename


"""Cropping of SVG files"""

def clip_path_to_boundary(path, boundary, width, height, close_path, num_samples_line=1, num_samples_bezier=20):
    """
    Clips a given path to the boundary using Shapely geometric operations.
    Samples each segment to better approximate curves, then optionally closes the resulting
    path by connecting the entry and exit points along the boundary, including any
    missing corner points.
    
    Args:
        path: The path to clip
        boundary: The boundary to clip against
        width: The width of the boundary
        height: The height of the boundary
        close_path: If True, the path will be closed; if False, no additional segments will be added to close it
        num_samples_line: Number of samples to take for line segments
        num_samples_bezier: Number of samples to take for bezier curves
        
    Returns:
        A new Path object representing the clipped path
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

            # Create line segments between consecutive points
            segments = []
            for i in range(len(coords) - 1):
                start = complex(coords[i][0], coords[i][1])
                end = complex(coords[i+1][0], coords[i+1][1])
                segments.append(Line(start, end))

            new_path = Path(*segments)
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
                    # and close_path is True, insert a connecting segment.
                    # Otherwise, keep the segments disconnected.
                    if (abs(all_coords[-1][0] - line_coords[0][0]) > 1e-6 or
                        abs(all_coords[-1][1] - line_coords[0][1]) > 1e-6):
                        if close_path:
                            all_coords.append(line_coords[0])
                            print("Adding connecting segment")
                        else:
                            # Start a new segment without connecting
                            print("Keeping segments disconnected")
                            # Force break by adding None as a marker (will be filtered out later)
                            all_coords.append(None)
                    all_coords.extend(line_coords)

            # Create segments from coordinates, handling disconnected parts
            segments = []
            for i in range(len(all_coords) - 1):
                # Skip if current or next coordinate is None (disconnection marker)
                if all_coords[i] is None or all_coords[i+1] is None:
                    continue

                start = complex(all_coords[i][0], all_coords[i][1])
                end = complex(all_coords[i+1][0], all_coords[i+1][1])
                segments.append(Line(start, end))

            new_path = Path(*segments)
            return new_path

        print("Warning: Unexpected geometry type from intersection:", type(clipped_shape))
        return None

    except Exception as e:
        print("Error while clipping path:", e)
        return None


def crop_svg(paths, starting_x, starting_y, width, height, close_path=True):
    """
    Crops all paths to fit within the given square_size.
    """
    #boundary = Polygon([(0, 0), (width, 0), (width,height), (0, height)])
    boundary = Polygon([(starting_x, starting_y), (starting_x + width, starting_y), (starting_x+ width, starting_y + height), (starting_x, starting_y + height)])
    #print("\nBoundary Polygon:", boundary)
    #print("Total Paths Received for Clipping:", len(paths))

    clipped_paths = []
    for path in paths:
        clipped = clip_path_to_boundary(path, boundary, width, height, close_path)
        if clipped:
            if isinstance(clipped, list):  # Handle MultiLineString cases
                clipped_paths.extend(clipped)
            else:
                clipped_paths.append(clipped)

    #print("Final Clipped Paths:", clipped_paths)

    return clipped_paths


def cropPrep(pattern, output_name, cropped_size, angle):

    # Step 1: Rotate the pattern 45 degrees clockwise
    rotated_path_name = f"{getFileStepCounter()}_rotated_pattern_step.svg"
    incrementFileStepCounter()
    rotateSVG(pattern, rotated_path_name, angle)

    # Step 2: Translate to correct position after rotation
    translateSVGBy(rotated_path_name, output_name, cropped_size, cropped_size)


def cropToTopHalf(input_svg, output_svg):
    # Crop to only the top half of the SVG
        paths, attributes = svg2paths(input_svg)
        stencil_1_clipped_paths = crop_svg(paths, 0, 0, 500, 500 // 2)

        # Adjust attributes if needed
        if len(stencil_1_clipped_paths) > len(attributes):
            attributes = attributes * len(stencil_1_clipped_paths)

        # Save the cropped half
        wsvg(stencil_1_clipped_paths, attributes=attributes, filename=output_svg,
                dimensions=(getDrawingSquareSize(), getDrawingSquareSize()))


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


def get_rgb_from_qcolor(qcolor):
    return (qcolor.blue(), qcolor.green(), qcolor.red())

def calculate_distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def drawCheckerboardOnPixmap(pixmap, painter):
    classic_lines_indices = getClassicIndicesLineDeleteList()
    lines = getNumClassicLines() * 2
    # Draw standard checkerboard pattern
    size = pixmap.size()
    width, height = size.width(), size.height()

    # Calculate square size to get `lines` squares along the diagonal
    diagonal = math.hypot(width, height)
    square_size = math.ceil(diagonal / lines)

    # Create a temporary pixmap for the checkerboard
    checker_pixmap = QPixmap(size)
    checker_pixmap.fill(Qt.GlobalColor.transparent)
    checker_painter = QPainter(checker_pixmap)

    rows = size.height() // square_size + 2
    cols = size.width() // square_size + 2
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                color = getShapeColor()
            else:
                color = getBackgroundColor()

            x = col * square_size
            y = row * square_size
            checker_painter.fillRect(QRect(x, y, square_size, square_size), QBrush(color))

    checker_painter.end()

    rotated = checker_pixmap.transformed(QTransform().rotate(-90), Qt.TransformationMode.SmoothTransformation)

    # Draw rotated checkerboard onto the original pixmap using the passed painter
    x_offset = (pixmap.width() - rotated.width()) // 2 + 20
    y_offset = (pixmap.height() - rotated.height()) // 2 - 20
    painter.drawPixmap(x_offset, y_offset, rotated)

def saveSvgFileAsPixmap(filepath, size=QSize(600, 600)):
    renderer = QSvgRenderer(filepath)

    pixmap = QPixmap(size)
    pixmap.fill(getBackgroundColor())  # Fill with background color background

    painter = QPainter(pixmap)
    if getCurrentPatternType() == PatternType.Classic:
        drawCheckerboardOnPixmap(pixmap, painter)

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


def convertLineToRectangle(input_line):
    if isinstance(input_line, Line):
            # Get start and end points of the line
            start = input_line.start
            end = input_line.end

            # Create a rectangle from the line endpoints
            rect_width = abs(end.real - start.real)
            rect_height = getLineThicknessAndExtension()
            x = min(start.real, end.real)
            y = min(start.imag, end.imag) - rect_height / 2

            # Create a rectangular path using 4 line segments
            rect_path = Path(
                Line(complex(x, y), complex(x + rect_width, y)),                   # top line
                Line(complex(x + rect_width, y), complex(x + rect_width, y + rect_height)),  # right line
                Line(complex(x + rect_width, y + rect_height), complex(x, y + rect_height)),  # bottom line
                Line(complex(x, y + rect_height), complex(x, y))                   # left line
            )
            return rect_path


def convertLinesToRectangles(input_svg, output_svg):
    paths, attributes = svg2paths(input_svg)

    # Create a new SVG drawing
    dwg = svgwrite.Drawing(output_svg, profile='tiny')

    # Iterate through each path and convert lines to rectangles
    for path in paths:
        for segment in path:
            if isinstance(segment, Line):
                # Get start and end points of the line
                start = segment.start
                end = segment.end

                # Create a rectangle from the line endpoints
                rect_width = abs(end.real - start.real)
                rect_height = getLineThicknessAndExtension()
                x = min(start.real, end.real)
                y = min(start.imag, end.imag) - rect_height / 2

                # Add the rectangle to the SVG drawing
                dwg.add(dwg.rect(insert=(x, y), size=(rect_width, rect_height), fill='black'))

    # Save the new SVG file
    dwg.save()


def extractSemiCirclesFromPattern(mirrored_pattern, bottom_stencil_semi_circles, top_stencil_semi_circles, pattern_no_semi_circles, width, height, square_size, side_type, n_lines):
    paths, attributes = svg2paths(mirrored_pattern)
    bottom_stencil_semi_circle_paths = []
    bottom_stencil_semi_circles_attributes = []
    top_stencil_semi_circle_paths = []
    top_stencil_semi_circles_attributes = []
    pattern_no_semi_circles_paths = []
    pattern_no_semi_circles_attributes = []

    # Iterate through each path and filter out lines from semi-circle paths
    for path, attribute in zip(paths, attributes):
        if 'stroke-width' in attribute and attribute['stroke-width'] == '2':
            print("FOUND SEMI CIRCLE")
            print("SEMI CIRCLE PATH: ", path)

            # Find the longest line segment
            # Find the two longest lines
            line_segments = [(segment, segment.length()) for segment in path if isinstance(segment, Line)]
            line_segments.sort(key=lambda x: x[1], reverse=True)

            # Get the two longest lines (if there are at least two)
            longest_lines = [segment for segment, _ in line_segments[:min(2, len(line_segments))]]

            # Check if the longest line is vertical (similar x-coordinates)
            top_circle = False
            if longest_lines:
                longest_line = longest_lines[0]
                x_diff = abs(longest_line.start.real - longest_line.end.real)
                y_diff = abs(longest_line.start.imag - longest_line.end.imag)

                if x_diff < y_diff / 5:
                    top_circle = True

            # Create a list of all segments except the two longest lines
            non_line_segments = [segment for segment in path if segment not in longest_lines]

            if non_line_segments:  # Only add if there are segments left
                print("SEMI CIRCLE FOUND part 2")
                filtered_path = Path(*non_line_segments)
                if top_circle:
                    wsvg(filtered_path, attributes=[attribute], filename="temp_semi_circle.svg", dimensions=(400, 400))
                    temp_paths, temp_attributes = svg2paths("temp_semi_circle.svg")
                    print("temp paths: ", temp_paths)
                    print("temp attributes: ", temp_attributes)
                    # rotate top_semi_circles -90 degrees
                    rotated_bottom_stencil_semi_circles = f"{getFileStepCounter()}_rotated_top_semi_circles.svg"
                    incrementFileStepCounter()
                    right_most_point = max(filtered_path, key=lambda p: p.start.real).start.real
                    top_most_point_y = max(filtered_path, key=lambda p: p.start.imag).start.imag
                    bottom_most_point_y = min(filtered_path, key=lambda p: p.start.imag).start.imag
                    mid_point_y = (top_most_point_y + bottom_most_point_y) / 2
                    rotateSVG("temp_semi_circle.svg", rotated_bottom_stencil_semi_circles, -90, right_most_point + (top_most_point_y - bottom_most_point_y) / 2, mid_point_y)

                    # mirror the top_semi_circles over the y-axis
                    mirrored_bottom_stencil_semi_circles = f"{getFileStepCounter()}_mirrored_top_semi_circles.svg"
                    incrementFileStepCounter()
                    mirrorSVGOverYAxisWithX(rotated_bottom_stencil_semi_circles, mirrored_bottom_stencil_semi_circles, width, height, getMargin() / 1.125 + square_size * 1.5)

                    translated_top_semi_circles = mirrored_bottom_stencil_semi_circles
                    if side_type == SideType.OneSided:
                        # translate the top_semi_circles to the bottom stencil position
                        translated_top_semi_circles = f"{getFileStepCounter()}_translated_top_semi_circles.svg"
                        incrementFileStepCounter()
                        translateSVGBy(mirrored_bottom_stencil_semi_circles, translated_top_semi_circles, 0, height)

                    corrected_filtered_path, _ = svg2paths(translated_top_semi_circles)

                    corrected_filtered_path = corrected_filtered_path[0]

                    bottom_stencil_semi_circle_paths.append(corrected_filtered_path)
                    bottom_stencil_semi_circles_attributes.append(attribute)
                else:
                    top_stencil_semi_circle_paths.append(filtered_path)
                    top_stencil_semi_circles_attributes.append(attribute)

                print("ATTRIBUTE: ", attribute)

                print("SEMI CIRCLE FILTERED PATH: ", filtered_path)
        else:
            pattern_no_semi_circles_paths.append(path)
            pattern_no_semi_circles_attributes.append(attribute)

    print("TOP SEMI CIRCLE PATHS: ", bottom_stencil_semi_circle_paths)
    print("BOTTOM SEMI CIRCLE PATHS: ", top_stencil_semi_circle_paths)

    if side_type == SideType.TwoSided:
        if bottom_stencil_semi_circle_paths:
            wsvg(bottom_stencil_semi_circle_paths, attributes=bottom_stencil_semi_circles_attributes, filename=bottom_stencil_semi_circles, dimensions=(width, height))
        if top_stencil_semi_circle_paths:
            wsvg(top_stencil_semi_circle_paths, attributes=top_stencil_semi_circles_attributes, filename=top_stencil_semi_circles, dimensions=(width, height))

        combined_semi_circles = f"{getFileStepCounter()}_combined_semi_circles.svg"
        incrementFileStepCounter()
        combineStencils(bottom_stencil_semi_circles, top_stencil_semi_circles, combined_semi_circles)

        two_sided_combined_semi_circles = f"{getFileStepCounter()}_translated_combined_semi_circles.svg"
        incrementFileStepCounter()
        if fileIsNonEmpty(combined_semi_circles):
            all_paths, all_attrs = svg2paths(combined_semi_circles)
            wsvg(all_paths, attributes=all_attrs, filename=two_sided_combined_semi_circles, dimensions=(width, height))

        translated_combined_semi_circles = f"{getFileStepCounter()}_translated_combined_semi_circles.svg"
        incrementFileStepCounter()
        if fileIsNonEmpty(two_sided_combined_semi_circles):
            translateSVGBy(two_sided_combined_semi_circles, translated_combined_semi_circles, 0, height)

            all_paths_2, all_attrs_2 = svg2paths(translated_combined_semi_circles)

            bottom_stencil_semi_circle_paths = all_paths_2
            top_stencil_semi_circle_paths = all_paths
            bottom_stencil_semi_circles_attributes = all_attrs_2
            top_stencil_semi_circles_attributes = all_attrs

    if bottom_stencil_semi_circle_paths:
        wsvg(bottom_stencil_semi_circle_paths, attributes=bottom_stencil_semi_circles_attributes, filename=bottom_stencil_semi_circles, dimensions=(width, height))
    if top_stencil_semi_circle_paths:
        wsvg(top_stencil_semi_circle_paths, attributes=top_stencil_semi_circles_attributes, filename=top_stencil_semi_circles)
    if pattern_no_semi_circles_paths:
        wsvg(pattern_no_semi_circles_paths, attributes=pattern_no_semi_circles_attributes, filename=pattern_no_semi_circles)


def is_point_on_line(point, line, tolerance=2):
    """
    Check if a point is on a given line segment within a small degree of tolerance.

    :param point: A complex number representing the point (e.g., 187.5+312.5j).
    :param line: A list of four numbers [x1, y1, x2, y2] representing the start and end points of the line segment.
    :param tolerance: A small degree of tolerance to account for floating-point imprecision.
    :return: True if the point is on the line segment, False otherwise.
    """
    x1, y1, x2, y2 = line
    px, py = point.real, point.imag

    # Calculate the squared length of the line segment
    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Handle the case where the line segment is a single point
    if line_length_sq == 0:
        return abs(px - x1) <= tolerance and abs(py - y1) <= tolerance

    # Calculate the projection of the point onto the line segment
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq

    # Clamp t to the range [0, 1] to ensure the projection is on the segment
    t = max(0, min(1, t))

    # Find the closest point on the line segment to the given point
    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)

    # Calculate the distance from the point to the closest point on the line segment
    distance_sq = (closest_x - px) ** 2 + (closest_y - py) ** 2

    # Check if the distance is within the tolerance
    return distance_sq <= tolerance ** 2


def findClassicLinesToDelete(left_snap_point, right_snap_point):
    # INSTEAD OF PASSING AN UPDATED snap_points AND classic_cuts, PASS THE ORIGINAL snap_points and classic_cuts (as global variables)
    # Sort the points based on their x-coordinates
    sorted_points = sorted(getClassicPatternSnapPoints(), key=lambda p: p.real)
    classic_cuts = getClassicPatternClassicLines()

    # determine orientation of the line given left and right snap points
    line_orientation_up = False
    if left_snap_point.imag > right_snap_point.imag:
        line_orientation_up = True

    print("line orientation up: ", line_orientation_up)

    # Figure out the snapped classic line index
    current_classic_line_index = classic_cuts[0][1] if line_orientation_up else classic_cuts[1][1]
    current_classic_line = classic_cuts[0][0] if line_orientation_up else classic_cuts[1][0]

    print("left snap point: ", left_snap_point)
    print("starting classic line index: ", current_classic_line_index)
    print("starting classic line: ", current_classic_line)

    for i in range(current_classic_line_index - 1, len(classic_cuts), 2):
        current_classic_line = classic_cuts[i][0]

        if is_point_on_line(left_snap_point, current_classic_line):
            print("allan snap point found")
            current_classic_line_index = classic_cuts[i][1]
            print("current classic line index: ", current_classic_line_index)
            break

    # Find number of intersections in between the left and right snap points
    num_intersections_counter = 1
    for i in range(len(sorted_points) - 1):
        current_snap_point = sorted_points[i]

        # Check if the current point is in between the left and right snap points on the current classic line
        if is_point_on_line(current_snap_point, current_classic_line):
            if left_snap_point.real < current_snap_point.real < right_snap_point.real:
                num_intersections_counter += 1

    # Determine the indices to delete by looking at the num_intersections_counter and subtracting that from the classic line index
    num_classic_lines_to_delete = math.floor(num_intersections_counter * 0.5)

    # Figure out the indices to delete
    indices_to_delete = []
    for i in range(num_classic_lines_to_delete):
        incrementor = (1 + i) * 2 if line_orientation_up else (1 + i) * (-2)
        indices_to_delete.append(current_classic_line_index - incrementor)

    print ("num intersections: ", num_intersections_counter)
    print("num classic lines to delete: ", num_classic_lines_to_delete)
    print("indices to delete: ", indices_to_delete)

    return indices_to_delete


def fileIsNonEmpty(file_path):
    try:
        paths, _ = svg2paths(file_path)
        return True
    except Exception as e:
        False
