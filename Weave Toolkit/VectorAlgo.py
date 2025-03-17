import sys
from PyQt6.QtSvg import QSvgRenderer, QSvgGenerator
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPainter, QTransform, QPixmap, QColor
from PyQt6.QtCore import QSize, QByteArray, QRectF
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon

import xml.etree.ElementTree as ET
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier, QuadraticBezier, parse_path

import cv2 as cv
import numpy as np
import math
import tempfile
import svgwrite
import copy
from svgwrite.container import Group
from collections import namedtuple

from ShapeMode import (
    ShapeMode
)

from PatternType import (
    PatternType
)

from SideType import (
    SideType
)

# VectorAlgo will be used when the user presses the "Update SVG" button

# The algorithm will be given a image, of SVG type, of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting), and user drawing
#   - this PNG that is returned is the final image once the algorithm has processesed the user drawing and weave pattern
# 2. Assembly instructions to guide the user in weaving the output image

MARGIN = 31
FILE_STEP_COUNTER = 1


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


def get_edge(pt, width, height, tol=1e-6):
    """Determine which edge of the square boundary a point lies on."""
    x, y = pt
    if abs(y) < tol:
        return 0  # bottom
    elif abs(x - width) < tol:
        return 1  # right
    elif abs(y - height) < tol:
        return 2  # top
    elif abs(x) < tol:
        return 3  # left
    elif x < tol or y < tol or x > width - tol or y > height - tol:
        return None  # Points very close but not exactly on an edge
    else:
        return None


def close_line_with_corners(coords, width, height, tol=1e-6):
    """
    Closes a polygon defined by `coords` by appending missing boundary corner points.
    This ensures that if the clipped path's endpoints lie on different edges,
    the boundary's corners are inserted between them.
    """
    if len(coords) < 2:
        return coords

    first = coords[0]
    last = coords[-1]

    edge_first = get_edge(first, width, height, tol)
    edge_last = get_edge(last, width, height, tol)

    # If either point isn't exactly on the boundary, simply close the loop.
    if edge_first is None or edge_last is None:
        if abs(first[0]-last[0]) > tol or abs(first[1]-last[1]) > tol:
            coords.append(first)
        return coords

    # If on the same edge, just close the loop if needed.
    if edge_first == edge_last:
        if abs(first[0]-last[0]) > tol or abs(first[1]-last[1]) > tol:
            coords.append(first)
        return coords

    # Define square corners in clockwise order.
    corners = [(0, 0), (width, 0), (width, height), (0, height)]

    # Insert corners from the last point's edge to the first point's edge,
    # following the boundary in clockwise order.
    inserted = []
    current_edge = edge_last
    # Loop until we reach the edge of the first point.
    while current_edge != edge_first:
        current_edge = (current_edge + 1) % 4
        inserted.append(corners[current_edge])
        if current_edge == edge_first:
            break

    # Build the new coordinate list: original coords, then the inserted corners, then close.
    new_coords = coords[:]  # make a copy
    new_coords.extend(inserted)
    #new_coords.append(first)
    return new_coords

from svgpathtools import Path, Line, CubicBezier, QuadraticBezier
from shapely.geometry import LineString, Polygon, MultiLineString

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


def translateSVGBy(input_svg, output_svg, x_shift, y_shift):
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

def createFinalHeartDisplay(image):
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

    # calculate the semi-circle positions
    point1 = [center[0] - half_size, center[1]]
    point2 = [center[0], center[1] + half_size]

    square_width = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    radius = math.floor(square_width // 2)

    halfway_point_upper_left = [(center[0] - half_size + center[0]) // 2, (center[1] + center[1] - half_size) //2]
    halfway_point_upper_right = [(center[0] + center[0] + half_size) // 2, (center[1] - half_size + center[1]) //2]

    # draw the semi-circles
    cv.ellipse(mask, halfway_point_upper_left, (radius, radius), 0, -225, -45, line_color, thickness=3)
    cv.ellipse(mask, halfway_point_upper_right, (radius, radius), 0, -135, 45, line_color, 3)

    # rotate the heart outline so it is ready to have the pattern overlayed on it
    rotated_mask = rotateImage(mask, -45)

    # scale the pattern to fit inside the square of the heart
    square_width_rounded = math.floor(square_width) - 15
    scaled_pattern = cv.resize(image, (square_width_rounded, square_width_rounded), interpolation=cv.INTER_LANCZOS4)

    # Calculate coordinates to overlay scaled_pattern on the square portion of the heart
    x_center = (rotated_mask.shape[1] - square_width_rounded) // 2
    y_center = (rotated_mask.shape[0] - square_width_rounded) // 2

    # Overlay scaled_pattern onto the square portion of the heart
    rotated_mask[y_center:y_center + square_width_rounded, x_center:x_center + square_width_rounded] = scaled_pattern

    # rotate the heart back to its original, upright, position
    reverse_rotated_mask = rotateImage(rotated_mask, 45)

    return reverse_rotated_mask


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

    dwg.save()

    return file_name


def combineStencils(first_stencil, second_stencil, filename='combined.svg'):
    paths1, attributes1 = svg2paths(first_stencil)
    paths2, attributes2 = svg2paths(second_stencil)

    combined_paths = paths1 + paths2
    combined_attributes = attributes1 + attributes2

    wsvg(combined_paths, attributes=combined_attributes, filename=filename)


def overlayDrawingOnStencil(stencil_file, user_drawing_file, size, square_size, pattern_type, margin_x=MARGIN, margin_y=0, filename='combined_output.svg'):
        global FILE_STEP_COUNTER

        translated_user_path = f"{FILE_STEP_COUNTER}_translated_for_overlay.svg"
        FILE_STEP_COUNTER += 1

        x_multi = 0
        y_multi = 0
        if pattern_type == PatternType.Simple:
            x_multi = 2
            y_multi = 2
        else:
            x_multi = 4
            y_multi = 3

        x_shift = margin_x * x_multi + square_size // 2
        y_shift = margin_y + (margin_x * y_multi)
        translateSVGBy(user_drawing_file, translated_user_path, x_shift, y_shift)

        paths1, attributes1 = svg2paths(stencil_file)
        paths2, attributes2 = svg2paths(translated_user_path)

        # print("overlay user attributes: ", attributes2)

        combined_paths = paths1 + paths2
        combined_attributes = attributes1 + attributes2

        # print("overlay user attributes 2: ", combined_attributes)

        dwg = svgwrite.Drawing(filename, size=(size, size))

        # Add each path from the combined paths to the new SVG
        for path, attr in zip(combined_paths, combined_attributes):
            # Extract stroke, fill, and stroke-width attributes (if they exist)
            stroke = attr.get('stroke', 'black')        # Default to black if no stroke is defined
            fill = attr.get('fill', 'none')             # Default to 'none' if no fill is defined
            stroke_width = attr.get('stroke-width', 1)  # Default to 1 if no stroke width is defined

            dwg.add(dwg.path(d=path.d(), stroke=stroke, fill=fill, stroke_width=stroke_width))

        dwg.save()
        return filename


def overlayPatternOnStencil(pattern, stencil, size, stencil_number, pattern_type, margin=MARGIN):
    global FILE_STEP_COUNTER

    # scale the pattern
    square_size = size // 2 // 1.5 - margin
    inner_cut_size = square_size - (margin * 2)
    resized_pattern_name = f"{FILE_STEP_COUNTER}_scaled_pattern.svg"
    FILE_STEP_COUNTER += 1
    resizeSVG(pattern, resized_pattern_name, inner_cut_size)

    # shift the pattern right and down (overlay on stencil)
    combined_output_name = f"{FILE_STEP_COUNTER}_stencil_{stencil_number}_overlayed.svg"
    FILE_STEP_COUNTER += 1
    margin_y = 0 if stencil_number == 1 else size // 2
    overlayDrawingOnStencil(stencil, resized_pattern_name, size, square_size, pattern_type, margin, margin_y, combined_output_name)

    return combined_output_name


def drawSimpleStencil(width, height, starting_y, margin_x=MARGIN, line_color='black', file_name="allans_test.svg"):
    dwg = svgwrite.Drawing(file_name, size=(width, height + starting_y))

    # Define the square size
    square_size = (height // 1.5) - margin_x
    margin_y = margin_x + starting_y
    extension = 20  # Amount to extend the lines

    # Define the left square margins
    left_top_line_start = (margin_x + square_size // 2, margin_y)
    left_top_line_end = (left_top_line_start[0] + square_size, left_top_line_start[1])
    left_bottom_line_start = (left_top_line_start[0], left_top_line_start[1] + square_size)
    left_bottom_line_end = (left_bottom_line_start[0] + square_size, left_bottom_line_start[1])

    # Define the right square margins
    right_top_line_start = left_top_line_end
    right_top_line_end = (right_top_line_start[0] + square_size, right_top_line_start[1])
    right_bottom_line_start = left_bottom_line_end
    right_bottom_line_end = (right_bottom_line_start[0] + square_size, right_bottom_line_start[1])

    # Draw the inner line cuts with extended length
    dwg.add(dwg.line(start=(left_top_line_start[0] - extension, left_top_line_start[1] + margin_x),
                      end=(right_top_line_end[0] + extension, right_top_line_end[1] + margin_x),
                      stroke="brown", stroke_width=3))

    dwg.add(dwg.line(start=(left_bottom_line_start[0] - extension, left_bottom_line_start[1] - margin_x),
                      end=(right_bottom_line_end[0] + extension, right_bottom_line_end[1] - margin_x),
                      stroke="brown", stroke_width=3))

    dwg.save()

    return file_name


def combine_overlapping_paths(paths, attrs, tolerance=1e-6):
    """
    Combine overlapping paths into single shapes using Shapely and simplify resulting paths.
    
    Args:
        paths: List of svgpathtools Path objects
        attrs: List of attribute dictionaries for each path
        tolerance: Tolerance for determining if points are close enough to be considered overlapping
    
    Returns:
        Tuple of (combined_paths, combined_attrs) with simplified lines
    """
    if not paths:
        return [], []

    # Print the number of input paths and segments
    print(f"Number of input paths: {len(paths)}")
    for i, path in enumerate(paths):
        print(f"Input path {i} has {len(path)} segments")
    
    # Convert svgpathtools paths to shapely geometries
    shapely_polygons = []
    path_to_attr_map = {}

    for i, path in enumerate(paths):
        # Sample points along the path to create a polygon
        points = []
        for segment in path:
            for t in np.linspace(0, 1, 10):  # Sample 10 points per segment
                pt = segment.point(t)
                points.append((pt.real, pt.imag))

        # Ensure we have the endpoint as well
        last_pt = path[-1].point(1.0)
        points.append((last_pt.real, last_pt.imag))

        if len(points) >= 3:  # Need at least 3 points to form a polygon
            try:
                poly = Polygon(points)
                if poly.is_valid:
                    shapely_polygons.append(poly)
                    path_to_attr_map[poly] = attrs[i]
            except Exception as e:
                print(f"Error creating polygon: {e}")

    if not shapely_polygons:
        return paths, attrs

    # Merge overlapping polygons
    result_polygons = []
    result_attrs = []
    processed = set()

    for i, poly1 in enumerate(shapely_polygons):
        if i in processed:
            continue

        current_poly = poly1
        current_attr = path_to_attr_map[poly1]
        processed.add(i)

        # Check for overlaps with other polygons
        for j, poly2 in enumerate(shapely_polygons):
            if j in processed:
                continue

            if current_poly.intersects(poly2) or current_poly.distance(poly2) < tolerance:
                try:
                    # Merge the polygons
                    current_poly = current_poly.union(poly2)
                    # Keep attributes of the first polygon
                    processed.add(j)
                except Exception as e:
                    print(f"Error merging polygons: {e}")

        result_polygons.append(current_poly)
        result_attrs.append(current_attr)

    # Convert back to svgpathtools paths with simplified segments
    combined_paths = []
    combined_attrs = []

    for poly, attr in zip(result_polygons, result_attrs):
        try:
            if isinstance(poly, Polygon):
                # Extract exterior coordinates and simplify the polygon
                exterior = poly.exterior
                # Simplify the exterior to remove redundant points
                simplified = exterior.simplify(tolerance)
                coords = list(simplified.coords)

                # Create line segments for the outline
                path_segments = []
                
                # Simplify collinear segments
                if len(coords) > 2:
                    simplified_coords = [coords[0]]
                    
                    for i in range(1, len(coords) - 1):
                        # Check if three points are collinear
                        p1 = simplified_coords[-1]
                        p2 = coords[i]
                        p3 = coords[i + 1]
                        
                        # Calculate slopes or check vertical alignment
                        if abs(p1[0] - p2[0]) < tolerance and abs(p2[0] - p3[0]) < tolerance:
                            # Points are vertically aligned, skip the middle point
                            continue
                        
                        if abs(p1[1] - p2[1]) < tolerance and abs(p2[1] - p3[1]) < tolerance:
                            # Points are horizontally aligned, skip the middle point
                            continue
                            
                        # Check for general collinearity
                        if abs((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) < tolerance:
                            # Points are collinear, skip the middle point
                            continue
                            
                        simplified_coords.append(p2)
                    
                    simplified_coords.append(coords[-1])
                    coords = simplified_coords
                
                # Create the path segments from simplified coordinates
                for i in range(len(coords) - 1):
                    start = complex(coords[i][0], coords[i][1])
                    end = complex(coords[i+1][0], coords[i+1][1])
                    path_segments.append(Line(start, end))

                if path_segments:
                    combined_paths.append(Path(*path_segments))
                    combined_attrs.append(attr)

            elif isinstance(poly, MultiPolygon):
                # Handle each polygon in the multipolygon
                for geom in poly.geoms:
                    exterior = geom.exterior
                    simplified = exterior.simplify(tolerance)
                    coords = list(simplified.coords)
                    
                    # Simplify collinear segments as above
                    if len(coords) > 2:
                        simplified_coords = [coords[0]]
                        
                        for i in range(1, len(coords) - 1):
                            p1 = simplified_coords[-1]
                            p2 = coords[i]
                            p3 = coords[i + 1]
                            
                            if abs(p1[0] - p2[0]) < tolerance and abs(p2[0] - p3[0]) < tolerance:
                                continue
                            
                            if abs(p1[1] - p2[1]) < tolerance and abs(p2[1] - p3[1]) < tolerance:
                                continue
                                
                            if abs((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) < tolerance:
                                continue
                                
                            simplified_coords.append(p2)
                        
                        simplified_coords.append(coords[-1])
                        coords = simplified_coords
                    
                    path_segments = []
                    for i in range(len(coords) - 1):
                        start = complex(coords[i][0], coords[i][1])
                        end = complex(coords[i+1][0], coords[i+1][1])
                        path_segments.append(Line(start, end))

                    if path_segments:
                        combined_paths.append(Path(*path_segments))
                        combined_attrs.append(attr)
        except Exception as e:
            print(f"Error converting polygon to path: {e}")

    # Print the number of output paths and segments
    print(f"Number of output paths: {len(combined_paths)}")
    for i, path in enumerate(combined_paths):
        print(f"Output path {i} has {len(path)} segments")

    return combined_paths, combined_attrs


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


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


def create_classic_pattern_stencils(width, height, starting_y, margin_x=31, line_color='black', file_name="test_1.svg"):
    dwg = svgwrite.Drawing(file_name, size=(width,height+starting_y))

    # define the square size
    square_size = (height // 1.5) - margin_x
    margin_y = margin_x + starting_y

    left_top_line_start = (margin_x + square_size // 2, margin_y)
    left_top_line_end = (left_top_line_start[0] + square_size, left_top_line_start[1])
    left_bottom_line_start = (left_top_line_start[0], left_top_line_start[1] + square_size)
    left_bottom_line_end = (left_bottom_line_start[0] + square_size, left_bottom_line_start[1])

    # draw the right square
    right_top_line_start = left_top_line_end
    right_top_line_end = (right_top_line_start[0] + square_size, right_top_line_start[1])
    right_bottom_line_start = left_bottom_line_end
    right_bottom_line_end = (right_bottom_line_start[0] + square_size, right_bottom_line_start[1])

    offset = 90
    y1 = left_top_line_start[1] + offset     # Top inner line
    y3 = left_top_line_start[1] + square_size - offset
    y2 = (y1 + y3) / 2   # Middle inner line
    dwg.add(dwg.line(start=(left_top_line_start[0], y1), end=(right_top_line_end[0], y1), stroke="brown", stroke_width=3))
    dwg.add(dwg.line(start=(left_top_line_start[0], y2), end=(right_top_line_end[0], y2), stroke="brown", stroke_width=3))
    dwg.add(dwg.line(start=(left_top_line_start[0], y3), end=(right_top_line_end[0], y3), stroke="brown", stroke_width=3))


    dwg.save()

    return file_name


def create_simple_pattern_stencils(width, height, size, stencil_1_pattern, empty_stencil_1, empty_stencil_2, side_type, pattern_type):
    global FILE_STEP_COUNTER

    # Create both simple stencils
    simpleStencil1 = drawSimpleStencil(width, height, 0, file_name=f"{FILE_STEP_COUNTER}_simpleStencil1.svg")
    FILE_STEP_COUNTER += 1
    simpleStencil2 = drawSimpleStencil(width, height, height, file_name=f"{FILE_STEP_COUNTER}_simpleStencil2.svg")
    FILE_STEP_COUNTER += 1

    # Combine all stencils first
    temp1 = f"{FILE_STEP_COUNTER}_temp1.svg"
    FILE_STEP_COUNTER += 1
    temp2 = f"{FILE_STEP_COUNTER}_temp2.svg"
    FILE_STEP_COUNTER += 1
    combined_stencils = f"{FILE_STEP_COUNTER}_combined_stencils.svg"
    FILE_STEP_COUNTER += 1

    # Combine empty stencil 1 with simple stencil 1
    combineStencils(empty_stencil_1, simpleStencil1, temp1)

    # Combine empty stencil 2 with simple stencil 2
    combineStencils(empty_stencil_2, simpleStencil2, temp2)

    # Combine both results
    combineStencils(temp1, temp2, combined_stencils)

    # rotate the pattern 90 degrees counter-clockwise
    rotated_path_name = f"{FILE_STEP_COUNTER}_fixed_pattern_rotation.svg"
    FILE_STEP_COUNTER += 1
    rotateSVG(stencil_1_pattern, rotated_path_name, -90)

    # Now overlay the pattern on the combined stencil
    overlayed_pattern = overlayPatternOnStencil(rotated_path_name, combined_stencils, size, 1, pattern_type)
    return overlayed_pattern, combined_stencils


def create_symmetric_pattern_stencils(stencil_1_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type):
    global FILE_STEP_COUNTER

    cropped_size = int((500 - DRAWING_SQUARE_SIZE) // 2)

    prepped_pattern = f"{FILE_STEP_COUNTER}_prepped_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropPrep(stencil_1_pattern, prepped_pattern, cropped_size, 45)

    half_of_pattern = f"{FILE_STEP_COUNTER}_half_of_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropToTopHalf(prepped_pattern, half_of_pattern)

    # undo the crop prep once the cropping is finished
    post_cropped_pattern = f"{FILE_STEP_COUNTER}_post_cropped_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropPrep(half_of_pattern, post_cropped_pattern, -cropped_size, -45)

    combined_simple_stencil_w_patt, combined_simple_stencil_no_patt = create_simple_pattern_stencils(width, height, size, post_cropped_pattern, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    # rotate the pattern, grab the 2 points on the line of symmetry, and then rotate it back (including the points we grabbed)

    # Draw lines from shapes to the edges of the stencil
    pattern_w_extended_lines = f"{FILE_STEP_COUNTER}_pattern_w_extended_lines.svg"
    FILE_STEP_COUNTER += 1
    drawExtensionLines(combined_simple_stencil_w_patt, combined_simple_stencil_no_patt, pattern_w_extended_lines, side_type, width, height, 0)

    mirrored_pattern_extended = f"{FILE_STEP_COUNTER}_mirrored_pattern_extended.svg"
    FILE_STEP_COUNTER += 1
    if side_type == SideType.OneSided:
        mirrorLines(pattern_w_extended_lines, mirrored_pattern_extended, width, height, pattern_type)
        combinePatternAndMirrorWithStencils(pattern_w_extended_lines, combined_simple_stencil_no_patt, mirrored_pattern_extended)
    
    elif side_type == SideType.TwoSided:
        mirrorLines(pattern_w_extended_lines, mirrored_pattern_extended, width, 0, pattern_type)
        combined_patt_and_mirror = f"{FILE_STEP_COUNTER}_combined_patt_and_mirror.svg"
        FILE_STEP_COUNTER += 1
        combineStencils(pattern_w_extended_lines, mirrored_pattern_extended, combined_patt_and_mirror)
        # create copy of the combined pattern and mirror
        paths, attributes = svg2paths(combined_patt_and_mirror)
        combined_patt_and_mirror_copy = f"{FILE_STEP_COUNTER}_combined_patt_and_mirror_copy.svg"
        FILE_STEP_COUNTER += 1
        paths_copy = copy.deepcopy(paths)
        attributes_copy = copy.deepcopy(attributes)
        wsvg(paths_copy, attributes=attributes_copy, filename=combined_patt_and_mirror_copy, dimensions=(width, height))
        translateSVGBy(combined_patt_and_mirror_copy, combined_patt_and_mirror_copy, 0, height)

        combinePatternAndMirrorWithStencils(combined_patt_and_mirror, combined_simple_stencil_no_patt, combined_patt_and_mirror_copy)


def create_asymmetric_pattern_stencils(stencil_1_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type):
    global FILE_STEP_COUNTER

    cropped_size = int((500 - DRAWING_SQUARE_SIZE) // 2)

    prepped_pattern = f"{FILE_STEP_COUNTER}_prepped_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropPrep(stencil_1_pattern, prepped_pattern, cropped_size, 45)

    half_of_pattern = f"{FILE_STEP_COUNTER}_half_of_pattern.svg"
    FILE_STEP_COUNTER
    cropToTopHalf(prepped_pattern, half_of_pattern)

    translated_for_bottom_half = f"{FILE_STEP_COUNTER}_translated_for_bottom_half.svg"
    FILE_STEP_COUNTER += 1
    translateSVGBy(prepped_pattern, translated_for_bottom_half, 0, -500 // 2)

    prepped_bottom_pattern = f"{FILE_STEP_COUNTER}_prepped_bottom_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropToTopHalf(translated_for_bottom_half, prepped_bottom_pattern)

    re_translated_for_bottom_half = f"{FILE_STEP_COUNTER}_post_cropped_bottom_pattern.svg"
    FILE_STEP_COUNTER += 1
    translateSVGBy(prepped_bottom_pattern, re_translated_for_bottom_half, 0, 500 // 2)

    # undo the crop prep once the cropping is finished
    post_cropped_pattern = f"{FILE_STEP_COUNTER}_post_cropped_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropPrep(half_of_pattern, post_cropped_pattern, -cropped_size, -45)

    # undo the crop prep for bottom half once the cropping is finished
    post_cropped_bottom_pattern = f"{FILE_STEP_COUNTER}_post_cropped_bottom_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropPrep(re_translated_for_bottom_half, post_cropped_bottom_pattern, -cropped_size, -45)

    # --- for top half ---
    combined_simple_stencil_w_top_patt, combined_simple_stencil_no_patt = create_simple_pattern_stencils(width, height, size, post_cropped_pattern, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    top_pattern_w_extended_lines = f"{FILE_STEP_COUNTER}_pattern_w_extended_lines.svg"
    FILE_STEP_COUNTER += 1
    drawExtensionLines(combined_simple_stencil_w_top_patt, combined_simple_stencil_no_patt, top_pattern_w_extended_lines, side_type, width, height, 0)
    # ------

    # --- for bottom half ---
    combined_simple_stencil_w_bot_patt, _ = create_simple_pattern_stencils(width, height, size, post_cropped_bottom_pattern, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    bottom_pattern_w_extended_lines = f"{FILE_STEP_COUNTER}_bottom_pattern_w_extended_lines.svg"
    FILE_STEP_COUNTER += 1
    drawExtensionLines(combined_simple_stencil_w_bot_patt, combined_simple_stencil_no_patt, bottom_pattern_w_extended_lines, side_type, width, height, 0)
    # ------

    mirrored_bottom_pattern_extended = f"{FILE_STEP_COUNTER}_mirrored_bottom_pattern_extended.svg"
    FILE_STEP_COUNTER += 1
    mirrored_top_pattern_extended = f"{FILE_STEP_COUNTER}_mirrored_top_pattern_extended.svg"
    FILE_STEP_COUNTER += 1
    if side_type == SideType.OneSided:
        mirrorLines(bottom_pattern_w_extended_lines, mirrored_bottom_pattern_extended, width, height, pattern_type)
        combinePatternAndMirrorWithStencils(top_pattern_w_extended_lines, combined_simple_stencil_no_patt, mirrored_bottom_pattern_extended)
    
    elif side_type == SideType.TwoSided:
        mirrorLines(top_pattern_w_extended_lines, mirrored_top_pattern_extended, width, 0, pattern_type)
        mirrorLines(bottom_pattern_w_extended_lines, mirrored_bottom_pattern_extended, width, 0, pattern_type)
        combined_patt_and_mirror_top = f"{FILE_STEP_COUNTER}_combined_patt_and_mirror_top.svg"
        FILE_STEP_COUNTER += 1
        combined_patt_and_mirror_bottom = f"{FILE_STEP_COUNTER}_combined_patt_and_mirror_bottom.svg"
        FILE_STEP_COUNTER += 1
        combineStencils(top_pattern_w_extended_lines, mirrored_top_pattern_extended, combined_patt_and_mirror_top)
        combineStencils(bottom_pattern_w_extended_lines, mirrored_bottom_pattern_extended, combined_patt_and_mirror_bottom)
        
        # create copy of the combined pattern and mirror
        paths, attributes = svg2paths(combined_patt_and_mirror_bottom)
        combined_patt_and_mirror_copy = f"{FILE_STEP_COUNTER}_combined_patt_and_mirror_copy.svg"
        FILE_STEP_COUNTER += 1
        paths_copy = copy.deepcopy(paths)
        attributes_copy = copy.deepcopy(attributes)
        wsvg(paths_copy, attributes=attributes_copy, filename=combined_patt_and_mirror_copy, dimensions=(width, height))
        
        # INSTEAD OF THIS TRANSLATE MIGHT BE WHERE WE MIRROR OVER THE X AXIS
        # translateSVGBy(combined_patt_and_mirror_copy, combined_patt_and_mirror_copy, 0, height)
        mirrorSVGOverXAxis(combined_patt_and_mirror_copy, combined_patt_and_mirror_copy, width, height)

        combined_patt_and_mirror = f"{FILE_STEP_COUNTER}_combined_patt_and_mirror.svg"
        FILE_STEP_COUNTER += 1
        combineStencils(combined_patt_and_mirror_top, combined_patt_and_mirror_copy, combined_patt_and_mirror)

        combinePatternAndMirrorWithStencils(combined_patt_and_mirror, combined_simple_stencil_no_patt, combined_patt_and_mirror_copy)


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


def set_fill_to_none(paths, attrs):
    """
    Set the fill attribute to none for all paths.
    
    Args:
        paths: List of svgpathtools Path objects
        attrs: List of attribute dictionaries for each path
    
    Returns:
        Tuple of (paths, updated_attrs)
    """
    updated_attrs = []
    for attr in attrs:
        updated_attr = attr.copy()
        updated_attr['fill'] = 'none'
        updated_attrs.append(updated_attr)
    return paths, updated_attrs


def find_rightmost_vertical_line(svg_file):
    """
    Find the rightmost vertical line in an SVG file.
    
    Args:
        svg_file: Path to the SVG file
        
    Returns:
        Tuple of (path, index) for the rightmost vertical line, or None if no vertical line is found
    """
    paths, attributes = svg2paths(svg_file)
    
    rightmost_x = float('-inf')
    rightmost_vertical_line = None
    rightmost_path_index = -1
    rightmost_segment_index = -1
    
    for path_index, path in enumerate(paths):
        for segment_index, segment in enumerate(path):
            # Check if the segment is a vertical line (or nearly vertical)
            if isinstance(segment, Line):
                # Calculate the angle of the line with the x-axis
                dx = segment.end.real - segment.start.real
                dy = segment.end.imag - segment.start.imag
                
                # Check if line is vertical (slope is close to infinity)
                if abs(dx) < 1e-10:  # Almost zero change in x direction
                    # Find the x-coordinate of this vertical line
                    x_coord = segment.start.real  # or segment.end.real, they're the same
                    
                    # If this is the rightmost vertical line found so far
                    if x_coord > rightmost_x:
                        rightmost_x = x_coord
                        rightmost_vertical_line = segment
                        rightmost_path_index = path_index
                        rightmost_segment_index = segment_index
    
    if rightmost_vertical_line is None:
        return None
    
    return (paths[rightmost_path_index], rightmost_path_index, rightmost_segment_index, rightmost_vertical_line)

def get_vertical_line_endpoints(vertical_line):
    """
    Get the top and bottom points of a vertical line.
    
    Args:
        vertical_line: Line segment object
        
    Returns:
        Tuple of (top_point, bottom_point)
    """
    if vertical_line.start.imag <= vertical_line.end.imag:
        return (vertical_line.start, vertical_line.end)
    else:
        return (vertical_line.end, vertical_line.start)
    

def rotatePoint(point, angle, center=None):
    """
    Rotate a point around the center by the given angle in degrees.
    
    Args:
        point: A complex number or tuple representing the point to rotate
        angle: Angle in degrees to rotate
        center: A tuple (x, y) representing the center of rotation, defaults to (0, 0)
    
    Returns:
        A complex number representing the rotated point
    """
    # Default center is (0, 0)
    if center is None:
        center = (0, 0)
    
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Convert point to complex number if it's a tuple
    if isinstance(point, tuple):
        point = complex(point[0], point[1])
    
    # Convert center to complex number
    center_complex = complex(center[0], center[1])
    
    # Translate point so that center becomes the origin
    translated = point - center_complex
    
    # Rotate the translated point
    rotated = translated * complex(math.cos(angle_rad), math.sin(angle_rad))
    
    # Translate back
    result = rotated + center_complex
    
    return result


def drawExtensionLines(combined_stencil, stencil_pattern, output_name, side_type, width, height, starting_y, margin_x=MARGIN):
    global FILE_STEP_COUNTER
    
    square_size = (height // 1.5) - margin_x
    margin_y = margin_x + starting_y

    combined_paths, combined_attrs = svg2paths(combined_stencil)
    stencil_paths, stencil_attrs = svg2paths(stencil_pattern)

    # Convert paths to strings for comparison
    combined_path_strings = [path.d() for path in combined_paths]
    stencil_path_strings = [path.d() for path in stencil_paths]

    # Find paths that are in combined_paths but not in stencil_paths
    pattern_paths = []
    pattern_attrs = []

    for i, path_str in enumerate(combined_path_strings):
        if path_str not in stencil_path_strings:
            pattern_paths.append(combined_paths[i])
            pattern_attrs.append(combined_attrs[i])

    # Set fill to none for pattern paths
    pattern_paths, pattern_attrs = set_fill_to_none(pattern_paths, pattern_attrs)

    # Combine overlapping paths
    combined_paths, combined_attrs = combine_overlapping_paths(pattern_paths, pattern_attrs)
    wsvg(combined_paths, attributes=combined_attrs, filename="combined_shapes.svg", dimensions=(width, width))

    combined_paths_w_lines = copy.deepcopy(combined_paths)
    combined_attrs_w_lines = copy.deepcopy(combined_attrs)

    # rotate "combined_shapes.svg" 45 degrees clockwise
    rotated_path_name = f"{FILE_STEP_COUNTER}_rotated_pattern_step.svg"
    FILE_STEP_COUNTER += 1
    rotateSVG("combined_shapes.svg", rotated_path_name, 45, width // 2, height // 2)
    
    # find the right-most vertical line of the pattern. Grab the top and bottom points from it
    rightmost_vertical_line = find_rightmost_vertical_line(rotated_path_name)
    if rightmost_vertical_line is None:
        print("No vertical line found.")
        return
    
    # extract the top point from the rightmost vertical line
    path, path_index, segment_index, line = rightmost_vertical_line
    top_point, bottom_point = get_vertical_line_endpoints(line)

    # rotate the top and bottom points back to the original orientation
    top_point_rotated = rotatePoint(top_point, -45, (width // 2, height // 2))
    bottom_point_rotated = rotatePoint(bottom_point, -45, (width // 2, height // 2))

    del combined_paths_w_lines[path_index][segment_index]

    # Draw a line from the bottom most point to the left edge of the stencil and draw a line from the top most point to the right edge of the stencil
    for path, attr in zip(combined_paths, combined_attrs):
        extension = 20

        stencil_square_start = margin_x + square_size // 2

        if side_type == SideType.OneSided:
            # Create a line from the top_point_rotated to the left edge of the stencil
            top_of_los = Line(top_point_rotated, complex(stencil_square_start - extension, top_point_rotated.imag))

            # Create a line from the bottom_point_rotated to the right edge of the stencil
            bottom_of_los = Line(bottom_point_rotated, complex(stencil_square_start + square_size * 2 + extension, bottom_point_rotated.imag))

        elif side_type == SideType.TwoSided:
            # Create a line from the top_point_rotated to the left edge of the stencil
            top_of_los = Line(top_point_rotated, complex(stencil_square_start - extension, top_point_rotated.imag))

            # Create a line from the bottom_point_rotated to the right edge of the stencil
            bottom_of_los = Line(bottom_point_rotated, complex(margin_x + square_size * 1.5, bottom_point_rotated.imag))

        # Add the left_line and right_line to the final paths
        combined_paths_w_lines.append(Path(top_of_los))
        combined_attrs_w_lines.append({'stroke': 'red', 'stroke-width': 1, 'fill': 'none'})
        combined_paths_w_lines.append(Path(bottom_of_los))
        combined_attrs_w_lines.append({'stroke': 'blue', 'stroke-width': 1, 'fill': 'none'})

    # Save the final SVG with extended lines
    wsvg(combined_paths_w_lines, attributes=combined_attrs_w_lines, filename=output_name, dimensions=(width, width))

def mirrorLines(pattern_w_extended_lines, output_name, width, height, pattern_type):
    global FILE_STEP_COUNTER
    global MARGIN

    # mirror the lines over the y-axis
    mirrored_lines = f"{FILE_STEP_COUNTER}_mirrored_lines.svg"
    FILE_STEP_COUNTER += 1
    mirrorSVGOverYAxis(pattern_w_extended_lines, mirrored_lines, width, height)

    paths, _ = svg2paths(pattern_w_extended_lines)
    mirrored_paths, _ = svg2paths(mirrored_lines)

    left_point = grabLeftMostPointOfPaths(mirrored_paths)
    right_point = grabRightMostPointOfPaths(paths)
    # Calculate the distance between the leftmost and rightmost points
    distance_between = abs(left_point.real - right_point.real)

    fixed_rounding_mirrored_lines = f"{FILE_STEP_COUNTER}_fixed_mirrored_lines.svg"
    FILE_STEP_COUNTER += 1
    translateSVGBy(mirrored_lines, fixed_rounding_mirrored_lines, -distance_between, 0)

    # translate the mirrored lines to the correct position
    translateSVGBy(fixed_rounding_mirrored_lines, output_name, distance_between - MARGIN, height)


def combinePatternAndMirrorWithStencils(pattern_w_extended_lines, combined_simple_stencil_no_patt, translated_mirrored_lines, output_name="final_output.svg"):
    global FILE_STEP_COUNTER

    # combine the mirrored lines with the original mirrored pattern
    combined_mirrored_lines = f"{FILE_STEP_COUNTER}_combined_mirrored_lines.svg"
    FILE_STEP_COUNTER += 1
    combineStencils(translated_mirrored_lines, pattern_w_extended_lines, combined_mirrored_lines)

    # combine the combined_mirrored_lines with the stencil pattern
    combineStencils(combined_mirrored_lines, combined_simple_stencil_no_patt, output_name)

    print("saved final stencil")


def createFinalHeartCutoutPatternExport(size, side_type, pattern_type, line_color='black', background_color='white'):
    global FILE_STEP_COUNTER

    print("pattern type: ", pattern_type)
    print("sides: ", side_type)

    width = size
    height = size // 2

    empty_stencil_1 = drawEmptyStencil(width, height, 0, file_name=f"{FILE_STEP_COUNTER}_stencil1.svg")
    FILE_STEP_COUNTER += 1
    empty_stencil_2 = drawEmptyStencil(width, height, height, file_name=f"{FILE_STEP_COUNTER}_stencil2.svg")
    FILE_STEP_COUNTER += 1

    preprocessed_pattern = "preprocessed_pattern.svg"

    if pattern_type == PatternType.Simple:
        print("Creating SIMPLE pattern")
        create_simple_pattern_stencils(width, height, size, preprocessed_pattern, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    elif pattern_type == PatternType.Symmetric:
        print("Creating SYMMETRICAL pattern")

        create_symmetric_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    elif pattern_type == PatternType.Asymmetric:
        print("Creating A-SYMMETRICAL pattern")

        create_asymmetric_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    elif pattern_type == PatternType.Classic:
        print("Creating CLASSIC pattern")
        combined_classic_stencil = f"{FILE_STEP_COUNTER}_combined_classic_stencil.svg"
        FILE_STEP_COUNTER += 1
        classic_stencil1 = create_classic_pattern_stencils(width, height, 0, file_name=f"{FILE_STEP_COUNTER}_classic_stencil1.svg")
        FILE_STEP_COUNTER += 1
        classic_stencil2 = create_classic_pattern_stencils(width, height, height, file_name=f"{FILE_STEP_COUNTER}_classic_stencil2.svg")
        FILE_STEP_COUNTER += 1
        final_stencil = f"{FILE_STEP_COUNTER}_classic_final_stencil.svg"
        FILE_STEP_COUNTER += 1
        combined_classic_stencil_final = f"{FILE_STEP_COUNTER}_combined_classic_stencil_final.svg"
        FILE_STEP_COUNTER += 1
        combineStencils(empty_stencil_1, classic_stencil1, combined_classic_stencil)
        combineStencils(empty_stencil_2, classic_stencil2, final_stencil)
        combineStencils(final_stencil, combined_classic_stencil, combined_classic_stencil_final)

        # resizeSvg(final_stencil, user_decided_export_size)

        # return final_stencil


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


def mainAlgorithmSvg(img, side_type, pattern_type, function='show'):

    match function:

        case 'show':
            # convert SVG to CV Image for createFinalHeartDisplay
            heartPixmap = saveSvgFileAsPixmap(img)
            heartCvImage = savePixmapToCvImage(heartPixmap)

            return createFinalHeartDisplay(heartCvImage)

        case _:
            return createFinalHeartCutoutPatternExport(1200, side_type, pattern_type)
