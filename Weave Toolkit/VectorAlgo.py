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
    translateSVG(rotated_path_name, translated_path_name, -cropped_size, -cropped_size)

    paths, attributes = svg2paths(translated_path_name)

    final_output_path_name = "preprocessed_pattern.svg"
    wsvg(paths, attributes=attributes, filename=final_output_path_name, dimensions=(square_size, square_size))

    # print("pre-processed attributes: ", attributes)

    # print(f"Original path ({len(paths)} segments):", paths)
    clipped_paths = crop_svg(paths, square_size, square_size)
    # print(f"Original path ({len(paths)} segments):", paths)

    print("length of clipped paths: ", len(clipped_paths))
    print("length of attributes: ", len(attributes))

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
    new_coords.append(first)
    return new_coords


def clip_path_to_boundary(path, boundary, width, height, num_samples=20):
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
            coords = close_line_with_corners(coords, width, height)
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
            all_coords = close_line_with_corners(all_coords, width, height)
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


def translateSVG(input_svg, output_svg, x_shift, y_shift):
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


def getPattern(original_pattern):
    match original_pattern:
        case 'front':
            return 'preprocessed_pattern.svg'

        case 'back':
            return 'svg_file_2.svg'

        case _:
            return 'error'


def overlayDrawingOnStencil(stencil_file, user_drawing_file, size, square_size, margin_x=MARGIN, margin_y=0, filename='combined_output.svg'):
        global FILE_STEP_COUNTER
        
        translated_user_path = f"{FILE_STEP_COUNTER}_translated_for_overlay.svg"
        FILE_STEP_COUNTER += 1
        translateSVG(user_drawing_file, translated_user_path, margin_x + square_size // 2, margin_y + (margin_x * 2))

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
    overlayDrawingOnStencil(stencil, resized_pattern_name, size, square_size, margin, margin_y, combined_output_name)

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
    Combine overlapping paths into single shapes using Shapely.
    
    Args:
        paths: List of svgpathtools Path objects
        attrs: List of attribute dictionaries for each path
        tolerance: Tolerance for determining if points are close enough to be considered overlapping
    
    Returns:
        Tuple of (combined_paths, combined_attrs) 
    """
    if not paths:
        return [], []
    
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
        
        # Close the path if it's not already closed
        if len(points) > 2 and distance(points[0], points[-1]) > tolerance:
            points.append(points[0])
        
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
                
            if current_poly.intersects(poly2):
                try:
                    # Merge the polygons
                    current_poly = current_poly.union(poly2)
                    # Keep attributes of the first polygon
                    processed.add(j)
                except Exception as e:
                    print(f"Error merging polygons: {e}")
        
        result_polygons.append(current_poly)
        result_attrs.append(current_attr)
    
    # Convert back to svgpathtools paths
    combined_paths = []
    combined_attrs = []
    
    for poly, attr in zip(result_polygons, result_attrs):
        try:
            if isinstance(poly, Polygon):
                # Extract exterior coordinates
                coords = list(poly.exterior.coords)
                
                # Create line segments for the outline
                path_segments = []
                for i in range(len(coords) - 1):
                    start = complex(coords[i][0], coords[i][1])
                    end = complex(coords[i+1][0], coords[i+1][1])
                    path_segments.append(Line(start, end))
                
                combined_paths.append(Path(*path_segments))
                combined_attrs.append(attr)
                
            elif isinstance(poly, MultiPolygon):
                # Handle each polygon in the multipolygon
                for geom in poly.geoms:
                    coords = list(geom.exterior.coords)
                    path_segments = []
                    for i in range(len(coords) - 1):
                        start = complex(coords[i][0], coords[i][1])
                        end = complex(coords[i+1][0], coords[i+1][1])
                        path_segments.append(Line(start, end))
                    
                    combined_paths.append(Path(*path_segments))
                    combined_attrs.append(attr)
        except Exception as e:
            print(f"Error converting polygon to path: {e}")
    
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


def create_simple_pattern_stencils(width, height, size, stencil_1_pattern, empty_stencil_1, empty_stencil_2, pattern_type):
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


def create_symmetric_pattern_stencils(stencil_1_pattern, width, height, size, empty_stencil_1, empty_stencil_2, pattern_type):
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

    combined_simple_stencil_w_patt, combined_simple_stencil_no_patt = create_simple_pattern_stencils(width, height, size, post_cropped_pattern, empty_stencil_1, empty_stencil_2, pattern_type)
    
    # Draw lines from shapes to the edges of the stencil
    pattern_w_extended_lines = f"{FILE_STEP_COUNTER}_pattern_w_extended_lines.svg"
    FILE_STEP_COUNTER += 1
    drawExtensionLines(combined_simple_stencil_w_patt, combined_simple_stencil_no_patt, pattern_w_extended_lines, width, height, 0)

    mirrored_pattern_extended = f"{FILE_STEP_COUNTER}_mirrored_pattern_extended.svg"
    FILE_STEP_COUNTER += 1
    mirrorLines(pattern_w_extended_lines, mirrored_pattern_extended, width, height, pattern_type)

    combinePatternAndMirrorWithStencils(pattern_w_extended_lines, combined_simple_stencil_no_patt, mirrored_pattern_extended)


def create_asymmetric_pattern_stencils(stencil_1_pattern, width, height, size, empty_stencil_1, empty_stencil_2, pattern_type):
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
    translateSVG(prepped_pattern, translated_for_bottom_half, 0, -500 // 2)

    prepped_bottom_pattern = f"{FILE_STEP_COUNTER}_prepped_bottom_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropToTopHalf(translated_for_bottom_half, prepped_bottom_pattern)

    re_translated_for_bottom_half = f"{FILE_STEP_COUNTER}_post_cropped_bottom_pattern.svg"
    FILE_STEP_COUNTER += 1
    translateSVG(prepped_bottom_pattern, re_translated_for_bottom_half, 0, 500 // 2)

    # undo the crop prep once the cropping is finished
    post_cropped_pattern = f"{FILE_STEP_COUNTER}_post_cropped_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropPrep(half_of_pattern, post_cropped_pattern, -cropped_size, -45)

    # undo the crop prep for bottom half once the cropping is finished
    post_cropped_bottom_pattern = f"{FILE_STEP_COUNTER}_post_cropped_bottom_pattern.svg"
    FILE_STEP_COUNTER += 1
    cropPrep(re_translated_for_bottom_half, post_cropped_bottom_pattern, -cropped_size, -45)

    # --- for top half ---
    combined_simple_stencil_w_patt, combined_simple_stencil_no_patt = create_simple_pattern_stencils(width, height, size, post_cropped_pattern, empty_stencil_1, empty_stencil_2, pattern_type)

    pattern_w_extended_lines = f"{FILE_STEP_COUNTER}_pattern_w_extended_lines.svg"
    FILE_STEP_COUNTER += 1
    drawExtensionLines(combined_simple_stencil_w_patt, combined_simple_stencil_no_patt, pattern_w_extended_lines, width, height, 0)
    # ------

    # --- for bottom half ---
    combined_simple_stencil_w_bot_patt, _ = create_simple_pattern_stencils(width, height, size, post_cropped_bottom_pattern, empty_stencil_1, empty_stencil_2, pattern_type)

    bottom_pattern_w_extended_lines = f"{FILE_STEP_COUNTER}_bottom_pattern_w_extended_lines.svg"
    FILE_STEP_COUNTER += 1
    drawExtensionLines(combined_simple_stencil_w_bot_patt, combined_simple_stencil_no_patt, bottom_pattern_w_extended_lines, width, height, 0)
    # ------
    
    mirrored_bottom_pattern_extended = f"{FILE_STEP_COUNTER}_mirrored_pattern_extended.svg"
    FILE_STEP_COUNTER += 1
    mirrorLines(pattern_w_extended_lines, mirrored_bottom_pattern_extended, width, height, pattern_type, bottom_pattern_w_extended_lines)

    combinePatternAndMirrorWithStencils(pattern_w_extended_lines, combined_simple_stencil_no_patt, mirrored_bottom_pattern_extended)



def cropPrep(pattern, output_name, cropped_size, angle):
    global FILE_STEP_COUNTER
    
    # Step 1: Rotate the pattern 45 degrees clockwise
    rotated_path_name = f"{FILE_STEP_COUNTER}_rotated_pattern_step.svg"
    FILE_STEP_COUNTER += 1
    rotateSVG(pattern, rotated_path_name, angle)
    
    # Step 2: Translate to correct position after rotation
    translateSVG(rotated_path_name, output_name, cropped_size, cropped_size)


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


def drawExtensionLines(combined_stencil, stencil_pattern, output_name, width, height, starting_y, margin_x=MARGIN):

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

    # wsvg(pattern_paths, attributes=pattern_attrs, filename=filename, dimensions=(width, width))

    # combine overlapping paths
    combined_paths, combined_attrs = combine_overlapping_paths(pattern_paths, pattern_attrs)
    wsvg(combined_paths, attributes=combined_attrs, filename="combined_shapes.svg", dimensions=(width, width))
    
    combined_paths_w_lines = copy.deepcopy(combined_paths)
    combined_attrs_w_lines = copy.deepcopy(combined_attrs)

    # draw a line from the bottom most point to the left edge of the stencil and draw a line from the top most point to the right edge of the stencil
    for path, attr in zip(combined_paths, combined_attrs):
        # Extract the start and end points of the path
        # Find all points in the path
        points = []
        for segment in path:
            # Sample points along the segment
            for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
                pt = segment.point(t)
                points.append(pt)
            # Make sure to include the endpoint
            points.append(segment.end)

        # Find the point with minimum y-coordinate (rightmost in case of ties)
        min_y = float('inf')
        min_y_point = None
        for pt in points:
            if pt.imag < min_y or (pt.imag == min_y and pt.real > (min_y_point.real if min_y_point else -float('inf'))):
                min_y = pt.imag
                min_y_point = pt

        # Find the point with maximum y-coordinate (rightmost in case of ties)
        max_y = float('-inf')
        max_y_point = None
        for pt in points:
            if pt.imag > max_y or (pt.imag == max_y and pt.real > (max_y_point.real if max_y_point else -float('inf'))):
                max_y = pt.imag
                max_y_point = pt

        # Use these points for drawing the lines
        start = min_y_point
        end = max_y_point
        extension = 20
        # left top line start
        left_top_line_start = (margin_x + square_size // 2, margin_y)
        right_top_line_start = (left_top_line_start[0] + square_size, left_top_line_start[1])
        right_top_line_end = (right_top_line_start[0] + square_size, right_top_line_start[1])
        rightmost_point = grabRightMostPointOfPaths(path)  # `paths` should be your drawing's paths



        # Create line from the bottom most point to the left edge of the stencil
        left_line = Line(complex(left_top_line_start[0] - extension, start.imag), complex(start.real, start.imag))
        combined_paths_w_lines.append(Path(left_line))
        combined_attrs_w_lines.append({'stroke': 'red', 'stroke-width': 1, 'fill': 'none'})

        right_line = Line(complex(rightmost_point.real, rightmost_point.imag), complex(right_top_line_end[0] + extension, rightmost_point.imag))
        combined_paths_w_lines.append(Path(right_line))
        combined_attrs_w_lines.append({'stroke': 'red', 'stroke-width': 1, 'fill': 'none'})

        # Create line from the bottom most point to the left edge of the stencil
        ##left_line = Line(complex(left_top_line_start[0] - extension, start.imag), complex(start.real, start.imag))
        ##combined_paths_w_lines.append(Path(left_line))
        ##combined_attrs_w_lines.append({'stroke': 'red', 'stroke-width': 1, 'fill': 'none'})
        
        ##right_line = Line(complex(end.real, end.imag), complex(left_top_line_start[0] + square_size, end.imag))
        combined_paths_w_lines.append(Path(right_line))
        combined_attrs_w_lines.append({'stroke': 'red', 'stroke-width': 1, 'fill': 'none'})

    # combined_shapes_w_lines = "combined_shapes_w_lines.svg"
    wsvg(combined_paths_w_lines, attributes=combined_attrs_w_lines, filename=output_name, dimensions=(width, width))


def mirrorLines(pattern_w_extended_lines, output_name, width, height, pattern_type, draw_symmetric_lines_bottom=None):
    global FILE_STEP_COUNTER
    
    # mirror the lines over the y-axis
    mirrored_lines = f"{FILE_STEP_COUNTER}_mirrored_lines.svg"
    FILE_STEP_COUNTER += 1
    if pattern_type == PatternType.Symmetric:
        mirrorSVGOverYAxis(pattern_w_extended_lines, mirrored_lines, width, height)
    elif pattern_type == PatternType.Asymmetric:
        mirrorSVGOverYAxis(draw_symmetric_lines_bottom, mirrored_lines, width, height)

    paths, _ = svg2paths(pattern_w_extended_lines)
    mirrored_paths, _ = svg2paths(mirrored_lines)

    left_point = grabLeftMostPointOfPaths(mirrored_paths)
    right_point = grabRightMostPointOfPaths(paths)
    # Calculate the distance between the leftmost and rightmost points
    distance_between = abs(left_point.real - right_point.real)

    fixed_rounding_mirrored_lines = f"{FILE_STEP_COUNTER}_fixed_mirrored_lines.svg"
    FILE_STEP_COUNTER += 1
    translateSVG(mirrored_lines, fixed_rounding_mirrored_lines, -distance_between, 0)

    # translate the mirrored lines to the correct position
    translateSVG(fixed_rounding_mirrored_lines, output_name, distance_between-31, height)


def combinePatternAndMirrorWithStencils(pattern_w_extended_lines, combined_simple_stencil_no_patt, translated_mirrored_lines, output_name="final_output.svg"):
    global FILE_STEP_COUNTER
    
    # combine the mirrored lines with the original mirrored pattern
    combined_mirrored_lines = f"{FILE_STEP_COUNTER}_combined_mirrored_lines.svg"
    FILE_STEP_COUNTER += 1
    combineStencils(translated_mirrored_lines, pattern_w_extended_lines, combined_mirrored_lines)

    # combine the combined_mirrored_lines with the stencil pattern
    combineStencils(combined_mirrored_lines, combined_simple_stencil_no_patt, output_name)

    print("saved final stencil")


def createFinalHeartCutoutPatternExport(size, pattern_type, sides='onesided', line_color='black', background_color='white'):
    global FILE_STEP_COUNTER
    
    print("pattern type: ", pattern_type)
    print("sides: ", sides)

    width = size
    height = size // 2

    empty_stencil_1 = drawEmptyStencil(width, height, 0, file_name=f"{FILE_STEP_COUNTER}_stencil1.svg")
    FILE_STEP_COUNTER += 1
    empty_stencil_2 = drawEmptyStencil(width, height, height, file_name=f"{FILE_STEP_COUNTER}_stencil2.svg")
    FILE_STEP_COUNTER += 1

    if sides=='onesided':
        pre_processed_pattern = getPattern("front")

        if pattern_type == PatternType.Simple:
            print("Creating SIMPLE pattern")
            create_simple_pattern_stencils(width, height, size, pre_processed_pattern, empty_stencil_1, empty_stencil_2, pattern_type)
        
        elif pattern_type == PatternType.Symmetric:
            print("Creating SYMMETRICAL pattern")
            
            create_symmetric_pattern_stencils(pre_processed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, pattern_type)
            
        elif pattern_type == PatternType.Asymmetric:
            print("Creating A-SYMMETRICAL pattern")
            
            create_asymmetric_pattern_stencils(pre_processed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, pattern_type)

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

    if sides =='twosided':
        pre_processed_pattern_front = getPattern("front")
        pre_processed_pattern_back = getPattern("back")

        return None


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


def mainAlgorithmSvg(img, pattern_type, function='show'):

    match function:

        case 'show':
            # convert SVG to CV Image for createFinalHeartDisplay
            heartPixmap = saveSvgFileAsPixmap(img)
            heartCvImage = savePixmapToCvImage(heartPixmap)

            return createFinalHeartDisplay(heartCvImage)

        case _:
            return createFinalHeartCutoutPatternExport(1200, pattern_type)
