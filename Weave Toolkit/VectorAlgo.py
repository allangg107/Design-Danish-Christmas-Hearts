import sys
from PyQt6.QtSvg import QSvgRenderer, QSvgGenerator
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPainter, QTransform, QPixmap, QColor
from PyQt6.QtCore import QSize, QByteArray, QRectF
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon

import xml.etree.ElementTree as ET
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier, QuadraticBezier, parse_path
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon, Point

import cv2 as cv
import numpy as np
import math
import tempfile
import svgwrite
from svgwrite.container import Group
from collections import namedtuple

from ShapeMode import (
    ShapeMode
)

# VectorAlgo will be used when the user presses the "Update SVG" button

# The algorithm will be given a image, of SVG type, of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting), and user drawing
#   - this PNG that is returned is the final image once the algorithm has processesed the user drawing and weave pattern
# 2. Assembly instructions to guide the user in weaving the output image

MARGIN = 31

def pre_process_user_input(original_pattern, shape_types, width, height, square_size):
    rotated_path_name = "rotated_pattern_step.svg"
    rotateSVG(original_pattern, rotated_path_name, 45)

    # crop to the designated drawing space
    cropped_size = int((width - square_size) // 2)
    translated_path_name = "translated_pattern_step.svg"
    translateSVG(rotated_path_name, translated_path_name, -cropped_size, -cropped_size)

    paths, attributes = svg2paths(translated_path_name)

    final_output_path_name = "preprocessed_pattern.svg"
    wsvg(paths, attributes=attributes, filename=final_output_path_name, dimensions=(square_size, square_size))

    # print("pre-processed attributes: ", attributes)

    # print(f"Original path ({len(paths)} segments):", paths)
    clipped_paths = crop_svg(paths, shape_types, square_size)
    # print(f"Original path ({len(paths)} segments):", paths)

    print("length of clipped paths: ", len(clipped_paths))
    print("length of attributes: ", len(attributes))

    if len(clipped_paths) > len(attributes):
        attributes = attributes * len(clipped_paths)

    wsvg(clipped_paths, attributes=attributes, filename=final_output_path_name, dimensions=(square_size, square_size))

def ensure_closed(path, minx, miny, maxx, maxy, tol=1e-9):
    """
    Ensures that a clipped shape remains closed, especially when it touches a corner.
    If the shape is clipped at two boundaries and also touches the corner, it is closed 
    by adding two extra segments via the corner instead of a direct closing line.
    """
    if not path:
        return path
    
    print("HERE path", path)

    start_pt = path[0].start
    end_pt = path[-1].end

    # If already closed, return as is.
    if abs(start_pt - end_pt) < tol:
        return Path(*path)

    def boundary_flags(pt):
        return {
            "left": abs(pt.real - minx) < tol,
            "right": abs(pt.real - maxx) < tol,
            "bottom": abs(pt.imag - miny) < tol,
            "top": abs(pt.imag - maxy) < tol
        }

    start_flags = boundary_flags(start_pt)
    end_flags = boundary_flags(end_pt)

    # Determine if the endpoints lie on adjacent boundaries
    corner = None
    if (start_flags["left"] and end_flags["bottom"]) or (start_flags["bottom"] and end_flags["left"]):
        corner = complex(minx, miny)
    elif (start_flags["left"] and end_flags["top"]) or (start_flags["top"] and end_flags["left"]):
        corner = complex(minx, maxy)
    elif (start_flags["right"] and end_flags["bottom"]) or (start_flags["bottom"] and end_flags["right"]):
        corner = complex(maxx, miny)
    elif (start_flags["right"] and end_flags["top"]) or (start_flags["top"] and end_flags["right"]):
        corner = complex(maxx, maxy)

    if corner is not None:
        # Check if the path already touches the corner
        touches_corner = any(abs(seg.start - corner) < tol or abs(seg.end - corner) < tol for seg in path)

        new_path = list(path)
        if not touches_corner:
            # Add two segments to close properly via the corner
            new_path.append(Line(end_pt, corner))
            new_path.append(Line(corner, start_pt))
        else:
            # If the corner is already part of the path, just close normally
            new_path.append(Line(end_pt, start_pt))

        return Path(*new_path)

    # Default: close directly if not a corner case
    new_path = list(path)
    new_path.append(Line(end_pt, start_pt))
    return Path(*new_path)



def clip_path_to_boundary(path, shape_type, boundary, num_samples=20):
    """
    Clips a given path to the boundary using Shapely geometric operations.
    For hearts and circles, segments are sampled to approximate curves.
    """
    try:
        # Obtain coordinates either by sampling (for curves) or directly.
        if shape_type in [ShapeMode.Heart, ShapeMode.Circle]:
            sampled_coords = []
            for seg in path:
                for t in np.linspace(0, 1, num_samples, endpoint=False):
                    point = seg.point(t)
                    sampled_coords.append((point.real, point.imag))
            last_point = path[-1].point(1.0)
            sampled_coords.append((last_point.real, last_point.imag))
        else:
            sampled_coords = [(seg.start.real, seg.start.imag) for seg in path]
            sampled_coords.append((path[-1].end.real, path[-1].end.imag))

        if len(sampled_coords) < 2:
            print("Skipping path: Not enough coordinates.")
            return None

        path_shape = LineString(sampled_coords)
        print("path_shape", path_shape, "with", len(path_shape.coords), "coords")

        clipped_shape = path_shape.intersection(boundary)
        print("clipped_shape", clipped_shape)

        if clipped_shape.is_empty:
            print("Warning: Clipped shape is empty!")
            return None

        # Get the boundary limits to use in ensure_closed.
        minx, miny, maxx, maxy = boundary.bounds
        tol = 1e-9

        if isinstance(clipped_shape, LineString):
            coords = list(clipped_shape.coords)
            if shape_type != ShapeMode.Line:
                # For fillable shapes, ensure the coordinate list is closed.
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
            segments = [
                Line(complex(x, y), complex(x2, y2))
                for (x, y), (x2, y2) in zip(coords[:-1], coords[1:])
            ]
            if shape_type != ShapeMode.Line:
                print("TEST2")
                print(f"TESTING Path before closing ({len(path)} segments):", path)
                segments = ensure_closed(segments, minx, miny, maxx, maxy, tol)
                print(f"TESTING Path after closing ({len(segments)} segments):", segments)
            return Path(*segments)

        elif isinstance(clipped_shape, MultiLineString):
            all_coords = []
            for line in clipped_shape.geoms:
                if len(line.coords) < 2:
                    continue
                all_coords.extend(list(line.coords))
            # Simple deduplication.
            all_coords = list(dict.fromkeys(all_coords))
            if shape_type != ShapeMode.Line:
                if all_coords[0] != all_coords[-1]:
                    all_coords.append(all_coords[0])
            segments = [
                Line(complex(x, y), complex(x2, y2))
                for (x, y), (x2, y2) in zip(all_coords[:-1], all_coords[1:])
            ]
            if shape_type != ShapeMode.Line:
                print(f"Path before closing ({len(path)} segments):", path)
                print("TEST")
                segments = ensure_closed(segments, minx, miny, maxx, maxy, tol)
                print(f"Path after closing ({len(segments)} segments):", segments)
            return Path(*segments)

        print("Warning: Unexpected geometry type from intersection:", type(clipped_shape))
        return None

    except Exception as e:
        print("Error while clipping path:", e)
        return None

    


def crop_svg(paths, shape_types, square_size):
    """
    Crops all paths to fit within the given square_size.
    """
    boundary = Polygon([(0, 0), (square_size, 0), (square_size, square_size), (0, square_size)])

    # print("\nBoundary Polygon:", boundary)
    # print("Total Paths Received for Clipping:", len(paths))

    print("given paths", paths) 

    clipped_paths = []
    for path, shape_type in zip(paths, shape_types):
        clipped = clip_path_to_boundary(path, shape_type, boundary)
        if clipped:
            if isinstance(clipped, list):  # Handle MultiLineString cases
                clipped_paths.extend(clipped)
            else:
                clipped_paths.append(clipped)

    # print("Final Clipped Paths:", clipped_paths)
    
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

def rotateSVG(input_svg, output_svg, angle):
    paths, attributes = svg2paths(input_svg)

    tree = ET.parse(input_svg)
    root = tree.getroot()
    width = float(root.get("width", "500"))  # Default to 500 if missing
    height = float(root.get("height", "500"))

    # Calculate the center of the SVG
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
         dimensions=(width, height))

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
        translated_user_path = "translated_for_overlay.svg"
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
    # scale the pattern
    square_size = size // 2 // 1.5 - margin
    inner_cut_size = square_size - (margin * 2)
    resized_pattern_name = "scaled_pattern.svg"
    resizeSVG(pattern, resized_pattern_name, inner_cut_size)

    # shift the pattern right and down (overlay on stencil)
    combined_output_name = f"stencil_{stencil_number}_overlayed.svg"
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

def drawClassicStencil(width, height, starting_y, margin_x=31, line_color='black', file_name="test_1.svg"):
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

def determinePatternType(pattern_type):
    if pattern_type == 'pattern_simple':
        return 'simple'
    elif pattern_type == 'pattern_symmetrical':
        return 'symmetrical'
    elif pattern_type == 'pattern_asymmetrical':
        return 'asymmetrical'


def createFinalHeartCutoutPatternExport(size, line_start=0, sides='onesided', line_color='black', background_color='white'):
    width = size
    height = size // 2
    empty_stencil_1 = drawEmptyStencil(width, height, 0, file_name="stencil1.svg")
    empty_stencil_2 = drawEmptyStencil(width, height, height, file_name="stencil2.svg")
    if sides=='onesided':
        pattern_type = determinePatternType()

        stencil_1_pattern = getPattern("front")
        stencil_2_pattern = getPattern("back")

        if pattern_type == "simple":
            simpleStencil = drawSimpleStencil(width, height, 0, file_name="simpleStencil1.svg")
            combined_simple_stencil1 = "combined_simple_stencil1.svg"
            combineStencils(empty_stencil_1, simpleStencil, combined_simple_stencil1)
            overlayed_pattern_1 = overlayPatternOnStencil(stencil_1_pattern, combined_simple_stencil1, size, 1, pattern_type)
            simpleStencil = drawSimpleStencil(width, height, height, file_name="simpleStencil2.svg")
            combined_simple_stencil2 = "combined_simple_stencil2.svg"
            final_stencil = "thisguy.svg"
            combineStencils(empty_stencil_2, simpleStencil, final_stencil)
            combineStencils(final_stencil, overlayed_pattern_1, combined_simple_stencil2)

            convertSvgToPng(combined_simple_stencil2, size, size, "final_output.png")

    else:
        combined_classic_stencil = "combined_classic_stencil.svg"
        classic_stencil1 = drawClassicStencil(width, height, 0, file_name="classic_stencil1.svg")
        classic_stencil2 = drawClassicStencil(width, height, height, file_name="classic_stencil2.svg")
        final_stencil = "classic_final_stencil.svg"
        combined_classic_stencil_final = "combined_classic_stencil_final.svg"
        combineStencils(empty_stencil_1, classic_stencil1, combined_classic_stencil)
        combineStencils(empty_stencil_2, classic_stencil2, final_stencil)
        combineStencils(final_stencil, combined_classic_stencil, combined_classic_stencil_final)

        # if pattern 1 == symetrical:
            # stencil_1_pattern = getSymetricalPattern(1)
            # stencil_2_pattern = getSymetricalPattern(2)
        # elif asymetrical:
            # stencil_1_pattern = getAsymtricalPattern(1)
            # stencil_2_pattern = getAsymtricalPattern(2)
        # else:


        #overlayed_pattern_1 = overlayPatternOnStencil(stencil_1_pattern, empty_stencil_1, size, 1, pattern_type)
        # overlayed_pattern_2 = overlayPatternOnStencil(stencil_2_pattern, empty_stencil_2, size, 2, pattern_type)

        # combined_stencil = combineStencils(overlayed_pattern_1, overlayed_pattern_2)

        # resizeSvg(combined_stencil, user_decided_export_size)

        # return combined_stencil

    # do the same for the mirrored version
    if sides =='twosided':
        return None

def convertSvgToPng(svg_file, width, height, output_file):
    cvImage = savePixmapToCvImage(saveSvgFileAsPixmap(svg_file, QSize(height, width)))

    transparentImage = makeTrans(cvImage, [255, 255, 255])
    cv.imwrite(output_file, transparentImage)

    # # Create a mask where the background is white
    # background_color = [255, 255, 255, 255]  # White background with full opacity
    # mask = np.all(cvImage[:, :, :3] == background_color[:3], axis=2)

    # # Set alpha (transparency) to 0 where the mask is true
    # cvImage[mask] = [0, 0, 0, 0]  # Set BGRA values to transparent

    # # Save the result
    # cv.imwrite('output_image.png', cvImage)

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

def mainAlgorithmSvg(img, function = 'create', shape_attributes=[]):

    match function:
        case 'create_simple':
            pattern_type = getPattern('pattern_simple')
            createFinalHeartCutoutPatternExport(1200, pattern_type)
        case 'create_symmetrical':
            pattern_type = getPattern('pattern_asymmetrical')
            createFinalHeartCutoutPatternExport(1200, pattern_type)
        case 'create_asymmetrical':
            pattern_type = getPattern('pattern_symmetrical')
            createFinalHeartCutoutPatternExport(1200, pattern_type)
        case 'show':
            # We start with a filepath to an svg image. But, we want to give createFinalHeartDisplay a CV Image
            heartPixmap = saveSvgFileAsPixmap(img)
            heartCvImage = savePixmapToCvImage(heartPixmap)

            return createFinalHeartDisplay(heartCvImage)

        case _:
            return createFinalHeartCutoutPatternExport(1200, sides= '')
