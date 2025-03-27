import svgwrite
import xml.etree.ElementTree as ET
import copy
import cv2 as cv
import numpy as np
import math
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier, QuadraticBezier, parse_path
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon

from VectorAlgoUtils import (
    rotateSVG,
    translateSVGBy,
    resizeSVG,
    cropPrep,
    cropToTopHalf,
    crop_svg,
    mirrorSVGOverXAxis,
    mirrorSVGOverYAxis,
    mirrorSVGOverYAxisWithX,
    getDrawingSquareSize,
    getFileStepCounter,
    incrementFileStepCounter,
    setDrawingSquareSize,
    removeDuplicateLinesFromSVG

)

from PatternType import (
    PatternType
)

from SideType import (
    SideType
)

MARGIN = 31

"""Get and set MARGIN"""

def getMargin():
    return MARGIN

def setMargin(value):
    global MARGIN
    MARGIN = value


def combineStencils(first_stencil, second_stencil, filename='combined.svg'):
    """Combines two SVG files together into one"""
    paths1, attributes1 = svg2paths(first_stencil)
    paths2, attributes2 = svg2paths(second_stencil)

    combined_paths = paths1 + paths2
    combined_attributes = attributes1 + attributes2

    wsvg(combined_paths, attributes=combined_attributes, filename=filename)


"""Overlaying of SVG files"""

def overlayDrawingOnStencil(stencil_file, user_drawing_file, size, square_size, pattern_type, margin_x=MARGIN, margin_y=0, filename='combined_output.svg'):

        translated_user_path = f"{getFileStepCounter()}_translated_for_overlay.svg"
        incrementFileStepCounter()

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
    # scale the pattern
    square_size = size // 2 // 1.5 - margin
    inner_cut_size = square_size - (margin * 2)
    resize_size = inner_cut_size

    resized_pattern_name = f"{getFileStepCounter()}_scaled_pattern.svg"
    incrementFileStepCounter()
    resizeSVG(pattern, resized_pattern_name, resize_size)

    # shift the pattern right and down (overlay on stencil)
    combined_output_name = f"{getFileStepCounter()}_stencil_{stencil_number}_overlayed.svg"
    incrementFileStepCounter()
    margin_y = 0 if stencil_number == 1 else size // 2
    overlayDrawingOnStencil(stencil, resized_pattern_name, size, square_size, pattern_type, margin, margin_y, combined_output_name)

    return combined_output_name


"""Stencils"""

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


def drawInnerCutLines(width, height, starting_y, margin_x=MARGIN, line_color='black', file_name="allans_test.svg"):
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


def createClassicInnerCuts(width, height, starting_y, n_lines, margin_x=MARGIN, line_color='black'):
    #define the square size
    square_size = (height // 1.5) - margin_x
    margin_y = margin_x + starting_y

    left_top_line_start = (margin_x + square_size // 2, margin_y)
    left_top_line_end = (left_top_line_start[0] + square_size, left_top_line_start[1])
    left_bottom_line_start = (left_top_line_start[0], left_top_line_start[1] + square_size)
    left_bottom_line_end = (left_bottom_line_start[0] + square_size, left_bottom_line_start[1])

    #draw the right square
    right_top_line_start = left_top_line_end
    right_top_line_end = (right_top_line_start[0] + square_size, right_top_line_start[1])
    right_bottom_line_start = left_bottom_line_end
    right_bottom_line_end = (right_bottom_line_start[0] + square_size, right_bottom_line_start[1])

    offset = square_size / (n_lines + 1)

    paths = []
    attributes = []

    new_paths = [Line(
        start=complex(left_top_line_start[0], left_top_line_start[1] + offset * (i + 1)),
        end=complex(right_top_line_end[0], left_top_line_start[1] + offset * (i + 1))
    ) for i in range(n_lines)]
    new_attributes = [{'stroke': line_color, 'stroke-width': 3, 'fill': 'none'} for _ in range(n_lines)]
    paths.extend(new_paths)
    attributes.extend(new_attributes)

    return paths, attributes


def create_and_combine_stencils_onesided(width, height, size, stencil_1_pattern, empty_stencil_1, empty_stencil_2, pattern_type, n_lines= -1, is_blank=False):
    # Create both simple stencils
    stencil_1_inner_cuts = drawInnerCutLines(width, height, 0, file_name=f"{getFileStepCounter()}_simpleStencil1.svg")
    incrementFileStepCounter()
    stencil_2_inner_cuts = drawInnerCutLines(width, height, height, file_name=f"{getFileStepCounter()}_simpleStencil2.svg")
    incrementFileStepCounter()

    # Combine all stencils first
    stencil_1_combined = f"{getFileStepCounter()}stencil_1_combined.svg"
    incrementFileStepCounter()
    combineStencils(empty_stencil_1, stencil_1_inner_cuts, stencil_1_combined)
    
    stencil_2_combined = f"{getFileStepCounter()}stencil_2_combined.svg"
    incrementFileStepCounter()
    combineStencils(empty_stencil_2, stencil_2_inner_cuts, stencil_2_combined)
    
    if is_blank:
        final_output = f"{getFileStepCounter()}_final_output.svg"
        incrementFileStepCounter()
        combineStencils(stencil_1_combined, stencil_2_combined, final_output)
        return final_output, final_output
    
    combined_stencils = f"{getFileStepCounter()}_combined_stencils.svg"
    incrementFileStepCounter()
    combineStencils(stencil_1_combined, stencil_2_combined, combined_stencils)

    # rotate the pattern 90 degrees counter-clockwise
    rotated_path_name = f"{getFileStepCounter()}_fixed_pattern_rotation.svg"
    incrementFileStepCounter()
    rotateSVG(stencil_1_pattern, rotated_path_name, -90)

    # Now overlay the pattern on the combined stencil
    overlayed_pattern = overlayPatternOnStencil(rotated_path_name, combined_stencils, size, 1, pattern_type)
    return overlayed_pattern, combined_stencils


""" Helper functions for symmetric and asymmetric cases"""

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
    rotated_path_name = f"{getFileStepCounter()}_rotated_pattern_step.svg"
    incrementFileStepCounter()
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

    global MARGIN

    # mirror the lines over the y-axis
    mirrored_lines = f"{getFileStepCounter()}_mirrored_lines.svg"
    incrementFileStepCounter()
    mirrorSVGOverYAxis(pattern_w_extended_lines, mirrored_lines, width, height)

    paths, _ = svg2paths(pattern_w_extended_lines)
    mirrored_paths, _ = svg2paths(mirrored_lines)

    left_point = grabLeftMostPointOfPaths(mirrored_paths)
    right_point = grabRightMostPointOfPaths(paths)
    # Calculate the distance between the leftmost and rightmost points
    distance_between = abs(left_point.real - right_point.real)

    fixed_rounding_mirrored_lines = f"{getFileStepCounter()}_fixed_mirrored_lines.svg"
    incrementFileStepCounter()
    translateSVGBy(mirrored_lines, fixed_rounding_mirrored_lines, -distance_between, 0)

    # translate the mirrored lines to the correct position
    translateSVGBy(fixed_rounding_mirrored_lines, output_name, distance_between - MARGIN, height)


def combinePatternAndMirrorWithStencils(pattern_w_extended_lines, combined_simple_stencil_no_patt, translated_mirrored_lines, output_name=None):
    
    # combine the mirrored lines with the original mirrored pattern
    combined_mirrored_lines = f"{getFileStepCounter()}_combined_mirrored_lines.svg"
    incrementFileStepCounter()
    combineStencils(translated_mirrored_lines, pattern_w_extended_lines, combined_mirrored_lines)

    # combine the combined_mirrored_lines with the stencil pattern
    if output_name is None:
        output_name = f"{getFileStepCounter()}_final_output.svg"
        incrementFileStepCounter()
        
    combineStencils(combined_mirrored_lines, combined_simple_stencil_no_patt, output_name)

    print("saved final stencil")


def create_simple_pattern_stencils(stencil_1_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type, is_blank):

    combined_simple_stencil_w_patt, combined_simple_stencil_no_patt = create_and_combine_stencils_onesided(width, height, size, stencil_1_pattern, empty_stencil_1, empty_stencil_2, pattern_type, is_blank=is_blank)

    if is_blank:
        return

    if side_type == SideType.TwoSided:
        processed_pattern = f"{getFileStepCounter()}_translated_pattern.svg"
        incrementFileStepCounter()
        processed_pattern = removeDuplicateLinesFromSVG(combined_simple_stencil_w_patt, combined_simple_stencil_no_patt)

        mirrored_pattern = f"{getFileStepCounter()}_mirrored_pattern.svg"
        incrementFileStepCounter()
        mirrorLines(processed_pattern, mirrored_pattern, width, 0, pattern_type)
        combined_patt_and_mirror = f"{getFileStepCounter()}_combined_patt_and_mirror.svg"
        incrementFileStepCounter()
        combineStencils(processed_pattern, mirrored_pattern, combined_patt_and_mirror)

        combinePatternAndMirrorWithStencils(processed_pattern, combined_simple_stencil_no_patt, mirrored_pattern)



def fitClassicCuts(classic_cuts, stencil_pattern, output_name, width, height, size):
    """
    Overlay the pattern onto the classic cuts.
    """
    classic_cut_paths, _ = svg2paths(classic_cuts)
    classic_cut_lines = []
    for path in classic_cut_paths:
        for segment in path:
            if isinstance(segment, Line):
                classic_cut_lines.append(segment)

    # Load the stencil pattern
    stencil_paths, stencil_attributes = svg2paths(stencil_pattern)

    # Convert classic cuts to svgpathtools Path objects
    classic_cut_paths = [Path(line) for line in classic_cut_lines]

    # Combine classic cuts with stencil pattern
    combined_paths = classic_cut_paths + stencil_paths
    combined_attributes = [{'stroke': 'black', 'stroke-width': 1, 'fill': 'none'}] * len(classic_cut_paths) + stencil_attributes

    # Save the combined result
    wsvg(combined_paths, attributes=combined_attributes, filename=output_name, dimensions=(width, height))


def snapShapeToClassicCuts(classic_cuts, begin_point, end_point, width, height):
    print("snapShapesToClassicCuts")
    print(classic_cuts)

    # Process classic_cuts as a list of [x1, y1, x2, y2] coordinates
    lines = []
    for line_coords in classic_cuts:
        line = LineString([(line_coords[0], line_coords[1]), (line_coords[2], line_coords[3])])
        lines.append(line)

    # Find all intersection points
    intersection_points = []

    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j:  # Skip duplicate checks and self-intersection
                continue

            # Check for intersection
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                if hasattr(intersection, 'x') and hasattr(intersection, 'y'):
                    intersection_points.append(complex(intersection.x, intersection.y))

    print(f"Found {len(intersection_points)} intersection points")

    # Find closest intersection points to begin_point and end_point
    if len(intersection_points) > 0:
        # Convert QPoint to complex before comparing
        begin_complex = complex(begin_point.x(), begin_point.y())
        end_complex = complex(end_point.x(), end_point.y())

        # Now use the complex versions for distance calculations
        closest_to_begin = min(intersection_points,
                               key=lambda p: abs(p - begin_complex))
        closest_to_end = min(intersection_points,
                             key=lambda p: abs(p - end_complex))

        # Convert back to QPoint for return
        from PyQt6.QtCore import QPoint
        return QPoint(int(closest_to_begin.real), int(closest_to_begin.imag)), \
               QPoint(int(closest_to_end.real), int(closest_to_end.imag))

    return begin_point, end_point


def findAllNonIntersectedShapes(pattern, classic_cuts, min_intersection_length=5.0):
    """
    Find all shapes in the pattern SVG that are not intersected by the horizontal classic cuts,
    or have minimal intersection below the threshold.
    
    Args:
        pattern: Path to the pattern SVG file
        classic_cuts: Path to the classic cuts SVG file
        min_intersection_length: Minimum length of intersection to be considered as intersected
        
    Returns:
        List of non-intersected path indices from the pattern
    """
    # Load pattern and classic cuts
    pattern_paths, pattern_attrs = svg2paths(pattern)
    cuts_paths, _ = svg2paths(classic_cuts)
    
    # Extract all lines from classic cuts (assuming they are all horizontal)
    horizontal_cuts = []
    for path in cuts_paths:
        for segment in path:
            if isinstance(segment, Line):
                horizontal_cuts.append(LineString([
                    (segment.start.real, segment.start.imag),
                    (segment.end.real, segment.end.imag)
                ]))
    
    # Find non-intersected shapes
    non_intersected_indices = []
    
    for i, path in enumerate(pattern_paths):
        # Convert path to Shapely geometry
        points = []
        for segment in path:
            # Sample points along the segment
            for t in np.linspace(0, 1, 10):
                pt = segment.point(t)
                points.append((pt.real, pt.imag))
        
        # Skip paths with insufficient points
        if len(points) < 2:
            continue
        
        # Create appropriate geometry
        shape = None
        if path.isclosed():
            # It's a closed shape, create polygon
            if len(points) >= 3:
                try:
                    shape = Polygon(points)
                    if not shape.is_valid:
                        shape = shape.buffer(0)  # Fix self-intersections
                except Exception as e:
                    print(f"Error creating polygon: {e}")
                    continue
        else:
            # It's an open shape, create linestring
            shape = LineString(points)
        
        if shape is None:
            continue
        
        # Check intersection with all horizontal cuts
        is_intersected = False
        for cut in horizontal_cuts:
            if shape.intersects(cut):
                intersection = shape.intersection(cut)
                
                # Calculate length of intersection
                if hasattr(intersection, 'length'):
                    intersection_length = intersection.length
                elif hasattr(intersection, 'geoms'):
                    # MultiLineString or GeometryCollection
                    intersection_length = sum(geom.length for geom in intersection.geoms 
                                            if hasattr(geom, 'length'))
                else:
                    # Point intersection
                    intersection_length = 0
                
                if intersection_length >= min_intersection_length:
                    is_intersected = True
                    break
        
        if not is_intersected:
            non_intersected_indices.append(i)
    
    return [pattern_paths[i] for i in non_intersected_indices], [pattern_attrs[i] for i in non_intersected_indices]


def splitShapesInHalf(shapes, classic_cuts, top_halves_output_name, bottom_halves_output_name):
    """
    Split shapes in half based on classic cuts.
    
    Args:
        shapes: Path to the shapes SVG file
        classic_cuts: Path to the classic cuts SVG file
        top_halves_output_name: Output path for the top halves
        bottom_halves_output_name: Output path for the bottom halves
    """

    combined_stencils = f"{getFileStepCounter()}_PATTERN_WITH_STENCIL_CHECKPOINT.svg"
    incrementFileStepCounter()
    combineStencils(shapes, classic_cuts, combined_stencils)
    
    # Load shapes and classic cuts from SVG files
    shape_paths, shape_attrs = svg2paths(shapes)
    cut_paths, _ = svg2paths(classic_cuts)

    # Assume classic_cuts contains exactly 3 horizontal lines
    horizontal_cuts = [segment.start.imag for path in cut_paths for segment in path if isinstance(segment, Line)]

    # Sort the horizontal cuts by their y-coordinate
    horizontal_cuts = sorted(horizontal_cuts)

    top_half_paths = []
    top_half_attrs = []
    bottom_half_paths = []
    bottom_half_attrs = []

    # Process each shape
    for i, path in enumerate(shape_paths):
        top_segments = []
        bottom_segments = []

        for segment in path:
            if isinstance(segment, Line):
                start_y = segment.start.imag
                end_y = segment.end.imag

                # Check if the segment crosses any horizontal cut
                for cut_y in horizontal_cuts:
                    if (start_y < cut_y and end_y > cut_y) or (start_y > cut_y and end_y < cut_y):
                        # Calculate the intersection point
                        t = (cut_y - segment.start.imag) / (segment.end.imag - segment.start.imag)
                        intersection_x = segment.start.real + t * (segment.end.real - segment.start.real)
                        intersection = complex(intersection_x, cut_y)

                        # Split the segment into two parts
                        if start_y < cut_y:
                            top_segments.append(Line(segment.start, intersection))
                            bottom_segments.append(Line(intersection, segment.end))
                        else:
                            bottom_segments.append(Line(segment.start, intersection))
                            top_segments.append(Line(intersection, segment.end))
                        break
                else:
                    # If no intersection, add the segment to the appropriate half
                    if start_y < min(horizontal_cuts) and end_y < min(horizontal_cuts):
                        top_segments.append(segment)
                    elif start_y > max(horizontal_cuts) and end_y > max(horizontal_cuts):
                        bottom_segments.append(segment)

        # Add the split segments to the respective paths
        if top_segments:
            top_half_paths.append(Path(*top_segments))
            top_half_attrs.append(shape_attrs[i])
        if bottom_segments:
            bottom_half_paths.append(Path(*bottom_segments))
            bottom_half_attrs.append(shape_attrs[i])

    # Save the top halves to the output file
    wsvg(top_half_paths, attributes=top_half_attrs, filename=top_halves_output_name)

    # Save the bottom halves to the output file
    wsvg(bottom_half_paths, attributes=bottom_half_attrs, filename=bottom_halves_output_name)
    
    # # Load shapes and cuts from SVG files
    # shape_paths, shape_attrs = svg2paths(shapes)
    # cut_paths, _ = svg2paths(classic_cuts)
    
    # # Extract horizontal cut lines from classic cuts
    # horizontal_cuts = []
    # for path in cut_paths:
    #     for segment in path:
    #         if isinstance(segment, Line):
    #             # Check if the line is horizontal (same y-coordinate at start and end)
    #             if abs(segment.start.imag - segment.end.imag) < 1e-6:
    #                 horizontal_cuts.append(LineString([
    #                     (segment.start.real, segment.start.imag),
    #                     (segment.end.real, segment.end.imag)
    #                 ]))
    
    # top_half_paths = []
    # top_half_attrs = []
    # bottom_half_paths = []
    # bottom_half_attrs = []
    
    # # Process each shape
    # for i, path in enumerate(shape_paths):
    #     # Convert path to Shapely geometry
    #     points = []
    #     for segment in path:
    #         # Sample points along the segment
    #         for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
    #             pt = segment.point(t)
    #             points.append((pt.real, pt.imag))
        
    #     # Skip paths with insufficient points
    #     if len(points) < 3:
    #         continue
        
    #     # Create a polygon from the points
    #     try:
    #         if path.isclosed():
    #             shape = Polygon(points)
    #             if not shape.is_valid:
    #                 shape = shape.buffer(0)  # Fix self-intersections
    #         else:
    #             shape = LineString(points)
    #     except Exception as e:
    #         print(f"Error creating geometry: {e}")
    #         continue
            
    #     # Check for intersections with each horizontal cut
    #     shape_processed = False
        
    #     for cut in horizontal_cuts:
    #         if shape.intersects(cut):
    #             cut_y = cut.coords[0][1]  # Y-coordinate of the cut line
                
    #             try:
    #                 # Split the shape by the cut line
    #                 # Create boxes that extend from the cut line to far above/below
    #                 minx, miny, maxx, maxy = shape.bounds
                    
    #                 # Box for top half (extends up from the cut line)
    #                 top_box = Polygon([
    #                     (minx - 10, cut_y), (maxx + 10, cut_y),
    #                     (maxx + 10, miny - 100), (minx - 10, miny - 100)
    #                 ])
                    
    #                 # Box for bottom half (extends down from the cut line)
    #                 bottom_box = Polygon([
    #                     (minx - 10, cut_y), (maxx + 10, cut_y),
    #                     (maxx + 10, maxy + 100), (minx - 10, maxy + 100)
    #                 ])
                    
    #                 # Get the top and bottom parts by intersection
    #                 top_part = shape.intersection(bottom_box)  # Top part is below the cut line
    #                 bottom_part = shape.intersection(top_box)  # Bottom part is above the cut line
                    
    #                 # Convert results to paths and add to respective lists
    #                 if not top_part.is_empty:
    #                     top_segments = []
                        
    #                     if isinstance(top_part, Polygon):
    #                         # Get exterior coordinates
    #                         coords = list(top_part.exterior.coords)
    #                         for i in range(len(coords) - 1):
    #                             start = complex(coords[i][0], coords[i][1])
    #                             end = complex(coords[i+1][0], coords[i+1][1])
    #                             top_segments.append(Line(start, end))
                                
    #                     elif isinstance(top_part, LineString):
    #                         coords = list(top_part.coords)
    #                         for i in range(len(coords) - 1):
    #                             start = complex(coords[i][0], coords[i][1])
    #                             end = complex(coords[i+1][0], coords[i+1][1])
    #                             top_segments.append(Line(start, end))
                                
    #                     elif hasattr(top_part, 'geoms'):
    #                         # Handle MultiPolygon or GeometryCollection
    #                         for geom in top_part.geoms:
    #                             if isinstance(geom, Polygon):
    #                                 coords = list(geom.exterior.coords)
    #                                 for i in range(len(coords) - 1):
    #                                     start = complex(coords[i][0], coords[i][1])
    #                                     end = complex(coords[i+1][0], coords[i+1][1])
    #                                     top_segments.append(Line(start, end))
    #                             elif isinstance(geom, LineString):
    #                                 coords = list(geom.coords)
    #                                 for i in range(len(coords) - 1):
    #                                     start = complex(coords[i][0], coords[i][1])
    #                                     end = complex(coords[i+1][0], coords[i+1][1])
    #                                     top_segments.append(Line(start, end))
                        
    #                     if top_segments:
    #                         top_half_paths.append(Path(*top_segments))
    #                         top_half_attrs.append(shape_attrs[i])
                    
    #                 if not bottom_part.is_empty:
    #                     bottom_segments = []
                        
    #                     if isinstance(bottom_part, Polygon):
    #                         coords = list(bottom_part.exterior.coords)
    #                         for i in range(len(coords) - 1):
    #                             start = complex(coords[i][0], coords[i][1])
    #                             end = complex(coords[i+1][0], coords[i+1][1])
    #                             bottom_segments.append(Line(start, end))
                                
    #                     elif isinstance(bottom_part, LineString):
    #                         coords = list(bottom_part.coords)
    #                         for i in range(len(coords) - 1):
    #                             start = complex(coords[i][0], coords[i][1])
    #                             end = complex(coords[i+1][0], coords[i+1][1])
    #                             bottom_segments.append(Line(start, end))
                                
    #                     elif hasattr(bottom_part, 'geoms'):
    #                         for geom in bottom_part.geoms:
    #                             if isinstance(geom, Polygon):
    #                                 coords = list(geom.exterior.coords)
    #                                 for i in range(len(coords) - 1):
    #                                     start = complex(coords[i][0], coords[i][1])
    #                                     end = complex(coords[i+1][0], coords[i+1][1])
    #                                     bottom_segments.append(Line(start, end))
    #                             elif isinstance(geom, LineString):
    #                                 coords = list(geom.coords)
    #                                 for i in range(len(coords) - 1):
    #                                     start = complex(coords[i][0], coords[i][1])
    #                                     end = complex(coords[i+1][0], coords[i+1][1])
    #                                     bottom_segments.append(Line(start, end))
                        
    #                     if bottom_segments:
    #                         bottom_half_paths.append(Path(*bottom_segments))
    #                         bottom_half_attrs.append(shape_attrs[i])
                    
    #                 shape_processed = True
    #                 break
                    
    #             except Exception as e:
    #                 print(f"Error splitting shape: {e}")
    #                 continue
        
    #     # If no intersection found, add the shape to both top and bottom halves
    #     if not shape_processed:
    #         top_half_paths.append(path)
    #         top_half_attrs.append(shape_attrs[i])
    #         bottom_half_paths.append(path)
    #         bottom_half_attrs.append(shape_attrs[i])
    
    # # Save the top halves to the output file
    # wsvg(top_half_paths, attributes=top_half_attrs, filename=top_halves_output_name)
    
    # # Save the bottom halves to the output file
    # wsvg(bottom_half_paths, attributes=bottom_half_attrs, filename=bottom_halves_output_name)


def removeIntersectingPortions(paths, classic_cuts):
    """
    Remove intersecting portions of paths based on classic cuts.
    
    Args:
        paths: List of svgpathtools Path objects
        classic_cuts: List of classic cut lines
    
    Returns:
        List of paths with intersecting portions removed
    """
    # Convert classic cuts to Shapely geometries
    classic_cut_lines = [LineString([(line.start.real, line.start.imag), (line.end.real, line.end.imag)]) for line in classic_cuts]

    # Process each path and remove intersections
    new_paths = []
    for path in paths:
        new_segments = []
        for segment in path:
            if isinstance(segment, Line):
                line = LineString([(segment.start.real, segment.start.imag), (segment.end.real, segment.end.imag)])
                for cut in classic_cut_lines:
                    if line.intersects(cut):
                        intersection = line.intersection(cut)
                        if hasattr(intersection, 'length') and intersection.length > 0:
                            # Remove intersecting portion
                            new_segments.append(Line(segment.start, complex(intersection.x, intersection.y)))
                            break
                else:
                    new_segments.append(segment)

        new_paths.append(Path(*new_segments))

    return new_paths


"""Create Non-Simple stencils"""

def create_classic_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, pattern_type, n_lines, is_blank):

    # 1. create the classic inner cuts
    stencil_1_classic_cuts_paths, stencil_1_classic_cuts_attr = createClassicInnerCuts(width, height, 0, n_lines)
    stencil_2_classic_cuts_paths, stencil_2_classic_cuts_attr = createClassicInnerCuts(width, height, height, n_lines)

    stencil_1_classic_cuts = f"{getFileStepCounter()}_stencil_1_classic_cuts.svg"
    incrementFileStepCounter()
    stencil_2_classic_cuts = f"{getFileStepCounter()}_stencil_2_classic_cuts.svg"
    incrementFileStepCounter()

    wsvg(stencil_1_classic_cuts_paths, attributes=stencil_1_classic_cuts_attr, filename=stencil_1_classic_cuts, dimensions=(width, width))
    wsvg(stencil_2_classic_cuts_paths, attributes=stencil_2_classic_cuts_attr, filename=stencil_2_classic_cuts, dimensions=(width, width))

    if is_blank:
        combineStencils(stencil_1_classic_cuts, empty_stencil_1, stencil_1_classic_cuts)
        combineStencils(stencil_2_classic_cuts, empty_stencil_2, stencil_2_classic_cuts)
        final_output = f"{getFileStepCounter()}_final_output.svg"
        incrementFileStepCounter()
        combineStencils(stencil_1_classic_cuts, stencil_2_classic_cuts, final_output)
        return

    # a. rotate -90
    rotated_path_name = f"{getFileStepCounter()}_fixed_pattern_rotation.svg"
    incrementFileStepCounter()
    rotateSVG(preprocessed_pattern, rotated_path_name, -90)
    
    # b. re-size to square_size
    global MARGIN
    square_size = (height // 1.5) - MARGIN
    resize_size = square_size
    resized_pattern_name = f"{getFileStepCounter()}_scaled_pattern.svg"
    incrementFileStepCounter()
    resizeSVG(rotated_path_name, resized_pattern_name, resize_size)
    
    # c. translate to the classic line position
    offset = square_size / (n_lines + 1)
    x_multi = 4
    y_multi = 3

    x_shift = MARGIN * x_multi + square_size // 2
    x_shift = x_shift - offset
    
    y_shift = (MARGIN * y_multi)
    y_shift = y_shift - MARGIN * 2

    translated_user_path = f"{getFileStepCounter()}_translated_for_overlay.svg"
    incrementFileStepCounter()
    translateSVGBy(resized_pattern_name, translated_user_path, x_shift, y_shift)

    # 0. set fill to none
    paths, attrs = svg2paths(translated_user_path)
    unfilled_paths, unfilled_attrs = set_fill_to_none(paths, attrs)
    wsvg(unfilled_paths, attributes=unfilled_attrs, filename=translated_user_path, dimensions=(width, width))
    pattern_paths, pattern_attrs = svg2paths(translated_user_path)
    unfilled_pattern_paths, unfilled_pattern_attrs = set_fill_to_none(pattern_paths, pattern_attrs)

    unfilled_pattern = f"{getFileStepCounter()}_unfilled_pattern.svg"
    incrementFileStepCounter()
    wsvg(unfilled_pattern_paths, attributes=unfilled_pattern_attrs, filename=unfilled_pattern, dimensions=(width, width))

    up_shape_paths, up_shape_attr = findAllNonIntersectedShapes(unfilled_pattern, stencil_1_classic_cuts)
    print(up_shape_paths)

    top_halves = "does not exist yet"
    bottom_halves = "does not exist yet"

    if up_shape_paths == []:
        top_halves = f"{getFileStepCounter()}_top_halves.svg"
        incrementFileStepCounter()
        bottom_halves = f"{getFileStepCounter()}_bottom_halves.svg"
        incrementFileStepCounter()
        splitShapesInHalf(unfilled_pattern, stencil_1_classic_cuts, top_halves, bottom_halves)

    elif up_shape_paths:
        up_shapes = f"{getFileStepCounter()}_up_shapes.svg"
        incrementFileStepCounter()
        wsvg(up_shape_paths, attributes=up_shape_attr, filename=up_shapes, dimensions=(width, width))

        down_shapes = f"{getFileStepCounter()}_down_shapes.svg"
        incrementFileStepCounter()
        down_shapes_paths = removeDuplicateLinesFromSVG(unfilled_pattern, up_shapes, down_shapes)

        translated_up_shapes = f"{getFileStepCounter()}_translated_up_shapes.svg"
        incrementFileStepCounter()
        translateSVGBy(up_shapes, translated_up_shapes, -MARGIN * 1.5, MARGIN * 1.5)

        left_top_line_end = MARGIN + square_size // 2 + square_size
        mirrored_up_shapes = f"{getFileStepCounter()}_mirrored_up_shapes.svg"
        incrementFileStepCounter()
        mirrorSVGOverYAxisWithX(translated_up_shapes, mirrored_up_shapes, width, height, left_top_line_end)

        combined_shapes = f"{getFileStepCounter()}_combined_shapes.svg"
        incrementFileStepCounter()
        if down_shapes_paths is None:
            combined_shapes = mirrored_up_shapes
        else:
            combineStencils(mirrored_up_shapes, down_shapes, combined_shapes)

        top_halves = f"{getFileStepCounter()}_top_halves.svg"
        incrementFileStepCounter()
        bottom_halves = f"{getFileStepCounter()}_bottom_halves.svg"
        incrementFileStepCounter()

        splitShapesInHalf(combined_shapes, stencil_1_classic_cuts, top_halves, bottom_halves)
        




    # top_halves_with_cuts = f"{getFileStepCounter()}_top_halves_with_cuts.svg"
    # incrementFileStepCounter()
    # combineStencils(top_halves, stencil_1_classic_cuts, top_halves_with_cuts)

    # # mirror top_halves across y_axis
    # mirrored_bottom_halves = f"{getFileStepCounter()}_mirrored_bottom_halves.svg"
    # incrementFileStepCounter()
    # mirrorSVGOverYAxisWithX(bottom_halves, mirrored_bottom_halves, width, height, left_top_line_end)

    # translated_bottom_halves = f"{getFileStepCounter()}_translated_bottom_halves.svg"
    # incrementFileStepCounter()
    # translateSVGBy(mirrored_bottom_halves, translated_bottom_halves, 0, height)
    
    # bottom_halves_with_cuts = f"{getFileStepCounter()}_top_halves_with_cuts.svg"
    # incrementFileStepCounter()
    # combineStencils(translated_bottom_halves, stencil_2_classic_cuts, bottom_halves_with_cuts)




    # top_with_removed_intersections = f"{getFileStepCounter()}_top_with_removed_intersections.svg"
    # incrementFileStepCounter()
    # removeIntersectingPortions(top_halves_with_cuts, top_with_removed_intersections)

    # bottom_with_removed_intersections = f"{getFileStepCounter()}_bottom_with_removed_intersections.svg"
    # incrementFileStepCounter()
    # removeIntersectingPortions(bottom_halves_with_cuts, bottom_with_removed_intersections)

    # final_output = f"{getFileStepCounter()}_final_output.svg"
    # incrementFileStepCounter()
    # combineStencils(top_with_removed_intersections, bottom_with_removed_intersections, final_output)


def create_symmetric_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type):

    cropped_size = int((500 - getDrawingSquareSize()) // 2)

    prepped_pattern = f"{getFileStepCounter()}_prepped_pattern.svg"
    incrementFileStepCounter()
    cropPrep(preprocessed_pattern, prepped_pattern, cropped_size, 45)

    half_of_pattern = f"{getFileStepCounter()}_half_of_pattern.svg"
    incrementFileStepCounter()
    cropToTopHalf(prepped_pattern, half_of_pattern)

    # undo the crop prep once the cropping is finished
    post_cropped_pattern = f"{getFileStepCounter()}_post_cropped_pattern.svg"
    incrementFileStepCounter()
    cropPrep(half_of_pattern, post_cropped_pattern, -cropped_size, -45)

    combined_simple_stencil_w_patt, combined_simple_stencil_no_patt = create_and_combine_stencils_onesided(width, height, size, post_cropped_pattern, empty_stencil_1, empty_stencil_2, pattern_type)

    # rotate the pattern, grab the 2 points on the line of symmetry, and then rotate it back (including the points we grabbed)

    # Draw lines from shapes to the edges of the stencil
    pattern_w_extended_lines = f"{getFileStepCounter()}_pattern_w_extended_lines.svg"
    incrementFileStepCounter()
    drawExtensionLines(combined_simple_stencil_w_patt, combined_simple_stencil_no_patt, pattern_w_extended_lines, side_type, width, height, 0)

    mirrored_pattern_extended = f"{getFileStepCounter()}_mirrored_pattern_extended.svg"
    incrementFileStepCounter()
    if side_type == SideType.OneSided:
        mirrorLines(pattern_w_extended_lines, mirrored_pattern_extended, width, height, pattern_type)
        combinePatternAndMirrorWithStencils(pattern_w_extended_lines, combined_simple_stencil_no_patt, mirrored_pattern_extended)

    elif side_type == SideType.TwoSided:
        mirrorLines(pattern_w_extended_lines, mirrored_pattern_extended, width, 0, pattern_type)
        combined_patt_and_mirror = f"{getFileStepCounter()}_combined_patt_and_mirror.svg"
        incrementFileStepCounter()
        combineStencils(pattern_w_extended_lines, mirrored_pattern_extended, combined_patt_and_mirror)
        # create copy of the combined pattern and mirror
        paths, attributes = svg2paths(combined_patt_and_mirror)
        combined_patt_and_mirror_copy = f"{getFileStepCounter()}_combined_patt_and_mirror_copy.svg"
        incrementFileStepCounter()
        paths_copy = copy.deepcopy(paths)
        attributes_copy = copy.deepcopy(attributes)
        wsvg(paths_copy, attributes=attributes_copy, filename=combined_patt_and_mirror_copy, dimensions=(width, height))
        translateSVGBy(combined_patt_and_mirror_copy, combined_patt_and_mirror_copy, 0, height)

        combinePatternAndMirrorWithStencils(combined_patt_and_mirror, combined_simple_stencil_no_patt, combined_patt_and_mirror_copy)


def create_asymmetric_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type):

    cropped_size = int((500 - getDrawingSquareSize()) // 2)

    prepped_pattern = f"{getFileStepCounter()}_prepped_pattern.svg"
    incrementFileStepCounter()
    cropPrep(preprocessed_pattern, prepped_pattern, cropped_size, 45)

    half_of_pattern = f"{getFileStepCounter()}_half_of_pattern.svg"
    getFileStepCounter()
    cropToTopHalf(prepped_pattern, half_of_pattern)

    translated_for_bottom_half = f"{getFileStepCounter()}_translated_for_bottom_half.svg"
    incrementFileStepCounter()
    translateSVGBy(prepped_pattern, translated_for_bottom_half, 0, -500 // 2)

    prepped_bottom_pattern = f"{getFileStepCounter()}_prepped_bottom_pattern.svg"
    incrementFileStepCounter()
    cropToTopHalf(translated_for_bottom_half, prepped_bottom_pattern)

    re_translated_for_bottom_half = f"{getFileStepCounter()}_post_cropped_bottom_pattern.svg"
    incrementFileStepCounter()
    translateSVGBy(prepped_bottom_pattern, re_translated_for_bottom_half, 0, 500 // 2)

    # undo the crop prep once the cropping is finished
    post_cropped_pattern = f"{getFileStepCounter()}_post_cropped_pattern.svg"
    incrementFileStepCounter()
    cropPrep(half_of_pattern, post_cropped_pattern, -cropped_size, -45)

    # undo the crop prep for bottom half once the cropping is finished
    post_cropped_bottom_pattern = f"{getFileStepCounter()}_post_cropped_bottom_pattern.svg"
    incrementFileStepCounter()
    cropPrep(re_translated_for_bottom_half, post_cropped_bottom_pattern, -cropped_size, -45)

    # --- for top half ---
    combined_simple_stencil_w_top_patt, combined_simple_stencil_no_patt = create_and_combine_stencils_onesided(width, height, size, post_cropped_pattern, empty_stencil_1, empty_stencil_2, pattern_type)

    top_pattern_w_extended_lines = f"{getFileStepCounter()}_pattern_w_extended_lines.svg"
    incrementFileStepCounter()
    drawExtensionLines(combined_simple_stencil_w_top_patt, combined_simple_stencil_no_patt, top_pattern_w_extended_lines, side_type, width, height, 0)
    # ------

    # --- for bottom half ---
    combined_simple_stencil_w_bot_patt, _ = create_and_combine_stencils_onesided(width, height, size, post_cropped_bottom_pattern, empty_stencil_1, empty_stencil_2, pattern_type)

    bottom_pattern_w_extended_lines = f"{getFileStepCounter()}_bottom_pattern_w_extended_lines.svg"
    incrementFileStepCounter()
    drawExtensionLines(combined_simple_stencil_w_bot_patt, combined_simple_stencil_no_patt, bottom_pattern_w_extended_lines, side_type, width, height, 0)
    # ------

    mirrored_bottom_pattern_extended = f"{getFileStepCounter()}_mirrored_bottom_pattern_extended.svg"
    incrementFileStepCounter()
    mirrored_top_pattern_extended = f"{getFileStepCounter()}_mirrored_top_pattern_extended.svg"
    incrementFileStepCounter()
    if side_type == SideType.OneSided:
        mirrorLines(bottom_pattern_w_extended_lines, mirrored_bottom_pattern_extended, width, height, pattern_type)
        combinePatternAndMirrorWithStencils(top_pattern_w_extended_lines, combined_simple_stencil_no_patt, mirrored_bottom_pattern_extended)

    elif side_type == SideType.TwoSided:
        mirrorLines(top_pattern_w_extended_lines, mirrored_top_pattern_extended, width, 0, pattern_type)
        mirrorLines(bottom_pattern_w_extended_lines, mirrored_bottom_pattern_extended, width, 0, pattern_type)
        combined_patt_and_mirror_top = f"{getFileStepCounter()}_combined_patt_and_mirror_top.svg"
        incrementFileStepCounter()
        combined_patt_and_mirror_bottom = f"{getFileStepCounter()}_combined_patt_and_mirror_bottom.svg"
        incrementFileStepCounter()
        combineStencils(top_pattern_w_extended_lines, mirrored_top_pattern_extended, combined_patt_and_mirror_top)
        combineStencils(bottom_pattern_w_extended_lines, mirrored_bottom_pattern_extended, combined_patt_and_mirror_bottom)

        # create copy of the combined pattern and mirror
        paths, attributes = svg2paths(combined_patt_and_mirror_bottom)
        combined_patt_and_mirror_copy = f"{getFileStepCounter()}_combined_patt_and_mirror_copy.svg"
        incrementFileStepCounter()
        paths_copy = copy.deepcopy(paths)
        attributes_copy = copy.deepcopy(attributes)
        wsvg(paths_copy, attributes=attributes_copy, filename=combined_patt_and_mirror_copy, dimensions=(width, height))

        # INSTEAD OF THIS TRANSLATE MIGHT BE WHERE WE MIRROR OVER THE X AXIS
        # translateSVGBy(combined_patt_and_mirror_copy, combined_patt_and_mirror_copy, 0, height)
        mirrorSVGOverXAxis(combined_patt_and_mirror_copy, combined_patt_and_mirror_copy, width, height)

        combined_patt_and_mirror = f"{getFileStepCounter()}_combined_patt_and_mirror.svg"
        incrementFileStepCounter()
        combineStencils(combined_patt_and_mirror_top, combined_patt_and_mirror_copy, combined_patt_and_mirror)

        combinePatternAndMirrorWithStencils(combined_patt_and_mirror, combined_simple_stencil_no_patt, combined_patt_and_mirror_copy)
