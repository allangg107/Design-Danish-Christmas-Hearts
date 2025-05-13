import svgwrite
import xml.etree.ElementTree as ET
import copy
import cv2 as cv
import numpy as np
import math
from PyQt6.QtCore import QPoint
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
    mirrorSVGOverXAxisWithY,
    mirrorSVGOver45DegreeLine,
    removeDuplicateLinesFromSVG,
    convertLinesToRectangles,
    convertLineToRectangle,
    extractSemiCirclesFromPattern,
    grabTopMostPointOfPaths,
    grabBottomMostPointOfPaths,
    grabLeftMostPointOfPaths,
    grabRightMostPointOfPaths,
    combineStencils,
    fileIsNonEmpty,
    findClassicLinesToDelete
)

from PatternType import (
    PatternType
)

from SideType import (
    SideType
)

from ShapeMode import (
    ShapeMode
)
from GlobalVariables import (
    getMargin,
    setMargin,
    setDrawingSquareSize,
    getDrawingSquareSize,
    incrementFileStepCounter,
    getFileStepCounter,
    getShapeColor,
    getNumClassicLines,
    getLineThicknessAndExtension,
    getUserOutputSVGFileName,
    getClassicIndicesLineDeleteList,
    setClassicIndicesLineDeleteList,
    setClassicPatternSnapPoints,
    setClassicPatternClassicLines
)




"""Overlaying of SVG files"""

def overlayDrawingOnStencil(stencil_file, user_drawing_file, size, square_size, pattern_type, margin_x=getMargin(), margin_y=0, filename='combined_output.svg'):

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

        combined_paths = paths1 + paths2
        combined_attributes = attributes1 + attributes2

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


def overlayPatternOnStencil(pattern, stencil, size, stencil_number, pattern_type, margin=getMargin()):
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

def drawEmptyStencil(width, height, starting_y, margin_x=getMargin(), line_color='black', file_name="allan is a miracle.svg"):
    dwg = svgwrite.Drawing(file_name, size=(width,height+starting_y))

    # define the square size
    square_size = (height // 1.5) - margin_x
    margin_y = margin_x + starting_y

    # draw the left square
    left_top_line_start = (margin_x + square_size // 2, margin_y)
    left_top_line_end = (left_top_line_start[0] + square_size, left_top_line_start[1])
    left_bottom_line_start = (left_top_line_start[0], left_top_line_start[1] + square_size)
    left_bottom_line_end = (left_bottom_line_start[0] + square_size, left_bottom_line_start[1])

    dwg.add(dwg.line(start=(left_top_line_start), end=(left_top_line_end), stroke="red", stroke_width=1))
    dwg.add(dwg.line(start=(left_bottom_line_start), end=(left_bottom_line_end), stroke="red", stroke_width=1))

    # draw the left arc
    radius_x = square_size / 2
    radius_y = square_size / 2
    arc_start = (left_bottom_line_start[0], left_bottom_line_start[1])
    arc_end = (left_top_line_start[0], left_top_line_start[1])
    left_arc_path = f"M {arc_start[0]},{arc_start[1]} A {radius_x},{radius_y} 0 0,1 {arc_end[0]},{arc_end[1]}"

    dwg.add(dwg.path(d=left_arc_path, stroke="purple", fill="none", stroke_width=1))

    # draw the right square
    right_top_line_start = left_top_line_end
    right_top_line_end = (right_top_line_start[0] + square_size, right_top_line_start[1])
    right_bottom_line_start = left_bottom_line_end
    right_bottom_line_end = (right_bottom_line_start[0] + square_size, right_bottom_line_start[1])

    dwg.add(dwg.line(start=(right_top_line_start), end=(right_top_line_end), stroke="blue", stroke_width=1))
    dwg.add(dwg.line(start=(right_bottom_line_start), end=(right_bottom_line_end), stroke="blue", stroke_width=1))

    # draw the right arc
    arc_start = (right_top_line_end[0], right_top_line_end[1])
    arc_end = (right_bottom_line_end[0], right_bottom_line_end[1])
    right_arc_path = f"M {arc_start[0]},{arc_start[1]} A {radius_x},{radius_y} 0 0,1 {arc_end[0]},{arc_end[1]}"

    dwg.add(dwg.path(d=right_arc_path, stroke="yellow", fill="none", stroke_width=1))

    # draw small cross in lower left corner
    if starting_y == 0:
        cross_start = (left_bottom_line_start[0], left_bottom_line_start[1] + 7 - 15)
        cross_end = (left_bottom_line_start[0] + 14, left_bottom_line_start[1] - 7 - 15)
        dwg.add(dwg.line(start=cross_start, end=cross_end, stroke="black", stroke_width=1))
        cross_start = (left_bottom_line_start[0] + 14, left_bottom_line_start[1] + 7 - 15)
        cross_end = (left_bottom_line_start[0], left_bottom_line_start[1] - 7 - 15)
        dwg.add(dwg.line(start=cross_start, end=cross_end, stroke="black", stroke_width=1))

        # draw a circle around the cross
        circle_center = (left_bottom_line_start[0] + 7, left_bottom_line_start[1] - 15)
        circle_radius = 10
        dwg.add(dwg.circle(center=circle_center, r=circle_radius, stroke="black", fill="none", stroke_width=1))

    else:
        cross_start = (right_bottom_line_end[0] - 14, right_bottom_line_end[1] + 7 - 15)
        cross_end = (right_bottom_line_end[0], right_bottom_line_end[1] - 7 - 15)
        dwg.add(dwg.line(start=cross_start, end=cross_end, stroke="black", stroke_width=1))
        cross_start = (right_bottom_line_end[0], right_bottom_line_end[1] + 7 - 15)
        cross_end = (right_bottom_line_end[0] - 14, right_bottom_line_end[1] - 7 - 15)
        dwg.add(dwg.line(start=cross_start, end=cross_end, stroke="black", stroke_width=1))

    dwg.save()

    return file_name


def drawInnerCutLines(width, height, starting_y, margin_x=getMargin(), line_color='black', file_name="allans_test.svg"):
    dwg = svgwrite.Drawing(file_name, size=(width, height + starting_y))

    # Define the square size
    square_size = (height // 1.5) - margin_x
    margin_y = margin_x + starting_y
    extension = getLineThicknessAndExtension()  # Amount to extend the lines

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
                            stroke="brown", stroke_width=1))

    dwg.add(dwg.line(start=(left_bottom_line_start[0] - extension, left_bottom_line_start[1] - margin_x),
                            end=(right_bottom_line_end[0] + extension, right_bottom_line_end[1] - margin_x),
                            stroke="brown", stroke_width=1))

    dwg.save()

    return file_name


def createClassicInnerCuts(width, height, starting_y, n_lines, side_type, margin_x=getMargin(), line_color='black'):
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

    offset = square_size / (n_lines + 1)

    paths = []
    attributes = []

    new_paths = []
    new_attributes = []

    current_index = n_lines * 2 - 1 if starting_y != 0 else 2
    print(f"current_index: {current_index}")
    for i in range(n_lines):
        if current_index not in getClassicIndicesLineDeleteList():
            start_of_line=complex(left_top_line_start[0] - 3, left_top_line_start[1] + offset * (i + 1))
            end_of_line=complex(right_top_line_end[0] + 3, left_top_line_start[1] + offset * (i + 1))

            if current_index % 2 == 1:
                # if the current index is odd, add only the left side of the line
                if side_type == SideType.TwoSided: # if two sided, don't add the other side of the respective line on the other stencil
                    if (2) + (i * 2) in getClassicIndicesLineDeleteList():
                        start_of_line=complex(left_top_line_end[0] + 3, left_top_line_start[1] + offset * (i + 1))
            else:
                if side_type == SideType.TwoSided:
                    if (n_lines * 2 - 1) - (i * 2) in getClassicIndicesLineDeleteList():
                        end_of_line=complex(right_top_line_start[0] - 3, left_top_line_start[1] + offset * (i + 1))
                        

            line = Line(
                start=start_of_line,
                end=end_of_line
            )
            new_paths.append(line)
            new_attributes.append({'stroke': line_color, 'stroke-width': 1, 'fill': 'none'})
        else: # if the current index is in the delete list:
            if current_index % 2 == 1: # if the current index is odd, add only the left side of the line
                end_of_line=complex(left_bottom_line_end[0] + 3, left_top_line_start[1] + offset * (i + 1))
                if side_type == SideType.TwoSided: # if two sided, don't add the other side of the respective line on the other stencil
                    if (2) + (i * 2) in getClassicIndicesLineDeleteList():
                        end_of_line=complex(left_top_line_start[0] - 3, left_top_line_start[1] + offset * (i + 1))
                
                line = Line(
                    start=complex(left_top_line_start[0] - 3, left_top_line_start[1] + offset * (i + 1)),
                    end=end_of_line
                )
                new_paths.append(line)
                new_attributes.append({'stroke': line_color, 'stroke-width': 1, 'fill': 'none'})
            else: # if the current index is even, add only the right side of the line
                start_of_line=complex(right_top_line_start[0] - 3, left_top_line_start[1] + offset * (i + 1))
                if side_type == SideType.TwoSided:
                    if (n_lines * 2 - 1) - (i * 2) in getClassicIndicesLineDeleteList():
                        start_of_line=complex(right_top_line_end[0] + 3, left_top_line_start[1] + offset * (i + 1))
                
                line = Line(
                    start=start_of_line,
                    end=complex(right_top_line_end[0] + 3, left_top_line_start[1] + offset * (i + 1))
                )
                new_paths.append(line)
                new_attributes.append({'stroke': line_color, 'stroke-width': 1, 'fill': 'none'})

        if starting_y != 0:
            current_index -= 2 # increment by 2, not 1
        else:
            current_index += 2

    paths.extend(new_paths)
    attributes.extend(new_attributes)

    return paths, attributes


def create_and_combine_stencils_onesided(width, height, size, stencil_1_pattern, empty_stencil_1, empty_stencil_2, pattern_type, n_lines= -1, is_blank=False):
    # Create both simple stencils
    stencil_1_inner_cuts = drawInnerCutLines(width, height, 0, file_name=f"{getFileStepCounter()}_simpleStencil1.svg")
    incrementFileStepCounter()
    stencil_2_inner_cuts = drawInnerCutLines(width, height, height, file_name=f"{getFileStepCounter()}_simpleStencil2.svg")
    incrementFileStepCounter()

    convertLinesToRectangles(stencil_1_inner_cuts, stencil_1_inner_cuts)
    convertLinesToRectangles(stencil_2_inner_cuts, stencil_2_inner_cuts)

    if is_blank:
        final_output_top = getUserOutputSVGFileName() + "_top.svg"
        final_output_bottom = getUserOutputSVGFileName() + "_bottom.svg"

        combineStencils(empty_stencil_1, stencil_1_inner_cuts, final_output_top)
        combineStencils(empty_stencil_2, stencil_2_inner_cuts, final_output_bottom)

        final_output_combined = getUserOutputSVGFileName() + "_combined.svg"
        combineStencils(final_output_top, final_output_bottom, final_output_combined)

        return -1, -1, -1

    # Combine all stencils first
    stencil_1_combined = f"{getFileStepCounter()}_stencil_1_combined.svg"
    incrementFileStepCounter()
    combineStencils(empty_stencil_1, stencil_1_inner_cuts, stencil_1_combined)

    stencil_2_combined = f"{getFileStepCounter()}_stencil_2_combined.svg"
    incrementFileStepCounter()
    combineStencils(empty_stencil_2, stencil_2_inner_cuts, stencil_2_combined)

    combined_stencils = f"{getFileStepCounter()}_combined_stencils.svg"
    incrementFileStepCounter()
    combineStencils(stencil_1_combined, stencil_2_combined, combined_stencils)

    # rotate the pattern 90 degrees counter-clockwise
    rotated_path_name = f"{getFileStepCounter()}_fixed_pattern_rotation.svg"
    incrementFileStepCounter()
    rotateSVG(stencil_1_pattern, rotated_path_name, -90)

    # Now overlay the pattern on the combined stencil
    overlayed_pattern = overlayPatternOnStencil(rotated_path_name, combined_stencils, size, 1, pattern_type)
    return overlayed_pattern, stencil_1_combined, stencil_2_combined


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

    print(f"Number of output paths: {len(combined_paths)}")
    for i, path in enumerate(combined_paths):
        print(f"Output path {i} has {len(path)} segments")

    return combined_paths, combined_attrs


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# def grabLeftMostPointOfPaths(paths):
#     """Grab the left most point from a path or a list of paths"""
#     min_x = float('inf')
#     min_point = None

#     # Convert single path to a list for consistent processing
#     if not isinstance(paths, list):
#         paths = [paths]

#     for path in paths:
#         for segment in path:
#             # Sample points along the segment
#             for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
#                 pt = segment.point(t)
#                 if pt.real < min_x:
#                     min_x = pt.real
#                     min_point = pt

#     return min_point


# def grabRightMostPointOfPaths(paths):
#     """Grab the right most point from a path or a list of paths"""
#     max_x = float('-inf')
#     max_point = None

#     # Convert single path to a list for consistent processing
#     if not isinstance(paths, list):
#         paths = [paths]

#     for path in paths:
#         for segment in path:
#             # Sample points along the segment
#             for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
#                 pt = segment.point(t)
#                 if pt.real > max_x:
#                     max_x = pt.real
#                     max_point = pt

#     return max_point


# def grabTopMostPointOfPaths(paths):
#     """Grab the top most point from a path or a list of paths"""
#     min_y = float('inf')  # Using min_y since in SVG coordinate system, lower y values are higher up
#     min_point = None

#     # Convert single path to a list for consistent processing
#     if not isinstance(paths, list):
#         paths = [paths]

#     for path in paths:
#         for segment in path:
#             # Sample points along the segment
#             for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
#                 pt = segment.point(t)
#                 if pt.imag < min_y:
#                     min_y = pt.imag
#                     min_point = pt

#     return min_point


# def grabBottomMostPointOfPaths(paths):
#     """Grab the bottom most point from a path or a list of paths"""
#     max_y = float('-inf')  # Using max_y since in SVG coordinate system, higher y values are lower down
#     max_point = None

#     # Convert single path to a list for consistent processing
#     if not isinstance(paths, list):
#         paths = [paths]

#     for path in paths:
#         for segment in path:
#             # Sample points along the segment
#             for t in np.linspace(0, 1, 20):  # Sample 20 points per segment
#                 pt = segment.point(t)
#                 if pt.imag > max_y:
#                     max_y = pt.imag
#                     max_point = pt

#     return max_point


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


def find_bottommost_horizontal_line(svg_file):
    """
    Find the bottommost horizontal line in an SVG file.
    
    Args:
        svg_file: Path to the SVG file
        
    Returns:
        Tuple of (path, path_index, segment_index, line) for the bottommost horizontal line,
        or None if no horizontal line is found
    """
    paths, attributes = svg2paths(svg_file)

    bottommost_y = float('-inf')
    bottommost_horizontal_line = None
    bottommost_path_index = -1
    bottommost_segment_index = -1

    for path_index, path in enumerate(paths):
        for segment_index, segment in enumerate(path):
            # Check if the segment is a horizontal line (or nearly horizontal)
            if isinstance(segment, Line):
                # Calculate the angle of the line with the y-axis
                dx = segment.end.real - segment.start.real
                dy = segment.end.imag - segment.start.imag

                # Check if line is horizontal (slope is close to zero)
                if abs(dy) < 1e-10:  # Almost zero change in y direction
                    # Find the y-coordinate of this horizontal line
                    y_coord = segment.start.imag  # or segment.end.imag, they're the same

                    # If this is the bottommost horizontal line found so far
                    if y_coord > bottommost_y:
                        bottommost_y = y_coord
                        bottommost_horizontal_line = segment
                        bottommost_path_index = path_index
                        bottommost_segment_index = segment_index

    if bottommost_horizontal_line is None:
        return None

    return (paths[bottommost_path_index], bottommost_path_index, bottommost_segment_index, bottommost_horizontal_line)


def find_two_topmost_horizontal_lines(svg_file):
    """
    Find the topmost horizontal line in an SVG file.
    
    Args:
        svg_file: Path to the SVG file
        
    Returns:
        Tuple of (path, path_index, segment_index, line) for the topmost horizontal line,
        or None if no horizontal line is found
    """
    paths, attributes = svg2paths(svg_file)

    # List to store horizontal lines with their information
    horizontal_lines = []

    for path_index, path in enumerate(paths):
        for segment_index, segment in enumerate(path):
            # Check if the segment is a horizontal line (or nearly horizontal)
            if isinstance(segment, Line):
                # Calculate the angle of the line with the y-axis
                dx = segment.end.real - segment.start.real
                dy = segment.end.imag - segment.start.imag

                # Check if line is horizontal (slope is close to zero)
                if abs(dy) < 1e-10:  # Almost zero change in y direction
                    # Find the y-coordinate of this horizontal line
                    y_coord = segment.start.imag  # or segment.end.imag, they're the same

                    # Store line info in our list
                    horizontal_lines.append({
                        'path': path,
                        'path_index': path_index,
                        'segment_index': segment_index,
                        'line': segment,
                        'y_coord': y_coord
                    })

    # Sort by y-coordinate (ascending - lower y is higher in SVG)
    horizontal_lines.sort(key=lambda x: x['y_coord'])

    # Return the two topmost lines (if we have them)
    if len(horizontal_lines) == 0:
        return None
    elif len(horizontal_lines) == 1:
        # Only one horizontal line found
        line_info = horizontal_lines[0]
        return (line_info['path'], line_info['path_index'], line_info['segment_index'], line_info['line'])
    else:
        # Two or more horizontal lines found
        line_info1 = horizontal_lines[0]
        line_info2 = horizontal_lines[1]
        return (
            (line_info1['path'], line_info1['path_index'], line_info1['segment_index'], line_info1['line']),
            (line_info2['path'], line_info2['path_index'], line_info2['segment_index'], line_info2['line'])
        )


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


def get_horizontal_line_endpoints(horizontal_line):
    """
    Get the left and right points of a horizontal line.
    
    Args:
        horizontal_line: Line segment object
        
    Returns:
        Tuple of (left_point, right_point)
    """
    if horizontal_line.start.real <= horizontal_line.end.real:
        return (horizontal_line.start, horizontal_line.end)
    else:
        return (horizontal_line.end, horizontal_line.start)


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


def drawExtensionLines(combined_stencil, stencil_pattern, output_name, side_type, is_asym_bot, width, height, starting_y, margin_x=getMargin()):

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
    combined_file_name = f"{getFileStepCounter()}_combined_shapes.svg"
    incrementFileStepCounter()
    wsvg(combined_paths, attributes=combined_attrs, filename=combined_file_name, dimensions=(width, width))

    combined_shapes_mirrored_over_self = f"{getFileStepCounter()}_combined_shapes_mirrored_over_self.svg"
    incrementFileStepCounter()
    mirrorSVGOverXAxisWithY(combined_file_name, combined_shapes_mirrored_over_self, width, height, getMargin() + square_size // 2)

    # rotate "combined_shapes.svg" 45 degrees clockwise
    rotated_path_name = f"{getFileStepCounter()}_rotated_pattern_step.svg"
    incrementFileStepCounter()
    rotateSVG(combined_shapes_mirrored_over_self, rotated_path_name, 45, width // 2, height // 2)

    # find the right-most vertical line of the pattern. Grab the top and bottom points from it
    bottommost_horizontal_line = find_bottommost_horizontal_line(rotated_path_name)
    bottommost_horizontal_line_2 = None
    if is_asym_bot:
        # the asymetric's LoS appears to be a combination of two horizontal lines instead of only one
        bottommost_horizontal_line, bottommost_horizontal_line_2 = find_two_topmost_horizontal_lines(rotated_path_name)
    if bottommost_horizontal_line is None:
        print("Error: No horizontal line found.")
        return

    rotated_back_name = f"{getFileStepCounter()}_rotated_back.svg"
    incrementFileStepCounter()
    rotateSVG(rotated_path_name, rotated_back_name, -45, width // 2, height // 2)

    # extract the top point from the rightmost vertical line
    path, path_index, segment_index, line = bottommost_horizontal_line
    left_point, right_point = get_horizontal_line_endpoints(line)

    path_index_2, segment_index_2, line_2 = None, None, None
    if is_asym_bot:
        # extract the top point from the rightmost vertical line
        _, path_index_2, segment_index_2, line_2 = bottommost_horizontal_line_2
        left_point_2, right_point_2 = get_horizontal_line_endpoints(line_2)

        # set left point to the leftmost point of the two lines
        left_point = left_point if left_point.real < left_point_2.real else left_point_2
        # set right point to the rightmost point of the two lines
        right_point = right_point if right_point.real > right_point_2.real else right_point_2

    # rotate the top and bottom points back to the original orientation
    left_point_rotated = rotatePoint(left_point, -45, (width // 2, height // 2))
    right_point_rotated = rotatePoint(right_point, -45, (width // 2, height // 2))

    if is_asym_bot:
        # flip combined_shapes_mirrored_over_self over the 45 degree line
        mirrorSVGOver45DegreeLine(rotated_back_name, rotated_back_name, right_point_rotated, width, height)

    mirrored_paths, mirrored_attrs = svg2paths(rotated_back_name)

    combined_paths_w_lines = copy.deepcopy(mirrored_paths)
    combined_attrs_w_lines = copy.deepcopy(mirrored_attrs)

    del combined_paths_w_lines[path_index][segment_index]
    if is_asym_bot:
        # If both segments are in the same path and the first deleted segment is before the second,
        # we need to adjust the index for the second deletion
        if path_index == path_index_2 and segment_index < segment_index_2:
            segment_index_2 -= 1

        # find the line in the second path and remove it
        del combined_paths_w_lines[path_index_2][segment_index_2]


    if is_asym_bot:
        wsvg(combined_paths_w_lines, attributes=combined_attrs_w_lines, filename="test.svg", dimensions=(width, width))

    # Draw a line from the bottom most point to the left edge of the stencil and draw a line from the top most point to the right edge of the stencil
    for path, attr in zip(combined_paths, combined_attrs):
        extension = getLineThicknessAndExtension()

        stencil_square_start = margin_x + square_size // 2

        if side_type == SideType.OneSided:
            # Create a line from the top_point_rotated to the left edge of the stencil
            top_of_los = Line(left_point_rotated, complex(stencil_square_start - extension, left_point_rotated.imag))

            # Create a line from the bottom_point_rotated to the right edge of the stencil
            bottom_of_los = Line(right_point_rotated, complex(stencil_square_start + square_size * 2 + extension, right_point_rotated.imag))

        elif side_type == SideType.TwoSided:
            # Create a line from the top_point_rotated to the left edge of the stencil
            top_of_los = Line(left_point_rotated, complex(stencil_square_start - extension, left_point_rotated.imag))

            # Create a line from the bottom_point_rotated to the right edge of the stencil
            bottom_of_los = Line(right_point_rotated, complex(margin_x + square_size * 1.5, right_point_rotated.imag))

        # Add the left_line and right_line to the final paths
        combined_paths_w_lines.append(convertLineToRectangle(top_of_los))
        combined_attrs_w_lines.append({'stroke': 'red', 'stroke-width': 1, 'fill': 'red'})
        combined_paths_w_lines.append(convertLineToRectangle(bottom_of_los))
        combined_attrs_w_lines.append({'stroke': 'blue', 'stroke-width': 1, 'fill': 'blue'})

    # Save the final SVG with extended lines
    wsvg(combined_paths_w_lines, attributes=combined_attrs_w_lines, filename=output_name, dimensions=(width, width))


def mirrorLines(pattern_w_extended_lines, output_name, width, height, pattern_type):

    temp_paths, temp_attrs = svg2paths(pattern_w_extended_lines)
    wsvg(temp_paths, attributes=temp_attrs, filename="test_2.svg", dimensions=(width, height))

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
    translateSVGBy(fixed_rounding_mirrored_lines, output_name, distance_between - getMargin(), height)


def combinePatternAndMirrorWithStencils(pattern_w_extended_lines, simple_stencil_1, translated_mirrored_lines, simple_stencil_2):

    final_output_top = getUserOutputSVGFileName() + "_top.svg"
    final_output_bottom = getUserOutputSVGFileName() + "_bottom.svg"

    combineStencils(pattern_w_extended_lines, simple_stencil_1, final_output_top)
    combineStencils(translated_mirrored_lines, simple_stencil_2, final_output_bottom)

    final_output_combined = getUserOutputSVGFileName() + "_combined.svg"
    combineStencils(final_output_top, final_output_bottom, final_output_combined)


def create_simple_pattern_stencils(stencil_1_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type, is_blank):

    combined_simple_stencil_w_patt, simple_stencil_1, simple_stencil_2 = create_and_combine_stencils_onesided(width, height, size, stencil_1_pattern, empty_stencil_1, empty_stencil_2, pattern_type, is_blank=is_blank)

    if is_blank:
        return

    combined_simple_stencil_no_patt = f"{getFileStepCounter()}_combined_simple_stencil_no_patt.svg"
    incrementFileStepCounter()
    combineStencils(simple_stencil_1, simple_stencil_2, combined_simple_stencil_no_patt)

    processed_pattern = f"{getFileStepCounter()}_translated_pattern.svg"
    incrementFileStepCounter()
    processed_pattern = removeDuplicateLinesFromSVG(combined_simple_stencil_w_patt, combined_simple_stencil_no_patt)

    mirrored_processed_pattern = f"{getFileStepCounter()}_combined_shapes_mirrored_over_self.svg"
    incrementFileStepCounter()
    square_size = (height // 1.5) - getMargin()
    mirrorSVGOverXAxisWithY(processed_pattern, mirrored_processed_pattern, width, height, getMargin() + square_size // 2)

    final_output_top = getUserOutputSVGFileName() + "_top.svg"
    final_output_bottom = getUserOutputSVGFileName() + "_bottom.svg"

    combineStencils(mirrored_processed_pattern, simple_stencil_1, final_output_top)

    if side_type == SideType.TwoSided:
        mirrored_pattern = f"{getFileStepCounter()}_mirrored_pattern.svg"
        incrementFileStepCounter()
        mirrorLines(mirrored_processed_pattern, mirrored_pattern, width, 0, pattern_type)

        combineStencils(final_output_top, mirrored_pattern, final_output_top)

    bottom_paths, bottom_attrs = svg2paths(simple_stencil_2)

    # Extract viewBox from simple_stencil_2
    tree = ET.parse(simple_stencil_2)
    root = tree.getroot()
    viewbox = root.get('viewBox')

    # If viewBox isn't at root level, search for svg element
    if viewbox is None:
        for elem in root.iter():
            if 'viewBox' in elem.attrib:
                viewbox = elem.get('viewBox')
                break

    # If still no viewBox found, use default dimensions
    if viewbox is None:
        viewbox = f"0 0 {width} {height}"
    wsvg(bottom_paths, attributes=bottom_attrs, filename=final_output_bottom, dimensions=(width, height), viewbox=viewbox)

    final_output_combined = getUserOutputSVGFileName() + "_combined.svg"
    combineStencils(final_output_top, final_output_bottom, final_output_combined)


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


def snapShapeToClassicCuts(classic_cuts, shape_type, begin_point, end_point, width, height):
    lines = []
    for line_coords, _ in classic_cuts:
        line = LineString([(line_coords[0], line_coords[1]), (line_coords[2], line_coords[3])])
        lines.append(line)

    # Add edge box lines (bounding box edges)
    border_lines = [
        LineString([(width // 2, 0), (width, height // 2)]),  # Top-right edge
        LineString([(width, height // 2), (width // 2, height)]),  # Bottom-right edge
        LineString([(width // 2, height), (0, height // 2)]),  # Bottom-left edge
        LineString([(0, height // 2), (width // 2, 0)])  # Top-left edge
    ]
    lines.extend(border_lines)

    # Find all intersection points
    intersection_points = []

    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j:  # Skip duplicate checks and self-intersection
                continue

            # Check for intersection
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                if intersection.is_empty:
                    continue
                if hasattr(intersection, 'x') and hasattr(intersection, 'y'):
                    intersection_points.append(complex(intersection.x, intersection.y))

    snap_points = intersection_points.copy()
    
    if len(classic_cuts) == getNumClassicLines() * 2:
        setClassicPatternSnapPoints(snap_points)
        setClassicPatternClassicLines(classic_cuts)

    # find the center point of each square formed by intersection points that are within close proximity
    center_points = []
    if shape_type == ShapeMode.Circle:
        for i in range(len(intersection_points)):
            for j in range(i + 1, len(intersection_points)):
                # Calculate distance between points
                dist = abs(intersection_points[i] - intersection_points[j])
                # Check if points are within a spacing equivalent to the spacing between classic cuts
                # if dist <= 89:
                    # Calculate the center point between two intersection points
                center_x = (intersection_points[i].real + intersection_points[j].real) / 2
                center_y = (intersection_points[i].imag + intersection_points[j].imag) / 2
                center_points.append(complex(center_x, center_y))

    # Find the midpoint between each intersection point on line segments that have a 45 degree angle
    up_midpoints = []
    all_midpoints = []
    # Process each line to find midpoints between intersections
    for line in lines:
        # Get the line coordinates
        line_coords = list(line.coords)
        if len(line_coords) < 2:
            continue

        # Calculate the slope
        dx = line_coords[1][0] - line_coords[0][0]
        dy = line_coords[1][1] - line_coords[0][1]

        # Find all intersections of this line with other lines
        line_intersections = []
        for other_line in lines:
            if line != other_line and line.intersects(other_line):
                intersection = line.intersection(other_line)
                if hasattr(intersection, 'x') and hasattr(intersection, 'y'):
                    line_intersections.append((intersection.x, intersection.y))

        # Sort intersections along the line
        if len(line_intersections) >= 2:
            # Sort by distance from start point
            start_point = line_coords[0]
            line_intersections.sort(key=lambda p: ((p[0] - start_point[0])**2 + (p[1] - start_point[1])**2)**0.5)

            # Calculate midpoints between consecutive intersections
            for i in range(len(line_intersections) - 1):
                mid_x = (line_intersections[i][0] + line_intersections[i+1][0]) / 2
                mid_y = (line_intersections[i][1] + line_intersections[i+1][1]) / 2

                current_line = LineString([(line_intersections[i][0], line_intersections[i][1]),
                                          (line_intersections[i+1][0], line_intersections[i+1][1])])

                def isPartofBorder(current_line, border_lines, tolerance=1e-6):
                    for border in border_lines:
                        # Check if the lines share an endpoint
                        if current_line.intersects(border):
                            intersection = current_line.intersection(border)

                            # If they overlap significantly or share endpoints
                            if (hasattr(intersection, 'length') and intersection.length > tolerance) or \
                               (current_line.distance(border) < tolerance):
                                return True

                        # Check if the lines are parallel and close to each other
                        if current_line.distance(border) < tolerance:
                            return True

                    return False

                is_part_of_border = isPartofBorder(current_line, border_lines)

                if (shape_type == ShapeMode.Square or shape_type == ShapeMode.Heart) and ((abs(dx) > 1e-6 and abs(dy) == abs(dx) and dy < 0) or (is_part_of_border)):
                    up_midpoints.append(complex(mid_x, mid_y))
                elif shape_type == ShapeMode.Circle:
                    all_midpoints.append(complex(mid_x, mid_y))

    if shape_type == ShapeMode.Square or shape_type == ShapeMode.Heart:
        snap_points = up_midpoints.copy()
    if shape_type == ShapeMode.Circle:
        snap_points = all_midpoints.copy()

    # Find closest snap points to begin_point and end_point
    if len(snap_points) > 0:
        # Convert QPoint to complex before comparing
        begin_complex = complex(begin_point.x(), begin_point.y())
        end_complex = complex(end_point.x(), end_point.y())

        # Now use the complex versions for distance
        if shape_type == ShapeMode.Circle:
            closest_to_begin = min(center_points,
                                key=lambda p: abs(p - begin_complex))
        else:
            closest_to_begin = min(snap_points,
                                key=lambda p: abs(p - begin_complex))
        closest_to_end = min(snap_points,
                             key=lambda p: abs(p - end_complex))

        # Convert back to QPoint for return

        if shape_type == ShapeMode.Semicircle:
            # delete classic lines in between the two points
            delete_list = getClassicIndicesLineDeleteList()
            # find the classic lines that are between the two points
            classic_lines_to_delete = findClassicLinesToDelete(closest_to_begin, closest_to_end)

            delete_list.extend(classic_lines_to_delete)
            setClassicIndicesLineDeleteList(delete_list)

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


def getLineGroup(path_file, file_name, orientation="vertical"):
    paths, attrs = svg2paths(path_file)
    # group the lines based on the specified orientation

    if orientation == "vertical":
        # Find the topmost and bottommost points in paths
        topmost_point = float('inf')
        bottommost_point = float('-inf')
        for path in paths:
            for line in path:
                if isinstance(line, Line):
                    topmost_point = min(topmost_point, line.start.imag, line.end.imag)
                    bottommost_point = max(bottommost_point, line.start.imag, line.end.imag)

        midpoint = (topmost_point + bottommost_point) / 2

        first_group = []
        second_group = []
        for path in paths:
            for line in path:
                if isinstance(line, Line):
                    # Check if the line is above or below the midpoint
                    if line.start.imag < midpoint:
                        first_group.append(line)
                    elif line.start.imag >= midpoint:
                        second_group.append(line)

    elif orientation == "horizontal":
        # Find the leftmost and rightmost points in paths
        leftmost_point = float('inf')
        rightmost_point = float('-inf')
        for path in paths:
            for line in path:
                if isinstance(line, Line):
                    leftmost_point = min(leftmost_point, line.start.real, line.end.real)
                    rightmost_point = max(rightmost_point, line.start.real, line.end.real)

        midpoint = (leftmost_point + rightmost_point) / 2

        first_group = []
        second_group = []
        for path in paths:
            for line in path:
                if isinstance(line, Line):
                    # Check if the line is to the left or right of the midpoint
                    if line.start.real < midpoint:
                        first_group.append(line)
                    elif line.start.real >= midpoint:
                        second_group.append(line)
    else:
        raise ValueError("Orientation must be either 'vertical' or 'horizontal'")

    # Convert the list of Line objects into a Path object
    first_path_group = [Path(*first_group)] if first_group else []
    second_path_group = [Path(*second_group)] if second_group else []

    path_group = first_path_group + second_path_group
    attrs = [{'stroke': getShapeColor().name(), 'stroke-width': 1, 'fill': 'none'}] * len(path_group)

    wsvg(path_group, attributes=attrs, filename=file_name)

def splitShapesIntoQuarters(shapes_file, horizontal_lines, vertical_lines, all_top_and_bottom_quarters, all_middle_halves, width, height, square_size, side_type):
    shapes, attrs = svg2paths(shapes_file)

    combined_middle_halves = f"{getFileStepCounter()}_combined_middle_halves.svg"
    incrementFileStepCounter()
    combined_top_and_bottom_quarters = f"{getFileStepCounter()}_combined_top_and_bottom_quarters.svg"
    incrementFileStepCounter()
    for shape in shapes:
        print("Shape:", shape)
        square_start, square_width, square_height = locateSquare(shape, horizontal_lines, vertical_lines)
        wsvg(shape, attributes=attrs, filename="shape.svg", dimensions=(square_width, square_height), viewbox=(square_start[0], square_start[1], square_width, square_height))
        shape_paths, attributes = svg2paths("shape.svg")

        # 2. call crop_svg to crop the top quarter
        top_quarter_paths = crop_svg(shape_paths, square_start[0], square_start[1] - 2, square_width, square_height / 4 + 2, False)
        top_quarter = f"{getFileStepCounter()}_top_quarter.svg"
        incrementFileStepCounter()
        wsvg(top_quarter_paths, attributes=attrs, filename=top_quarter, dimensions=(square_width, square_height / 4), viewbox=(square_start[0], square_start[1], square_width, square_height / 4))

        # 6. call crop_svg to crop the bottom quarter
        bottom_quarter_paths = crop_svg(shape_paths, square_start[0], square_start[1] + square_height * 3 / 4, square_width, square_height / 4 + 2, False)
        bottom_quarter = f"{getFileStepCounter()}_bottom_quarter.svg"
        incrementFileStepCounter()
        wsvg(bottom_quarter_paths, attributes=attrs, filename=bottom_quarter, dimensions=(square_width, square_height / 4), viewbox=(square_start[0], square_start[1] + square_height * 3 / 4, square_width, square_height / 4))

        # Combine top and bottom quarters into a single SVG file
        top_and_bottom_quarter = f"{getFileStepCounter()}_top_and_bottom_quarter.svg"
        incrementFileStepCounter()
        combineStencils(top_quarter, bottom_quarter, top_and_bottom_quarter)

        # Combine all top and bottom quarters
        combineStencils(combined_top_and_bottom_quarters, top_and_bottom_quarter, combined_top_and_bottom_quarters)

        # 3. call crop_svg to crop the middle half
        middle_half_paths = crop_svg(shape_paths, square_start[0] - 2, square_start[1] + square_height / 4, square_width + 4, square_height / 2, False)
        middle_half = f"{getFileStepCounter()}_middle_half.svg"
        incrementFileStepCounter()
        wsvg(middle_half_paths, attributes=attrs, filename=middle_half, dimensions=(square_width, square_height / 2), viewbox=(square_start[0] - 2, square_start[1] + square_height / 4, square_width + 4, square_height / 2))

        translated_middle_half = f"{getFileStepCounter()}_translated_middle_half.svg"
        incrementFileStepCounter()
        translateSVGBy(middle_half, translated_middle_half, square_size, 0)

        split_halves = f"{getFileStepCounter()}_split_halves.svg"
        incrementFileStepCounter()
        getLineGroup(translated_middle_half, split_halves, "horizontal")

        # Combine all middle halves
        # combineStencils(combined_middle_halves, translated_middle_half, combined_middle_halves)
        combineStencils(combined_middle_halves, split_halves, combined_middle_halves)

    temp_file = f"{getFileStepCounter()}_temp_file.svg"
    incrementFileStepCounter()
    temp_paths, temp_attrs = svg2paths(combined_middle_halves)
    wsvg(temp_paths, attributes=temp_attrs, filename=temp_file, dimensions=(width, height))

    # grab the pattern area on the other side and rotate by -90 degrees
    rotated_combined_middle_halves = f"{getFileStepCounter()}_rotated_combined_middle_halves.svg"
    incrementFileStepCounter()
    rotateSVG(temp_file, rotated_combined_middle_halves, -90, getMargin() + square_size * 2, getMargin() + square_size / 2)

    translated_combined_middle_halves = rotated_combined_middle_halves
    if side_type == SideType.OneSided:
        # translate the pattern to the other stencil
        translated_combined_middle_halves = f"{getFileStepCounter()}_translated_combined_middle_halves.svg"
        incrementFileStepCounter()
        translateSVGBy(rotated_combined_middle_halves, translated_combined_middle_halves, 0, height)
    elif side_type == SideType.TwoSided:
        combineStencils(combined_top_and_bottom_quarters, rotated_combined_middle_halves, combined_top_and_bottom_quarters)

        two_sided_combined_middle_halves = f"{getFileStepCounter()}_translated_combined_middle_halves.svg"
        incrementFileStepCounter()
        all_paths, all_attrs = svg2paths(combined_top_and_bottom_quarters)
        wsvg(all_paths, attributes=all_attrs, filename=two_sided_combined_middle_halves, dimensions=(width, height))

        translated_combined_middle_halves = f"{getFileStepCounter()}_translated_combined_middle_halves.svg"
        incrementFileStepCounter()
        translateSVGBy(two_sided_combined_middle_halves, translated_combined_middle_halves, 0, height)

    quarter_paths, quarter_attrs = svg2paths(combined_top_and_bottom_quarters)
    half_paths, half_attrs = svg2paths(translated_combined_middle_halves)

    # Save the final combined SVG files
    wsvg(quarter_paths, attributes=quarter_attrs, filename=all_top_and_bottom_quarters)
    wsvg(half_paths, attributes=half_attrs, filename=all_middle_halves)


def locateSquare(shape, horizontal_cuts, vertical_cuts):
    """
    Locate the square in which the given shape resides based on the classic cuts.

    Args:
        shape: The shape whose position needs to be located.
        classic_cuts: A list of classic cut lines (both horizontal and vertical).

    Returns:
        A tuple representing the square coordinates (x, y, width, height).
    """

    # Step 1: Find the topmost point of the shape
    topmost_point = min((segment.point(t) for segment in shape for t in np.linspace(0, 1, 20)),
                        key=lambda pt: pt.imag)

    # Step 2: Locate the nearest horizontal classic cut line to the topmost point
    nearest_horizontal_cut = min(
        (cut for cut in horizontal_cuts),  # Horizontal lines
        key=lambda cut: abs(cut.start.imag - topmost_point.imag)
    )

    # Step 3: Find the leftmost point of the shape
    leftmost_point = min((segment.point(t) for segment in shape for t in np.linspace(0, 1, 20)),
                         key=lambda pt: pt.real)

    # Step 4: Locate the nearest vertical classic cut line to the leftmost point
    nearest_vertical_cut = min(
        (cut for cut in vertical_cuts),  # Vertical lines
        key=lambda cut: abs(cut.start.real - leftmost_point.real)
    )

    # Step 5: Determine the square coordinates based on the intersection of the nearest horizontal and vertical cuts
    intersection_point = (
        nearest_vertical_cut.start.real,  # x-coordinate from vertical line
        nearest_horizontal_cut.start.imag  # y-coordinate from horizontal line
    )

    vertical_line_index = vertical_cuts.index(nearest_vertical_cut)
    horizontal_line_index = horizontal_cuts.index(nearest_horizontal_cut)

    square_width = abs(vertical_cuts[vertical_line_index].start.real - vertical_cuts[vertical_line_index + 1].start.real)
    square_height = abs(horizontal_cuts[horizontal_line_index].start.imag - horizontal_cuts[horizontal_line_index + 1].start.imag)

    print(f"Vertical line index: {vertical_line_index}, Horizontal line index: {horizontal_line_index}")

    return intersection_point, square_width, square_height


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


def createSquareGrid(square_size, n_lines, offset):
    grid_start = (getMargin() + square_size // 2, getMargin())
    horizontal_lines = []
    vertical_lines = []
    attributes = []
    for i in range(0, n_lines + 1 + 1):
        # Horizontal lines
        horizontal_lines.append(Path(Line(complex(grid_start[0], grid_start[1] + i * offset), complex(grid_start[0] + square_size, grid_start[1] + i * offset))))
        # Vertical lines
        vertical_lines.append(Path(Line(complex(grid_start[0] + i * offset, grid_start[1]), complex(grid_start[0] + i * offset, grid_start[1] + square_size))))

        attributes.append({'stroke': 'black', 'stroke-width': 1, 'fill': 'none'})

    wsvg(horizontal_lines, attributes=attributes, filename="horizontal_lines.svg", dimensions=(square_size, square_size))
    wsvg(vertical_lines, attributes=attributes, filename="vertical_lines.svg", dimensions=(square_size, square_size))

    combineStencils("horizontal_lines.svg", "vertical_lines.svg", "grid_lines.svg")

    return horizontal_lines, vertical_lines


def attach45DegreeLinesAndRemoveInbetween(quarters, classic_cuts, output_name):
    # load in the svg file of one instance of a single drawing
    # each drawing should have a top and bottom quarter disconnected from each other
    quarter_paths, quarter_attrs = svg2paths(quarters)
    classic_cut_paths, classic_cut_attrs = svg2paths(classic_cuts)

    quarter_paths_copy = copy.deepcopy(quarter_paths)
    quarter_attributes_copy = copy.deepcopy(quarter_attrs)

    # Store modified classic cut lines in a dictionary keyed by y-coordinate
    modified_classic_lines = {}
    attached_endpoints = []

    for quarter_path in quarter_paths:
        # find the left- and right-most point of each path
        left_most_point = None
        right_most_point = None
        min_x = float('inf')
        max_x = float('-inf')

        for segment in quarter_path:
            # Sample points along the segment
            for t in np.linspace(0, 1, 10):  # Sample 10 points per segment
                point = segment.point(t)
                x, y = point.real, point.imag

                if x < min_x:
                    min_x = x
                    left_most_point = (x, y)

                if x > max_x:
                    max_x = x
                    right_most_point = (x, y)

        # find the closest classic cut line to the left and right-most point of each path
        closest_classic_line = None
        min_left_distance = float('inf')

        for i, classic_cut in enumerate(classic_cut_paths):
            for segment in classic_cut:
                if isinstance(segment, Line):
                    line = LineString([(segment.start.real, segment.start.imag), (segment.end.real, segment.end.imag)])
                    # Get the y-coordinate from the first point of the line
                    distance = abs(line.coords[0][1] - left_most_point[1])
                    if distance < min_left_distance:
                        min_left_distance = distance
                        closest_classic_line = line

        # Draw a 45-degree line from the leftmost point to the closest classic cut line
        if closest_classic_line is not None:
            classic_line_y = closest_classic_line.coords[0][1]

            # Calculate intersection point for left side
            left_intersection_x = left_most_point[0] - abs(left_most_point[1] - classic_line_y)

            # Calculate intersection point for right side
            right_intersection_x = right_most_point[0] + abs(right_most_point[1] - classic_line_y)

            attached_endpoints.append((complex(left_intersection_x, classic_line_y), complex(right_intersection_x, classic_line_y)))
            intersection_y = classic_line_y

            # Draw left diagonal
            left_diagonal = Line(complex(left_most_point[0], left_most_point[1]),
                                complex(left_intersection_x, intersection_y))
            quarter_paths_copy.append(Path(left_diagonal))
            quarter_attributes_copy.append({'stroke': 'red', 'stroke-width': 1, 'fill': 'none'})

            # Draw right diagonal
            right_diagonal = Line(complex(right_most_point[0], right_most_point[1]),
                                complex(right_intersection_x, intersection_y))
            quarter_paths_copy.append(Path(right_diagonal))
            quarter_attributes_copy.append({'stroke': 'blue', 'stroke-width': 1, 'fill': 'none'})

            # Get the original line coordinates
            orig_start_x, orig_start_y = closest_classic_line.coords[0]
            orig_end_x, orig_end_y = closest_classic_line.coords[-1]

            # Check if this classic line is already in our modified dictionary
            if classic_line_y in modified_classic_lines:
                # The line has already been modified, update the segments
                segments = modified_classic_lines[classic_line_y]
                new_segments = []

                for segment in segments:
                    seg_start_x, seg_start_y = segment[0]
                    seg_end_x, seg_end_y = segment[1]

                    # Check if this segment overlaps with our current removal
                    if (seg_start_x <= left_intersection_x and seg_end_x >= left_intersection_x) or \
                       (seg_start_x <= right_intersection_x and seg_end_x >= right_intersection_x) or \
                       (seg_start_x >= left_intersection_x and seg_end_x <= right_intersection_x):

                        # This segment overlaps with our removal area
                        if seg_start_x < left_intersection_x:
                            # Keep part to the left of our removal
                            new_segments.append([(seg_start_x, seg_start_y), (left_intersection_x, intersection_y)])

                        if seg_end_x > right_intersection_x:
                            # Keep part to the right of our removal
                            new_segments.append([(right_intersection_x, intersection_y), (seg_end_x, seg_end_y)])
                    else:
                        # This segment doesn't overlap with our removal area
                        new_segments.append(segment)

                modified_classic_lines[classic_line_y] = new_segments
            else:
                # First modification to this line
                segments = []

                # Add left segment if it exists
                if orig_start_x < left_intersection_x:
                    segments.append([(orig_start_x, orig_start_y), (left_intersection_x, intersection_y)])

                # Add right segment if it exists
                if orig_end_x > right_intersection_x:
                    segments.append([(right_intersection_x, intersection_y), (orig_end_x, orig_end_y)])

                modified_classic_lines[classic_line_y] = segments

    # check for overlapping removed classic cut portions
    # Group by y-coordinate (classic cut line)
    by_y_coord = {}
    for left_point, right_point in attached_endpoints:
        y = left_point.imag  # Both points have the same y-coordinate
        if y not in by_y_coord:
            by_y_coord[y] = []
        # Store the segment endpoints
        by_y_coord[y].append((left_point.real, right_point.real))

    # For each classic cut line, find segments removed more than once
    for y_coord, endpoints in by_y_coord.items():
        if len(endpoints) < 2:  # Need at least 2 pairs of endpoints to have overlap
            continue

        # Create a list of events (start or end of a segment)
        events = []
        for left, right in endpoints:
            # Ensure left < right
            if left > right:
                left, right = right, left
            events.append((left, 1))  # Start of segment
            events.append((right, -1))  # End of segment

        # Sort events by x-coordinate
        events.sort()

        # Sweep through events and detect overlapping segments
        count = 0
        overlap_start = None

        for x, event_type in events:
            count += event_type

            if count > 1 and overlap_start is None:
                # Start of an overlapping segment
                overlap_start = x
            elif count <= 1 and overlap_start is not None:
                # End of an overlapping segment
                # Add the overlapping segment back to modified_classic_lines
                if y_coord in modified_classic_lines:
                    modified_classic_lines[y_coord].append([(overlap_start, y_coord), (x, y_coord)])
                else:
                    modified_classic_lines[y_coord] = [[(overlap_start, y_coord), (x, y_coord)]]
                overlap_start = None

    # Create new classic cut paths from the modified segments
    new_classic_cut_paths = []
    for y_coord, segments in modified_classic_lines.items():
        for segment in segments:
            start_x, start_y = segment[0]
            end_x, end_y = segment[1]
            line = Line(complex(start_x, start_y), complex(end_x, end_y))
            new_classic_cut_paths.append(Path(line))

    # Keep any classic cut paths that weren't modified
    for quarter_path in classic_cut_paths:
        for segment in quarter_path:
            if isinstance(segment, Line):
                line_y = segment.start.imag
                if line_y not in modified_classic_lines:
                    new_classic_cut_paths.append(Path(segment))

    # Update classic cut attributes
    new_classic_cut_attrs = [{'stroke': 'yellow', 'stroke-width': 1, 'fill': 'none'}] * len(new_classic_cut_paths)

    # Save the modified classic cuts
    wsvg(new_classic_cut_paths, attributes=new_classic_cut_attrs, filename=classic_cuts)

    # Save the output with the diagonal lines
    wsvg(quarter_paths_copy, attributes=quarter_attributes_copy, filename=output_name)


"""Create Non-Simple stencils"""

def create_classic_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, n_lines, is_blank):
    # 1. create the classic inner cuts
    stencil_1_classic_cuts_paths, stencil_1_classic_cuts_attr = createClassicInnerCuts(width, height, 0, n_lines, side_type)
    stencil_2_classic_cuts_paths, stencil_2_classic_cuts_attr = createClassicInnerCuts(width, height, height, n_lines, side_type)

    stencil_1_classic_cuts = f"{getFileStepCounter()}_stencil_1_classic_cuts.svg"
    incrementFileStepCounter()
    stencil_2_classic_cuts = f"{getFileStepCounter()}_stencil_2_classic_cuts.svg"
    incrementFileStepCounter()

    wsvg(stencil_1_classic_cuts_paths, attributes=stencil_1_classic_cuts_attr, filename=stencil_1_classic_cuts, dimensions=(width, width))
    wsvg(stencil_2_classic_cuts_paths, attributes=stencil_2_classic_cuts_attr, filename=stencil_2_classic_cuts, dimensions=(width, width))

    if is_blank:
        final_output_top = getUserOutputSVGFileName() + "_top.svg"
        final_output_bottom = getUserOutputSVGFileName() + "_bottom.svg"

        converted_lines_to_rectangles_1 = f"{getFileStepCounter()}_converted_lines_to_rectangles_1.svg"
        incrementFileStepCounter()
        convertLinesToRectangles(stencil_1_classic_cuts, converted_lines_to_rectangles_1)

        converted_lines_to_rectangles_2 = f"{getFileStepCounter()}_converted_lines_to_rectangles_2.svg"
        incrementFileStepCounter()
        convertLinesToRectangles(stencil_2_classic_cuts, converted_lines_to_rectangles_2)

        combineStencils(converted_lines_to_rectangles_1, empty_stencil_1, final_output_top)
        combineStencils(converted_lines_to_rectangles_2, empty_stencil_2, final_output_bottom)

        final_output_combined = getUserOutputSVGFileName() + "_combined.svg"
        combineStencils(final_output_top, final_output_bottom, final_output_combined)

        return

    # a. rotate -90
    rotated_path_name = f"{getFileStepCounter()}_fixed_pattern_rotation.svg"
    incrementFileStepCounter()
    rotateSVG(preprocessed_pattern, rotated_path_name, -90)

    # b. re-size to square_size
    square_size = (height // 1.5) - getMargin()
    resize_size = square_size
    resized_pattern_name = f"{getFileStepCounter()}_scaled_pattern.svg"
    incrementFileStepCounter()
    resizeSVG(rotated_path_name, resized_pattern_name, resize_size)

    # c. translate drawing to the classic line position
    offset = square_size / (n_lines + 1)

    x_shift = getMargin() * 4 + square_size // 4
    x_shift = x_shift - 1

    y_shift = (getMargin() * 3)
    y_shift = y_shift - getMargin() * 2

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

    mirrored_pattern = f"{getFileStepCounter()}_mirrored_pattern.svg"
    incrementFileStepCounter()
    mirrorSVGOverXAxisWithY(unfilled_pattern, mirrored_pattern, width, height, getMargin() + square_size / 2)

    bottom_stencil_semi_circles = f"{getFileStepCounter()}_bottom_stencil_semi_circles.svg"
    incrementFileStepCounter()

    top_stencil_semi_circles = f"{getFileStepCounter()}_top_stencil_semi_circles.svg"
    incrementFileStepCounter()

    pattern_no_semi_circles = f"{getFileStepCounter()}_pattern_no_semi_circles.svg"
    incrementFileStepCounter()

    extractSemiCirclesFromPattern(mirrored_pattern, bottom_stencil_semi_circles, top_stencil_semi_circles, pattern_no_semi_circles, width, height, square_size, side_type, n_lines)

    # split each shape into a top quarter, middle half, and bottom quarter
    top_and_bottom_quarters_of_shapes = f"{getFileStepCounter()}_top_bottom_quarters.svg"
    incrementFileStepCounter()
    middle_halves = f"{getFileStepCounter()}_middle_half_pattern.svg"
    incrementFileStepCounter()
    horizontal_lines, vertical_lines = createSquareGrid(square_size, n_lines, offset)
    if fileIsNonEmpty(pattern_no_semi_circles):
        splitShapesIntoQuarters(pattern_no_semi_circles, horizontal_lines, vertical_lines, top_and_bottom_quarters_of_shapes, middle_halves, width, height, square_size, side_type)

    #  attach a 45 degree line from each end of the quarters to their closest classic cut line/stencil line
    updated_classic_cuts = f"{getFileStepCounter()}_updated_classic_cuts.svg"
    incrementFileStepCounter()
    classic_paths, classic_attrs = svg2paths(stencil_1_classic_cuts)
    wsvg(classic_paths, attributes=classic_attrs, filename=updated_classic_cuts)

    top_stencil_shapes = ""
    if fileIsNonEmpty(top_and_bottom_quarters_of_shapes) and not fileIsNonEmpty(top_stencil_semi_circles):
        top_stencil_shapes = top_and_bottom_quarters_of_shapes
    elif fileIsNonEmpty(top_stencil_semi_circles) and not fileIsNonEmpty(top_and_bottom_quarters_of_shapes):
        top_stencil_shapes = top_stencil_semi_circles
    else:
        top_stencil_shapes = f"{getFileStepCounter()}_top_stencil_shapes.svg"
        incrementFileStepCounter()
        combineStencils(top_and_bottom_quarters_of_shapes, top_stencil_semi_circles, top_stencil_shapes)

    bottom_stencil_shapes = ""
    if fileIsNonEmpty(middle_halves) and not fileIsNonEmpty(bottom_stencil_semi_circles):
        bottom_stencil_shapes = middle_halves
    elif fileIsNonEmpty(bottom_stencil_semi_circles) and not fileIsNonEmpty(middle_halves):
        bottom_stencil_shapes = bottom_stencil_semi_circles
    else:
        bottom_stencil_shapes = f"{getFileStepCounter()}_bottom_stencil_shapes.svg"
        incrementFileStepCounter()
        combineStencils(middle_halves, bottom_stencil_semi_circles, bottom_stencil_shapes)

    top_stencil_shapes_with_lines = f"{getFileStepCounter()}_top_stencil_shapes_w_lines.svg"
    incrementFileStepCounter()
    if fileIsNonEmpty(top_stencil_shapes):
        attach45DegreeLinesAndRemoveInbetween(top_stencil_shapes, updated_classic_cuts, top_stencil_shapes_with_lines)

    # repeat above with middle_halves and stencil_2_classic_cuts
    updated_classic_cuts_2 = f"{getFileStepCounter()}_updated_classic_cuts_2.svg"
    incrementFileStepCounter()
    classic_paths_2, classic_attrs_2 = svg2paths(stencil_2_classic_cuts)
    wsvg(classic_paths_2, attributes=classic_attrs_2, filename=updated_classic_cuts_2)

    bottom_stencil_shapes_w_lines = f"{getFileStepCounter()}_bottom_stencil_shapes_w_lines.svg"
    incrementFileStepCounter()
    if fileIsNonEmpty(bottom_stencil_shapes):
        attach45DegreeLinesAndRemoveInbetween(bottom_stencil_shapes, updated_classic_cuts_2, bottom_stencil_shapes_w_lines)

    converted_lines_to_rectangles_1 = f"{getFileStepCounter()}_converted_lines_to_rectangles_1.svg"
    incrementFileStepCounter()
    convertLinesToRectangles(updated_classic_cuts, converted_lines_to_rectangles_1)

    converted_lines_to_rectangles_2 = f"{getFileStepCounter()}_converted_lines_to_rectangles_2.svg"
    incrementFileStepCounter()
    convertLinesToRectangles(updated_classic_cuts_2, converted_lines_to_rectangles_2)

    final_output_top = getUserOutputSVGFileName() + "_top.svg"
    combineStencils(converted_lines_to_rectangles_1, empty_stencil_1, final_output_top)
    combineStencils(final_output_top, top_stencil_shapes_with_lines, final_output_top)

    final_output_bottom = getUserOutputSVGFileName() + "_bottom.svg"
    combineStencils(converted_lines_to_rectangles_2, empty_stencil_2, final_output_bottom)
    combineStencils(final_output_bottom, bottom_stencil_shapes_w_lines, final_output_bottom)

    final_output_combined = getUserOutputSVGFileName() + "_combined.svg"
    combineStencils(final_output_top, final_output_bottom, final_output_combined)


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

    combined_simple_stencil_w_patt, simple_stencil_1, simple_stencil_2 = create_and_combine_stencils_onesided(width, height, size, post_cropped_pattern, empty_stencil_1, empty_stencil_2, pattern_type)

    combined_simple_stencil_no_patt = f"{getFileStepCounter()}_combined_simple_stencil_no_patt.svg"
    incrementFileStepCounter()
    combineStencils(simple_stencil_1, simple_stencil_2, combined_simple_stencil_no_patt)

    # rotate the pattern, grab the 2 points on the line of symmetry, and then rotate it back (including the points we grabbed)

    # Draw lines from shapes to the edges of the stencil
    pattern_w_extended_lines = f"{getFileStepCounter()}_pattern_w_extended_lines.svg"
    incrementFileStepCounter()
    drawExtensionLines(combined_simple_stencil_w_patt, combined_simple_stencil_no_patt, pattern_w_extended_lines, side_type, False, width, height, 0)

    mirrored_pattern_extended = f"{getFileStepCounter()}_mirrored_pattern_extended.svg"
    incrementFileStepCounter()
    if side_type == SideType.OneSided:
        mirrorLines(pattern_w_extended_lines, mirrored_pattern_extended, width, height, pattern_type)
        combinePatternAndMirrorWithStencils(pattern_w_extended_lines, simple_stencil_1, mirrored_pattern_extended, simple_stencil_2)

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

        combinePatternAndMirrorWithStencils(combined_patt_and_mirror, simple_stencil_1, combined_patt_and_mirror_copy, simple_stencil_2)


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
    combined_simple_stencil_w_top_patt, simple_stencil_1, simple_stencil_2 = create_and_combine_stencils_onesided(width, height, size, post_cropped_pattern, empty_stencil_1, empty_stencil_2, pattern_type)

    combined_simple_stencil_no_patt = f"{getFileStepCounter()}_combined_simple_stencil_no_patt.svg"
    incrementFileStepCounter()
    combineStencils(simple_stencil_1, simple_stencil_2, combined_simple_stencil_no_patt)

    top_pattern_w_extended_lines = f"{getFileStepCounter()}_pattern_w_extended_lines.svg"
    incrementFileStepCounter()
    drawExtensionLines(combined_simple_stencil_w_top_patt, combined_simple_stencil_no_patt, top_pattern_w_extended_lines, side_type, False, width, height, 0)
    # ------

    # --- for bottom half ---
    combined_simple_stencil_w_bot_patt, _, _ = create_and_combine_stencils_onesided(width, height, size, post_cropped_bottom_pattern, empty_stencil_1, empty_stencil_2, pattern_type)

    bottom_pattern_w_extended_lines = f"{getFileStepCounter()}_bottom_pattern_w_extended_lines.svg"
    incrementFileStepCounter()
    drawExtensionLines(combined_simple_stencil_w_bot_patt, combined_simple_stencil_no_patt, bottom_pattern_w_extended_lines, side_type, True, width, height, 0)
    # ------

    temp_paths, temp_attrs = svg2paths(top_pattern_w_extended_lines)
    temp_paths, temp_attrs = svg2paths(bottom_pattern_w_extended_lines)

    mirrored_bottom_pattern_extended = f"{getFileStepCounter()}_mirrored_bottom_pattern_extended.svg"
    incrementFileStepCounter()
    mirrored_top_pattern_extended = f"{getFileStepCounter()}_mirrored_top_pattern_extended.svg"
    incrementFileStepCounter()
    if side_type == SideType.OneSided:
        mirrorLines(bottom_pattern_w_extended_lines, mirrored_bottom_pattern_extended, width, height, pattern_type)
        combinePatternAndMirrorWithStencils(top_pattern_w_extended_lines, simple_stencil_1, mirrored_bottom_pattern_extended, simple_stencil_2)

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
        mirrorSVGOverXAxisWithY(combined_patt_and_mirror_copy, combined_patt_and_mirror_copy, width, height, 500)

        combined_patt_and_mirror = f"{getFileStepCounter()}_combined_patt_and_mirror.svg"
        incrementFileStepCounter()
        combineStencils(combined_patt_and_mirror_top, combined_patt_and_mirror_copy, combined_patt_and_mirror)

        combinePatternAndMirrorWithStencils(combined_patt_and_mirror, simple_stencil_1, combined_patt_and_mirror_copy, simple_stencil_2)
