import sys
from PyQt6.QtSvg import QSvgRenderer, QSvgGenerator
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtGui import QImage, QPainter, QTransform, QPixmap, QColor
from PyQt6.QtCore import QSize, QByteArray, QRectF
from shapely.geometry import LineString, Polygon, MultiLineString, MultiPolygon

import xml.etree.ElementTree as ET
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier, QuadraticBezier, parse_path

import tempfile
import svgwrite
import copy
import numpy as np
import cv2 as cv
import math
from svgwrite.container import Group

from VectorAlgoUtils import (
    rotateSVG,
    resizeSVG,
    translateSVGBy,
    mirrorSVGOverXAxis,
    mirrorSVGOverYAxis,
    translateSVGTo,
    crop_svg,
    savePixmapToCvImage,
    saveSvgFileAsPixmap,
    rotateImage,
    get_rgb_from_qcolor
)

from VectorAlgoStencils import (
    create_classic_pattern_stencils,
    create_and_combine_stencils_onesided,
    combineStencils,
    drawEmptyStencil,
    create_asymmetric_pattern_stencils,
    create_symmetric_pattern_stencils,
    create_simple_pattern_stencils
)

from ShapeMode import (
    ShapeMode
)

from PatternType import (
    PatternType
)

from SideType import (
    SideType
)

from GlobalVariables import(
    getFileStepCounter,
    incrementFileStepCounter,
    getDrawingSquareSize,
    setDrawingSquareSize,
    getNumClassicLines,
    getShapeColor,
    getBackgroundColor,
    getCurrentPatternType,
)

# VectorAlgo will be used when the user presses the "Update SVG" button

# The algorithm will be given a image, of SVG type, of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting), and user drawing
#   - this PNG that is returned is the final image once the algorithm has processesed the user drawing and weave pattern
# 2. Assembly instructions to guide the user in weaving the output image


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

def createFinalHeartDisplaySimpleCase(mask, points, foreground_color, background_color):
    if background_color != (255,255,255):
        cv.fillPoly(mask, [points], foreground_color)

    # Draw diamonds at each corner
    corner_diamond_size = math.ceil(math.sqrt(31**2 / 2) / 2)
    corner_diamond_diagonal = math.ceil(math.sqrt(31**2 * 2) / 2)
    
    left_x, left_y = points[0][0], points[0][1]
    top_x, top_y = points[1][0], points[1][1]
    right_x, right_y = points[2][0], points[2][1]
    bottom_x, bottom_y = points[3][0], points[3][1]
    
    upper_left_rect = np.array([
        [left_x + corner_diamond_size, left_y - corner_diamond_size],  # Top Left
        [top_x - corner_diamond_size, top_y + corner_diamond_size],  # Top Right
        [top_x - 1.5, top_y + corner_diamond_diagonal - 1.5],  # Bottom Right
        [left_x + corner_diamond_diagonal - 1.5, left_y - 1.5]  # Bottom Left
    ], dtype=np.int32)

    cv.fillPoly(mask, [upper_left_rect], background_color)

    upper_right_rect = np.array([
        [top_x + corner_diamond_size, top_y + corner_diamond_size],  # Top Left
        [right_x - corner_diamond_size, right_y - corner_diamond_size],  # Top Right
        [right_x - corner_diamond_diagonal, right_y],  # Bottom Right
        [top_x, top_y + corner_diamond_diagonal]  # Bottom Left
    ], dtype=np.int32)

    cv.fillPoly(mask, [upper_right_rect], background_color)

    lower_left_rect = np.array([
        [left_x + corner_diamond_diagonal - 1.5, left_y + 1.5],  # Top Left
        [bottom_x - 1.5, bottom_y - corner_diamond_diagonal + 1.5],  # Top Right
        [bottom_x - corner_diamond_size, bottom_y - corner_diamond_size + 1],  # Bottom Right
        [left_x + corner_diamond_size, left_y + corner_diamond_size + 1]  # Bottom Left
    ], dtype=np.int32)

    cv.fillPoly(mask, [lower_left_rect], background_color)

    lower_right_rect = np.array([
        [bottom_x, bottom_y - corner_diamond_diagonal],  # Top Left
        [right_x - corner_diamond_diagonal, right_y],  # Top Right
        [right_x - corner_diamond_size, right_y + corner_diamond_size],  # Bottom Right
        [bottom_x + corner_diamond_size, bottom_y - corner_diamond_size]  # Bottom Left
    ], dtype=np.int32)

    cv.fillPoly(mask, [lower_right_rect], background_color)

def createFinalHeartDisplaySymAsymCase(mask, points, foreground_color, background_color):
    if background_color != (255,255,255):
        cv.fillPoly(mask,[points],background_color)

    # Draw diamonds at each corner
    corner_diamond_size = math.sqrt(31**2 // 2)
    # distance_of_pattern_top_from_top_diamond = ?
    # distance_of_pattern_bottom_from_bottom_diamond = ?
    
    counter = 1
    for (x, y) in points:
        diamond = np.array([
            [x, y - corner_diamond_size],  # Top
            [x + corner_diamond_size, y],  # Right
            [x, y + corner_diamond_size],  # Bottom
            [x - corner_diamond_size, y]   # Left
        ], dtype=np.int32)

        if counter == 2 or counter == 4:
            cv.fillPoly(mask, [diamond], background_color)  
        
        else:  
            cv.fillPoly(mask, [diamond], foreground_color)

        counter += 1

def creatFinalHeartDisplayClassicCase(mask, points, foreground_color, background_color):
    num_segments = getNumClassicLines() + 1
    for i in range(len(points)):
        pt1 = np.array(points[i])
        pt2 = np.array(points[(i + 1) % len(points)])
        
        edge_vector = pt2 - pt1
        segment_vector = edge_vector / num_segments

        # Perpendicular vector to push line inward slightly
        normal = np.array([-edge_vector[1], edge_vector[0]])
        normal = normal / np.linalg.norm(normal)
        inward_offset = normal * 4  # adjust this for inward offset
        
        for j in range(num_segments):
            t = j + 0.5  # center of the segment
            midpoint = pt1 + segment_vector * t
            start_pos = midpoint + inward_offset - segment_vector * 0.5
            end_pos = midpoint + inward_offset + segment_vector * 0.5

            color = foreground_color if j % 2 == 0 else background_color

            # Draw the line segment
            cv.line(
                mask,
                tuple(start_pos.astype(int)),
                tuple(end_pos.astype(int)),
                color,
                thickness=7  # Adjust thickness as needed
            )


def createFinalHeartDisplay(image, pattern_type):
    line_color = (0, 0, 0)  # Black color
    height, width, _ = image.shape # isolated_pattern's shape is a square
    square_size =   height // 2  # Ensuring a balanced square
    center = (height // 2, width // 2)

    # Create a blank mask
    mask = np.full_like(image, 255)

    # Define square corners
    half_size = square_size // 2
    points = np.array([
        [center[0] - half_size, center[1]], # Left
        [center[0], center[1] - half_size], # Top
        [center[0] + half_size, center[1]], # Right
        [center[0], center[1] + half_size]  # Bottom
    ])
    
    foreground_color = get_rgb_from_qcolor(getShapeColor())
    background_color = get_rgb_from_qcolor(getBackgroundColor())
    
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
    
    if getShapeColor() != QColor("black"):
        cv.ellipse(mask, halfway_point_upper_left, (radius, radius), 0, -225, -45, foreground_color, thickness=-1)
        
    if getBackgroundColor() != QColor("white"):
        cv.ellipse(mask, halfway_point_upper_right, (radius, radius), 0, -135, 45, background_color , -1)
        
    # Draw the rotated square (diamond)
    cv.polylines(mask, [points], isClosed=True, color=line_color, thickness=3)
    
    if pattern_type == PatternType.Classic:
        creatFinalHeartDisplayClassicCase(mask, points, foreground_color, background_color)
    elif pattern_type == PatternType.Simple:
        createFinalHeartDisplaySimpleCase(mask, points, foreground_color, background_color)
    else:
        createFinalHeartDisplaySymAsymCase(mask, points, foreground_color, background_color)
    
    rotated_mask = rotateImage(mask, -45)

    # scale the pattern to fit inside the square of the heart
    square_width_rounded = math.floor(square_width)
    if pattern_type != PatternType.Classic:
        square_width_rounded = square_width_rounded - 31 # and use padding if not Classic
    scaled_pattern = cv.resize(image, (square_width_rounded, square_width_rounded), interpolation=cv.INTER_LANCZOS4)

    # Calculate coordinates to overlay scaled_pattern on the square portion of the heart
    x_center = (rotated_mask.shape[1] - square_width_rounded) // 2
    y_center = (rotated_mask.shape[0] - square_width_rounded) // 2

    # Overlay scaled_pattern onto the square portion of the heart
    rotated_mask[y_center:y_center + square_width_rounded, x_center:x_center + square_width_rounded] = scaled_pattern

    # rotate the heart back to its original, upright, position
    reverse_rotated_mask = rotateImage(rotated_mask, 45)

    return reverse_rotated_mask

def createFinalHeartCutoutPatternExport(size, side_type, pattern_type, n_lines=0, line_color='black', background_color='white'):
    print("pattern type: ", pattern_type)
    print("sides: ", side_type)

    width = size
    height = size // 2

    empty_stencil_1 = drawEmptyStencil(width, height, 0, file_name=f"{getFileStepCounter()}_stencil1.svg")
    incrementFileStepCounter()
    empty_stencil_2 = drawEmptyStencil(width, height, height, file_name=f"{getFileStepCounter()}_stencil2.svg")
    incrementFileStepCounter()

    preprocessed_pattern = "preprocessed_pattern.svg"

    # Check if the preprocessed pattern is blank
    is_blank = False
    try:
        paths, _ = svg2paths(preprocessed_pattern)
        if paths:
            is_blank = False
        else:
            is_blank = True
    except:
        is_blank = True
        print("Error: Preprocessed pattern is blank or not found.")

    print("is blank: ", is_blank)

    if pattern_type != PatternType.Classic and is_blank:
        pattern_type = PatternType.Simple

    if pattern_type == PatternType.Simple:
        print("Creating SIMPLE pattern")

        create_simple_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type, is_blank)

    elif pattern_type == PatternType.Symmetric:
        print("Creating SYMMETRICAL pattern")

        create_symmetric_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    elif pattern_type == PatternType.Asymmetric:
        print("Creating A-SYMMETRICAL pattern")

        create_asymmetric_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    elif pattern_type == PatternType.Classic:
        print("Creating CLASSIC pattern")
        n_lines = getNumClassicLines()
        create_classic_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, n_lines, is_blank)

def mainAlgorithmSvg(img, side_type, pattern_type, function='show', n_lines = 0):

    match function:

        case 'show':
            # convert SVG to CV Image for createFinalHeartDisplay
            heartPixmap = saveSvgFileAsPixmap(img)
            heartCvImage = savePixmapToCvImage(heartPixmap)

            return createFinalHeartDisplay(heartCvImage, pattern_type)

        case _:
            return createFinalHeartCutoutPatternExport(1200, side_type, pattern_type, n_lines)
