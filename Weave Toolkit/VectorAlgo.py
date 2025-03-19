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
    getFileStepCounter,
    incrementFileStepCounter,
    getDrawingSquareSize,
    setDrawingSquareSize
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

def createFinalHeartCutoutPatternExport(size, side_type, pattern_type, line_color='black', background_color='white'):
    

    print("pattern type: ", pattern_type)
    print("sides: ", side_type)

    width = size
    height = size // 2

    empty_stencil_1 = drawEmptyStencil(width, height, 0, file_name=f"{getFileStepCounter()}_stencil1.svg")
    incrementFileStepCounter()
    empty_stencil_2 = drawEmptyStencil(width, height, height, file_name=f"{getFileStepCounter()}_stencil2.svg")
    incrementFileStepCounter()

    preprocessed_pattern = "preprocessed_pattern.svg"

    if pattern_type == PatternType.Simple:
        print("Creating SIMPLE pattern")
        
        create_simple_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type)
    
    elif pattern_type == PatternType.Symmetric:
        print("Creating SYMMETRICAL pattern")

        create_symmetric_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    elif pattern_type == PatternType.Asymmetric:
        print("Creating A-SYMMETRICAL pattern")

        create_asymmetric_pattern_stencils(preprocessed_pattern, width, height, size, empty_stencil_1, empty_stencil_2, side_type, pattern_type)

    elif pattern_type == PatternType.Classic:
        print("Creating CLASSIC pattern")
        combined_classic_stencil = f"{getFileStepCounter()}_combined_classic_stencil.svg"
        incrementFileStepCounter()
        classic_stencil1 = create_classic_pattern_stencils(width, height, 0, file_name=f"{getFileStepCounter()}_classic_stencil1.svg")
        incrementFileStepCounter()
        classic_stencil2 = create_classic_pattern_stencils(width, height, height, file_name=f"{getFileStepCounter()}_classic_stencil2.svg")
        incrementFileStepCounter()
        final_stencil = f"{getFileStepCounter()}_classic_final_stencil.svg"
        incrementFileStepCounter()
        combined_classic_stencil_final = f"{getFileStepCounter()}_combined_classic_stencil_final.svg"
        incrementFileStepCounter()
        combineStencils(empty_stencil_1, classic_stencil1, combined_classic_stencil)
        combineStencils(empty_stencil_2, classic_stencil2, final_stencil)
        combineStencils(final_stencil, combined_classic_stencil, combined_classic_stencil_final)

        # resizeSvg(final_stencil, user_decided_export_size)

        # return final_stencil

def mainAlgorithmSvg(img, side_type, pattern_type, function='show'):

    match function:

        case 'show':
            # convert SVG to CV Image for createFinalHeartDisplay
            heartPixmap = saveSvgFileAsPixmap(img)
            heartCvImage = savePixmapToCvImage(heartPixmap)

            return createFinalHeartDisplay(heartCvImage)

        case _:
            return createFinalHeartCutoutPatternExport(1200, side_type, pattern_type)
