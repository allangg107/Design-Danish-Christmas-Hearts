from ShapeMode import(
    ShapeMode
)
from PyQt6.QtGui import (
    QColor
)
from PatternType import (
    PatternType
)
from SideType import (
    SideType
)

"""Standarised global variable values"""
MARGIN = 31
FILE_STEP_COUNTER = 1 
HAS_IMAGE = True # Global variable for if stencil has a drawn image only matters for 
                 # the classic case currently in GuideWindow
SHAPE_MODE = ShapeMode.Cursor
SHAPE_COLOR = QColor(0, 0, 0, 255)
BACKGROUND_COLOR = QColor(255, 255, 255, 255)
PEN_WIDTH = 1
FILLED = True
USER_OUTPUT_SVG_FILENAME = "svg_file.svg"
USER_PREPROCESSED_PATTERN = "preprocessed_pattern.svg"
CURRENT_PATTERN_TYPE = PatternType.Simple
CURRENT_SIDE = SideType.OneSided

"""Functions to get the global values"""

def getMargin():
    return MARGIN

def getFileStepCounter():
    return FILE_STEP_COUNTER

def getDrawingSquareSize():
    return DRAWING_SQUARE_SIZE

def getHasImage():
    return HAS_IMAGE

def getShapeMode():
    return SHAPE_MODE

def getShapeColor():
    return SHAPE_COLOR

def getBackgroundColor():
    return BACKGROUND_COLOR

def getPenWidth():
    return PEN_WIDTH

def getFilled():
    return FILLED

def getUserOutputSVGFileName():
    return USER_OUTPUT_SVG_FILENAME

def getUserPreprocessedPattern():
    return USER_PREPROCESSED_PATTERN

def getCurrentPatternType():
    return CURRENT_PATTERN_TYPE

def getCurrentSideType():
    return CURRENT_SIDE    

"""Functions to set global values"""

def incrementFileStepCounter():
    global FILE_STEP_COUNTER
    FILE_STEP_COUNTER += 1

def setDrawingSquareSize(value):
    global DRAWING_SQUARE_SIZE
    DRAWING_SQUARE_SIZE = value

def setMargin(value):
    global MARGIN
    MARGIN = value

def setHasImage(value):
    global HAS_IMAGE
    if isinstance(value, bool):
        HAS_IMAGE = value
    else:
        print("Error: ", value, "is not an instance of a boolean value")

def setShapeMode(value):
    global SHAPE_MODE
    if isinstance(value, ShapeMode):
        SHAPE_MODE = value
    else:
        print("Error: ", value, "is not an instance of ", ShapeMode)

def setShapeColor(color):
    global SHAPE_COLOR
    if isinstance(color, QColor):
        SHAPE_COLOR = color
    else:
        print("Error: ", color, "is not an instance of QColor")

def setBackgroundColor(color):
    global BACKGROUND_COLOR
    if isinstance(color, QColor):
        BACKGROUND_COLOR = color
    else:
        print("Error: ", color, "is not an instance of QColor")
        

def setPenWidth(width):
    global PEN_WIDTH
    PEN_WIDTH = width

def setFilled(value):
    global FILLED
    if isinstance(value, bool):
        FILLED = value
    
    else:
        print("Error: ", value, "is not an instance of a boolean value")

def setUserOutputSVGFileName(name):
    global USER_OUTPUT_SVG_FILENAME
    USER_OUTPUT_SVG_FILENAME = name

def setUserPreprocessedPattern(name):
    global USER_PREPROCESSED_PATTERN
    USER_PREPROCESSED_PATTERN = name

def setCurrentPatternType(pattern):
    global CURRENT_PATTERN_TYPE
    if isinstance(pattern, PatternType):
        CURRENT_PATTERN_TYPE = pattern
    else:
        print("Error: ", pattern, "is not an instance of ", PatternType)
    

def setCurrentSideType(side):
    global CURRENT_SIDE
    if isinstance(side, SideType):
        CURRENT_SIDE = side
    else:
        print("Error: ", side, "is not an instance of ", SideType)