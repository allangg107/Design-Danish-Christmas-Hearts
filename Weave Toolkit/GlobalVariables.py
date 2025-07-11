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
SHAPE_MODE = ShapeMode.Cursor
SHAPE_COLOR = QColor(0, 0, 0, 255)
BACKGROUND_COLOR = QColor(255, 255, 255, 255)
PEN_WIDTH = 1
FILLED = True
USER_OUTPUT_SVG_FILENAME = "0_final_output"
USER_PREPROCESSED_PATTERN = "preprocessed_pattern.svg"
CURRENT_PATTERN_TYPE = PatternType.Simple
CURRENT_SIDE = SideType.OneSided
NUM_CLASSIC_LINES = 3 # Number of lines for the classic pattern, only used in the classic case
LINE_THICKNESS_AND_EXTENSION = 4
CLASSIC_INDICES_LINE_DELETE_LIST = []
CLASSIC_PATTERN_SNAP_POINTS = []
CLASSIC_PATTERN_CLASSIC_LINES = [] 
SYMMETRY_LINE = None
DEGREE_LINES_TOP = None
DEGREE_LINES_BOT = None
CLASSIC_CELLS = [] # A list of the classic cells on the canvas
ENABLE_CONSTRAINTS = True # constraints enabling
CELL_ADJACENCY_MAP = None # A map of adjacent cells


"""Functions to get the global values"""

def getMargin():
    return MARGIN

def getFileStepCounter():
    return FILE_STEP_COUNTER

def getDrawingSquareSize():
    return DRAWING_SQUARE_SIZE

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

def getNumClassicLines():
    return NUM_CLASSIC_LINES

def getLineThicknessAndExtension():
    return LINE_THICKNESS_AND_EXTENSION

def getClassicIndicesLineDeleteList():
    return CLASSIC_INDICES_LINE_DELETE_LIST

def getClassicPatternSnapPoints():
    return CLASSIC_PATTERN_SNAP_POINTS

def getClassicPatternClassicLines():
    return CLASSIC_PATTERN_CLASSIC_LINES

def getSymmetryLine():
    return SYMMETRY_LINE

def getDegreeLinesTop():
    return DEGREE_LINES_TOP[0], DEGREE_LINES_TOP[1]

def getDegreeLinesBot():
    return DEGREE_LINES_BOT[0], DEGREE_LINES_BOT[1]

def getClassicCells():
    return CLASSIC_CELLS

def getEnableConstraints():
    return ENABLE_CONSTRAINTS

def getCellAdjacencyMap():
    return CELL_ADJACENCY_MAP

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

def setNumClassicLines(num):
    global NUM_CLASSIC_LINES
    if isinstance(num, int) and num > 0 and num < 10:
        NUM_CLASSIC_LINES = num
    else:
        print("Error: ", num, "is not a positive integer between 0 and 10")

def setLineThicknessAndExtension(value):
    global LINE_THICKNESS_AND_EXTENSION
    if isinstance(value, int) and value > 0:
        LINE_THICKNESS_AND_EXTENSION = value
    else:
        print("Error: ", value, "is not a positive integer")

def setClassicIndicesLineDeleteList(lst):
    global CLASSIC_INDICES_LINE_DELETE_LIST
    CLASSIC_INDICES_LINE_DELETE_LIST = lst

def setClassicPatternSnapPoints(lst):
    global CLASSIC_PATTERN_SNAP_POINTS
    CLASSIC_PATTERN_SNAP_POINTS = lst

def setClassicPatternClassicLines(lst):
    global CLASSIC_PATTERN_CLASSIC_LINES
    CLASSIC_PATTERN_CLASSIC_LINES = lst

def setSymmetryLine(val):
    global SYMMETRY_LINE
    SYMMETRY_LINE = val

def setDegreeLinesTop(val1, val2):
    global DEGREE_LINES_TOP
    DEGREE_LINES_TOP = (val1, val2)

def setDegreeLinesBot(val1, val2):
    global DEGREE_LINES_BOT
    DEGREE_LINES_BOT = (val1, val2) 

def setClassicCells(lst):
    global CLASSIC_CELLS
    CLASSIC_CELLS = lst

def setEnableConstraints(boolval):
    global ENABLE_CONSTRAINTS
    ENABLE_CONSTRAINTS = boolval

def setCellAdjacencyMap(val):
    global CELL_ADJACENCY_MAP
    CELL_ADJACENCY_MAP = val