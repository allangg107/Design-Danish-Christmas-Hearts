#from PIL import Image

import cv2 as cv

import numpy as np

import math

from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtGui import QImage, QPainter, QTransform
from PyQt6.QtCore import QSize

# Will be called when the user presses the "Update" button

# The algorithm will be given a PNG image of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting)
# 2. Assembly instructions to guide the user in weaving the output image


#     return matrix
def padArray (matrix, row_padding, column_padding):
    return np.pad(matrix, ((row_padding,row_padding),(column_padding, 0),(0,0)), constant_values=255), column_padding

def upscale_image(image, scale_factor):
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Use INTER_NEAREST to maintain sharp lines
    upscaled = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_NEAREST)

    return upscaled

def downscale_image(image, scale_factor):
    height, width = image.shape[:2]
    new_width = int(width // scale_factor)
    new_height = int(height // scale_factor)

    # Use INTER_NEAREST to maintain sharp lines
    downscaled = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_NEAREST)

    return downscaled

def rotateImage(matrix, angle=-60):
    height, width = matrix.shape[:2]

    # Get the center of the image
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

def find_non_background_colored_rows_columns(image, background_color=(255,255,255)):

    pixel_color = background_color[0]-5
    # Identify colored pixels
    mask = np.all(image > pixel_color, axis=2)

    # Find columns where at least one pixel is NOT the pixel color
    non_background_colored_columns = np.any(~mask, axis=0)

    # Find rows where at least one pixel is NOT the pixecl color
    non_background_colored_rows = np.any(~mask, axis=1)

    # Get indices
    non_background_colored_column_indices = np.where(non_background_colored_columns)[0]
    non_background_colored_row_indices = np.where(non_background_colored_rows)[0]

    return non_background_colored_column_indices, non_background_colored_row_indices

# Helper function for drawing lines in CreateHeartCuts which finds the column_index
# of the first and last pixel of a figure based on the pixels known row locations
def find_non_background_colored_column_pix_in_row_in_img(matrix,row_indicies, background_color=(255,255,255)):
    pixel_color = background_color[0]
    res_list = []
    counter = 0

    for submatrix in row_indicies:
      pix_columns_max = []
      pix_columns_min = []
      for i in submatrix:
          max = 0
          min = len(matrix[0])
          for j in range(len(matrix[0])):
            if matrix[i][j][0] != pixel_color:
              if j > max:
                max = j
              if j < min:
                min = j
          pix_columns_max.append(max)
          pix_columns_min.append(min)
      res_list.append([counter,pix_columns_min[0],np.max(pix_columns_max)])
      counter +=1

    return res_list


def split_matrix_by_non_background_colored_columns(matrix, non_colored_indices):
    # Split indices into groups where consecutive indices are <= 1 apart
    split_indices = np.split(non_colored_indices, np.where(np.diff(non_colored_indices) > 1)[0] + 1)

    # Extract sub-matrices and find their middle column
    sub_matrices = []
    middle_columns = []
    column_indicies = []

    for group in split_indices:
        if len(group) > 0:
            start_col, end_col = group[0], group[-1]  # Get start and end columns
            sub_matrix = matrix[:, start_col:end_col + 1]  # Extract sub-matrix
            middle_col = (start_col + end_col) // 2  # Find middle column

            sub_matrices.append(sub_matrix)
            middle_columns.append((start_col,middle_col,end_col))


    return sub_matrices, middle_columns, column_indicies

def split_matrix_by_non_background_colored_rows(matrix, non_colored_indices):

    # Split indices into groups where consecutive indices are <= 1 apart
    split_indices = np.split(non_colored_indices, np.where(np.diff(non_colored_indices) > 1)[0] + 1)

    # Extract sub-matrices and find their middle row
    sub_matrices = []
    middle_rows = []
    row_indicies = []
    for group in split_indices:
        if len(group) > 0:
            start_row, end_row = group[0], group[-1]  # Get start and end rows
            sub_matrix = matrix[start_row:end_row + 1, :]  # Extract sub-matrix
            middle_row = (start_row + end_row) // 2  # Find middle row

            sub_matrices.append(sub_matrix)
            middle_rows.append((start_row,middle_row,end_row))
            row_indicies.append(group)

    return sub_matrices, middle_rows, row_indicies

def preProcessing(image):
  new_image= rotateImage(image, angle=45)

  # Removes the canvas lines and non-draw zones from the image
  matrix = new_image[180:-180,180:-180]
  show_matrix = np.copy(matrix)
  sym_matrix = rotateImage(matrix, angle=-45)
  return matrix, show_matrix, sym_matrix

def baseStencil(matrix, margin=31, line_start=0, line_color = (0,0,0)):
  height, width, _ = matrix.shape

  # Define inner horizontal lines
  line_y1 = margin * 4 # First line (upper)
  line_y2 = height-margin * 4  # Second line (lower)
  line_x_start = line_start  # Left side start
  line_x_end = width  # Right side end

  # Define outer horizontal lines
  line_y3 = margin # First line (upper)
  line_y4 = height-margin  # Second line (lower)

  # Draw two inner horizontal lines
  cv.line(matrix, (line_x_start, line_y1), (line_x_end, line_y1), line_color, thickness=3)
  cv.line(matrix, (line_x_start, line_y2), (line_x_end, line_y2), line_color, thickness=3)

  # Draw two outer horizontal lines
  cv.line(matrix, (line_x_start, line_y3), (line_x_end, line_y3), line_color, thickness=3)
  cv.line(matrix, (line_x_start, line_y4), (line_x_end, line_y4), line_color, thickness=3)
  axes = (330, (line_y4 - line_y3) // 2)  # Small width, height matches the gap

  # Arch at the left end
  center = (line_x_start, (line_y3 + line_y4) // 2)  # Middle of the height at the left edge
  cv.ellipse(matrix, center, axes, 0, 90, 270, line_color, thickness=3) # Left-facing arch
  return matrix

def symmetryStencil(matrix, non_colored_rows, margin=10, line_start=0, line_color = (0,0,0)):
  # Splits the by non-white rows and saves the figures start, middle and end row as well as the
  # row indicies of the figures highest and lowest pixels
  _, points_list, row_indicies = split_matrix_by_non_background_colored_rows(matrix, non_colored_rows)

  pix_columns = find_non_background_colored_column_pix_in_row_in_img(matrix,row_indicies)
  ## Draws the cutting flaps for the cricut
  height, width, _ = matrix.shape

  # Define outer vertical lines
  line_x1 = margin  # Leftmost vertical line
  line_x2 = width - margin  # Rightmost vertical line

  # Define inner vertical lines
  line_x3 = margin * 4  # Inner left vertical line
  line_x4 = width - margin * 4  # Inner right vertical line

  line_y_start = 0  # Top start
  line_y_end = height - margin * 4  # Bottom end

  # Draws the figure lines based on the amount of present figures in the picture
  for i in range(len(pix_columns)):
    cv.line(matrix, (pix_columns[i][2]-5, line_y_end), (pix_columns[i][2]-5, points_list[i][2]-10), line_color, thickness=3)
    cv.line(matrix, (pix_columns[i][1], line_y_start), (pix_columns[i][1], points_list[i][0]+10), line_color, thickness=3)

  # Draw two inner vertical lines
  cv.line(matrix, (line_x3, line_y_start), (line_x3, line_y_end), line_color, thickness=3)
  cv.line(matrix, (line_x4, line_y_start), (line_x4, line_y_end), line_color, thickness=3)

  # Draw two outer vertical lines
  cv.line(matrix, (line_x1, line_y_start), (line_x1, line_y_end), line_color, thickness=3)
  cv.line(matrix, (line_x2, line_y_start), (line_x2, line_y_end), line_color, thickness=3)

  # Arch at the bottom between the two outer vertical lines
  axes = ((line_x2 - line_x1) // 2, 120)  # Width based on gap, height of arch
  center = ((line_x1 + line_x2) // 2, line_y_end)  # Center at bottom middle

  cv.ellipse(matrix, center, axes, 0, 0, 180, line_color, thickness=3)  # Bottom-facing arch

  return matrix

def createHeartCutoutSimplestpattern(matrix, line_start = 0, sides='onesided', line_color=(0,0,0), background_color=(255,255,255)):
  matrix = baseStencil(matrix, 31, line_start)
  # Creates the pattern on both sides of one half of the heart
  if sides == 'twosided':
    temp_matrix = np.copy(matrix)
    matrix = np.hstack((matrix, np.flip(matrix, axis=1)))
    temp_matrix[:] = background_color
    temp_matrix = baseStencil(temp_matrix, 31, line_start)
    temp_matrix = np.hstack((temp_matrix, np.flip(temp_matrix, axis=1)))
    final_matrix = np.vstack((matrix, temp_matrix))
    return final_matrix

  # Creates the pattern on one side of one half of the heart
  elif sides == 'onesided':
    temp_matrix = np.copy(matrix)
    temp_matrix[:] = background_color
    temp_matrix = baseStencil(temp_matrix, 31, line_start, line_color)
    matrix = np.hstack((matrix, np.flip(temp_matrix, axis=1)))
    temp_matrix = np.hstack((temp_matrix, np.flip(temp_matrix, axis=1)))
    final_matrix = np.vstack((matrix, temp_matrix))
    return final_matrix

def createHeartCutoutSymmetrical(matrix, symmetry='symmetrical', line_color=(0,0,0), background_color=(255,255,255)):
    index_arr, _ = find_non_background_colored_rows_columns(matrix)
    _, middle_m, _= split_matrix_by_non_background_colored_columns(matrix,index_arr)

    if symmetry == 'symmetrical':
      for i in middle_m:
        matrix[:, i[1]:i[2]+1] = background_color

      # rotates and pads the image
      matrix_2 = rotateImage(matrix, angle=45)
      matrix_2 = np.pad(matrix_2, ((0,100),(0, 0),(0,0)), constant_values=255)

      # Finds the non_white_rows for the padded image
      _, non_white_rows = find_non_background_colored_rows_columns(matrix_2)

      # creates the stencil for the cutout
      res_matrix = symmetryStencil(matrix_2, non_white_rows, line_color=line_color)

      # Creates the cutout
      upscaled_img = upscale_image(res_matrix, 5)
      copy_img = np.copy(upscaled_img)
      stacked_img = np.vstack((np.flip(rotateImage(copy_img, angle=180),axis=1), upscaled_img))
      copy_stack = np.copy(stacked_img)

    return np.hstack((copy_stack, stacked_img))

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

def rotate_image_qimage(image, angle=-90):
    transform = QTransform().rotate(angle)  # Rotate by the specified angle
    return image.transformed(transform)

def createFinalHeartDisplay(svg_path):
    # Load the SVG
    renderer = QSvgRenderer(svg_path)

    # Create an image with the desired size
    size = renderer.defaultSize()
    image = QImage(size, QImage.Format.Format_ARGB32)
    image.fill(0)  # Transparent background

    # Render SVG onto the QImage
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()

    # Now you can use the image (e.g., rotate it)
    rotated = rotate_image_qimage(image, angle=-90)
    return rotated




def mainAlgorithmSvg(img, function = 'create'):

  match function:

    case 'show':
        return createFinalHeartDisplay(img)

    case _:
      return 'error'