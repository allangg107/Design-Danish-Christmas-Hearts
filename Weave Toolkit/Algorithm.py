#from PIL import Image

import cv2 as cv

import numpy as np

import math

# Will be called when the user presses the "Update" button

# The algorithm will be given a PNG image of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting)
# 2. Assembly instructions to guide the user in weaving the output image

# def createHeartCuts(matrix):
#     line_color = [0,0,0]
#     for i in range(3,8):
#         matrix[i,:] = line_color
#         matrix[-i,:] = line_color

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

def find_non_white_rows_columns(image):

    # Identify white pixels (assumed threshold: 250 in all channels)
    white_mask = np.all(image > 250, axis=2)

    # Find columns where at least one pixel is NOT white
    non_white_columns = np.any(~white_mask, axis=0)
    
    # Find rows where at least one pixel is NOT white
    non_white_rows = np.any(~white_mask, axis=1)

    # Get indices
    non_white_column_indices = np.where(non_white_columns)[0]
    non_white_row_indices = np.where(non_white_rows)[0]

    return non_white_column_indices, non_white_row_indices

# Helper function for drawing lines in CreateHeartCuts which finds the column_index 
# of the first and last pixel of a figure based on the pixels known row locations
def find_non_white_column_pix_in_row_in_img(matrix,row_indicies):
    
    res_list = []
    counter = 0
    
    for submatrix in row_indicies:
      pix_columns_max = []
      pix_columns_min = []
      for i in submatrix:
          max = 0
          min = len(matrix[0])
          for j in range(len(matrix[0])):
            if matrix[i][j][0] != 255:
              if j > max:
                max = j
              if j < min:
                min = j
          pix_columns_max.append(max)
          pix_columns_min.append(min)
      res_list.append([counter,pix_columns_min[0],np.max(pix_columns_max)])
      counter +=1 

    return res_list
      

def split_matrix_by_non_white_columns(matrix, non_white_indices):
   
    # Split indices into groups where consecutive indices are <= 1 apart
    split_indices = np.split(non_white_indices, np.where(np.diff(non_white_indices) > 1)[0] + 1)

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
            

    return sub_matrices, middle_columns

def split_matrix_by_non_white_rows(matrix, non_white_indices):

    # Split indices into groups where consecutive indices are <= 1 apart
    split_indices = np.split(non_white_indices, np.where(np.diff(non_white_indices) > 1)[0] + 1)

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
  matrix = new_image[180:-179,179:-180]
  
  # Might have to be put outside preprocessing eventually
  padded_array, canvas_extended_width = padArray(matrix, 130, 340)
  show_matrix = np.copy(matrix) 
  show_matrix_padded, canvas_width = padArray(show_matrix, 130, 340)
  return padded_array, show_matrix_padded, canvas_extended_width, canvas_width

def preprocessingSymmetry(image):
  #image_path = cv.imread(image_path, cv.IMREAD_UNCHANGED)  # Load as is (including alpha)
  
  # Converts image to a matrix and rotates the canvas
  #image = np.array(image_path)
  new_image = rotateImage(image, angle=45)
  matrix = new_image[180:-179,179:-180]
  new_matrix = rotateImage(matrix, angle=-45)
  return new_matrix

def createHeartCutsChild(matrix, margin = 10, line_start = 0, sides='onesided', lines='both'):
    line_color = (0, 0, 0)  # Black color
    background_color = (255,255,255)
    height, width, _ = matrix.shape
      
    # Define inner horizontal lines
    line_y1 = margin * 4 # First line (upper)
    line_y2 = height-margin * 4  # Second line (lower)
    line_x_start = line_start  # Left side start
    line_x_end = width  # Right side end

    # Define outer horizontal lines
    line_y3 = margin # First line (upper)
    line_y4 = height-margin  # Second line (lower)

    if lines=='both':
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

    # Does nothing just exists to prevent infinite recursion i.e. it is the recursive stop
    if sides == 'None':
      
      return matrix
    
    # Creates a blank pattern on one side of one half of the heart
    elif sides == 'blank':
      matrix[:] = background_color
      matrix = createHeartCutsChild(matrix, margin, line_start=line_start, sides='None', lines=lines)
      return matrix
    
    # Creates the pattern on both sides of one half of the heart
    elif sides == 'twosided':
      temp_matrix = np.copy(matrix)
      matrix = np.hstack((matrix, np.flip(matrix, axis=1)))
      temp_matrix = createHeartCutsChild(temp_matrix, margin, line_start=line_start, sides='blank', lines=lines)
      temp_matrix = np.hstack((temp_matrix, np.flip(temp_matrix, axis=1)))
      final_matrix = np.vstack((matrix, temp_matrix))
      return final_matrix
    
    # Creates the pattern on one side of one half of the heart
    elif sides == 'onesided':
      temp_matrix = np.copy(matrix)
      temp_matrix = createHeartCutsChild(temp_matrix, margin, line_start=line_start, sides='blank', lines=lines)
      matrix = np.hstack((matrix, np.flip(temp_matrix, axis=1)))
      temp_matrix = np.hstack((temp_matrix, np.flip(temp_matrix, axis=1)))
      final_matrix = np.vstack((matrix, temp_matrix))
      return final_matrix   

def createHeartCuts(matrix, margin = 10, sides='blank', symmetry='symmetrical'):
    line_color = (0, 0, 0)  # Black color
    background_color = (255,255,255)
    
    index_arr, _ = find_non_white_rows_columns(matrix)
    _, middle_m = split_matrix_by_non_white_columns(matrix,index_arr)

    if symmetry == 'symmetrical': 
      for i in middle_m: 
        matrix[:, i[1]:i[2]+1] = background_color

      # rotates and pads the image
      matrix_2 = rotateImage(matrix, angle=45)
      #cv.imwrite('test2.png', matrix_2)
      matrix_2 = np.pad(matrix_2, ((0,100),(0, 0),(0,0)), constant_values=255)
      
      # Finds the non_white_rows for the padded image
      _, non_white_rows = find_non_white_rows_columns(matrix_2)
     
      _, points_list, row_indicies = split_matrix_by_non_white_rows(matrix_2, non_white_rows)

      pix_columns = find_non_white_column_pix_in_row_in_img(matrix_2,row_indicies)

      ## Draws the cutting flaps for the cricut 
      height, width, _ = matrix_2.shape
      
      # Define outer vertical lines
      line_x1 = margin  # Leftmost vertical line
      line_x2 = width - margin  # Rightmost vertical line

      # Define inner vertical lines
      line_x3 = margin * 4  # Inner left vertical line
      line_x4 = width - margin * 4  # Inner right vertical line

      line_y_start = 0  # Top start
      line_y_end = height - margin * 4  # Bottom end

      # Draws the figure lines baserd on the amount of present figures in th4e picture
      for i in range(len(pix_columns)):
        cv.line(matrix_2, (pix_columns[i][2]-5, line_y_end), (pix_columns[i][2]-5, points_list[i][2]-10), line_color, thickness=3)
        cv.line(matrix_2, (pix_columns[i][1], line_y_start), (pix_columns[i][1], points_list[i][0]+10), line_color, thickness=3)
          
    # Draw two inner vertical lines
      cv.line(matrix_2, (line_x3, line_y_start), (line_x3, line_y_end), line_color, thickness=3)
      cv.line(matrix_2, (line_x4, line_y_start), (line_x4, line_y_end), line_color, thickness=3)

      # Draw two outer vertical lines
      cv.line(matrix_2, (line_x1, line_y_start), (line_x1, line_y_end), line_color, thickness=3)
      cv.line(matrix_2, (line_x2, line_y_start), (line_x2, line_y_end), line_color, thickness=3)

      # Arch at the bottom between the two outer vertical lines
      axes = ((line_x2 - line_x1) // 2, 120)  # Width based on gap, height of arch
      center = ((line_x1 + line_x2) // 2, line_y_end)  # Center at bottom middle

      cv.ellipse(matrix_2, center, axes, 0, 0, 180, line_color, thickness=3)  # Bottom-facing arch

    # Creates the cutout
    upscaled_img = upscale_image(matrix_2, 5)
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

def showHeart(matrix, margin=10, line_start = 0): 
    line_color = (0, 0, 0)  # Black color
    height, width, _ = matrix.shape
    square_size = min(width, height) // 2  # Ensuring a balanced square
    center = (square_size, square_size)

    # Create a blank mask
    mask = np.full_like(matrix, 255)

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

    # Draw a circle at the center of the square
    point1 = [center[0] - half_size, center[1]]
    point2 = [center[0], center[1] + half_size]

    square_width = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    radius = math.floor(square_width // 2)
    halfway_point_upper_left = [(center[0] - half_size + center[0]) // 2, (center[1] + center[1] - half_size) //2]
    halfway_point_upper_right = [(center[0] + center[0] + half_size) // 2, (center[1] - half_size + center[1]) //2]
    cv.ellipse(mask, halfway_point_upper_left, (radius, radius), 0, -225, -45, line_color, thickness=3)
    cv.ellipse(mask, halfway_point_upper_right, (radius, radius), 0, -135, 45, line_color, 3)

    return mask

def mainAlgorithm(img, function = 'create'):
  
  match function:
    case 'create':
      processed_image, shown_image, processed_canvas_width, shown_image_canvas_width  = preProcessing(img)
      final_output_array = createHeartCutsChild(processed_image, 31, processed_canvas_width, 'onesided')

      # second input is background color user chooses from MainWindow wip
      rgba_image = makeTrans(final_output_array, [255,255,255])
      cv.imwrite('output_image.png', rgba_image)

    case 'create_symmetry':
      processed_image = preprocessingSymmetry(img)
      final_output_array = createHeartCuts(processed_image)   
      #rgba_image = makeTrans(final_output_array, [255,255,255])
      cv.imwrite('output_image.png', final_output_array)      

    case 'show':
      return showHeart(shown_image, 31, shown_image_canvas_width)         
     
    case _:
      return 'error'

#print (preprocessing2ElectricBoogaloo('canvas_output.png'))
# a_m = preprocessing2ElectricBoogaloo('canvas_output.png')
# print(a_m.shape)
# index_arr, _ = find_non_white_rows_columns(a_m)

# kk, km = split_matrix_by_non_white_columns(a_m, index_arr)
# #print(f"Number of sub-matrices: {len(kk)}")
# #print(km[0][0])
# #print(a_m.shape)

# createHeartCuts(a_m,km)

# Save the 3D array as a PNG file

#cv.imwrite('image.png', test_matrix)
#cv.imshow('image', test_matrix)






# Define the background color (white in this case)
#background_color = [255, 255, 255, 255]  # White background with full opacity
#
## Create a mask where the background is white
#mask = np.all(image[:, :, :3] == background_color[:3], axis=2)
#
## Set alpha (transparency) to 0 where the mask is true
#image[mask] = [0, 0, 0, 0]  # Set BGRA values to transparent
#
## Save the result
#cv.imwrite('output_image.png', image)
# Pad the array with 1 row on top, 1 row at the bottom, 2 columns on the left, and 2 columns on the right