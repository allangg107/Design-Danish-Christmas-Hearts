#from PIL import Image

import cv2 as cv

import numpy as np

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

def preProcessing(image):
  #image = cv.imread(image_path, cv.IMREAD_UNCHANGED)  # Load as is (including alpha)
  #print(image_path)
  # Converts image to a matrix and rotates the canvas
  #new_image = np.array(image_path)
  #new_image = image
  new_image= rotateImage(image, angle=45)
  
  # Removes the canvas lines and non-draw zones from the image
  matrix = new_image[180:-179,179:-180]

  # Might have to be put outside preprocessing eventually
  padded_array, canvas_extended_width = padArray(matrix, 130, 340)
  show_matrix = np.copy(matrix) 
  show_matrix_padded, canvas_width = padArray(show_matrix, 130, 340)
  return padded_array, show_matrix_padded, canvas_extended_width, canvas_width

def createHeartCuts(matrix, margin = 10, line_start = 0, sides='onesided', lines='both'):
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
      matrix = createHeartCuts(matrix, margin, line_start=line_start, sides='None', lines=lines)
      return matrix
    
    # Creates the pattern on both sides of one half of the heart
    elif sides == 'twosided':
      temp_matrix = np.copy(matrix)
      matrix = np.hstack((matrix, np.flip(matrix, axis=1)))
      temp_matrix = createHeartCuts(temp_matrix, margin, line_start=line_start, sides='blank', lines=lines)
      temp_matrix = np.hstack((temp_matrix, np.flip(temp_matrix, axis=1)))
      final_matrix = np.vstack((matrix, temp_matrix))
      return final_matrix
    
    # Creates the pattern on one side of one half of the heart
    elif sides == 'onesided':
      temp_matrix = np.copy(matrix)
      temp_matrix = createHeartCuts(temp_matrix, margin, line_start=line_start, sides='blank', lines=lines)
      matrix = np.hstack((matrix, np.flip(temp_matrix, axis=1)))
      temp_matrix = np.hstack((temp_matrix, np.flip(temp_matrix, axis=1)))
      final_matrix = np.vstack((matrix, temp_matrix))
      return final_matrix    
       
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
  background_color = (255,255,255)
  height, width, _ = matrix.shape
  line_x_start = line_start  # Left side start
  line_x_end = width  # Right side end
  line_y1 = margin # First line (upper)
  line_y2 = height-margin  # Second line (lower)
  
  # Draw two outer horizontal lines
  cv.line(matrix, (line_x_start, line_y1), (line_x_end, line_y1), line_color, thickness=3)
  cv.line(matrix, (line_x_start, line_y2), (line_x_end, line_y2), line_color, thickness=3)
  axes = (330, (line_y2 - line_y1) // 2)  # Small width, height matches the gap

  # Arch at the left end
  center = (line_x_start, (line_y1 + line_y2) // 2)  # Middle of the height at the left edge
  cv.ellipse(matrix, center, axes, 0, 90, 270, line_color, thickness=3) # Left-facing arch
  
  temp_matrix = np.copy(matrix)
  temp_matrix = createHeartCuts(temp_matrix, margin, line_start, sides='blank', lines='None')
  temp_matrix = rotateImage(temp_matrix, angle= -90)
  #temp_matrix.resize((689,689,3))
  #matrix = rotateImage(matrix, angle=-60)
  matrix = np.pad(matrix, ((200,0),(0,200),(0,0)), constant_values=255)
  #temp_matrix = np.pad(temp_matrix, ((0,0),(0,81),(0,0)), constant_values=255)
  #final_matrix = np.vstack((temp_matrix,np.flip(matrix)))
  h, w, _ = temp_matrix.shape
  y_start, x_start = 50, 130  # Adjust these for placement
  y_end, x_end = min(y_start + h, matrix.shape[0]), min(x_start + w, matrix.shape[1])

  # # Only draw black pixels from temp_matrix
  mask = np.all(temp_matrix == [0, 0, 0], axis=-1)  # Find black pixels
  matrix[y_start:y_end, x_start:x_end][mask[:y_end-y_start, :x_end-x_start]] = [0, 0, 0]
  
  # Returns the heart
  matrix = rotateImage(matrix, angle=-45)
  return  matrix

def mainAlgorithm(img, function = 'create'):
  processed_image, shown_image, processed_canvas_width, shown_image_canvas_width  = preProcessing(img)
  match function:
    case 'create':
      final_output_array = createHeartCuts(processed_image, 31, processed_canvas_width, 'onesided')

      # second input is background color user chooses from MainWindow wip
      rgba_image = makeTrans(final_output_array, [255,255,255])
      cv.imwrite('output_image.png', rgba_image)
    case 'show':
      return showHeart(shown_image, 31, shown_image_canvas_width)         
    case _:
      return 'error'

#print(matrix.shape)


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