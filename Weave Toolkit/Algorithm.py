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
  return np.pad(matrix, ((row_padding,row_padding),(column_padding,column_padding),(0,0)), constant_values=255), column_padding

def createHeartCuts(matrix, margin = 0, line_start = 0, sides='onesided'):
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

    # Draw two inner horizontal lines
    cv.line(matrix, (line_x_start, line_y1), (line_x_end, line_y1), line_color, thickness=2)
    cv.line(matrix, (line_x_start, line_y2), (line_x_end, line_y2), line_color, thickness=2)

    # Draw two outer horizontal lines
    cv.line(matrix, (line_x_start, line_y3), (line_x_end, line_y3), line_color, thickness=2)
    cv.line(matrix, (line_x_start, line_y4), (line_x_end, line_y4), line_color, thickness=2)
    axes = (100, (line_y4 - line_y3) // 2)  # Small width, height matches the gap

    # Arch at the left end
    center = (line_x_start, (line_y3 + line_y4) // 2)  # Middle of the height at the left edge
    cv.ellipse(matrix, center, axes, 0, 90, 270, line_color, thickness=2) # Left-facing arch

    # Does nothing just exists to prevent infinite recursion i.e. it is the recursive stop
    if sides == 'None':
      
      return matrix
    
    # Creates a blank pattern on one side of one half of the heart
    elif sides == 'blank':
      matrix[:] = background_color
      matrix = createHeartCuts(matrix, margin, line_start=line_start, sides='None')
      return matrix
    
    # Creates the pattern on both sides of one half of the heart
    elif sides == 'twosided':
      temp_matrix = np.copy(matrix)
      matrix = np.hstack((matrix, np.flip(matrix, axis=1)))
      temp_matrix = createHeartCuts(temp_matrix, margin, line_start=line_start, sides='blank')
      temp_matrix = np.hstack((temp_matrix, np.flip(temp_matrix, axis=1)))
      final_matrix = np.vstack((matrix, temp_matrix))
      return final_matrix
    
    # Creates the pattern on one side of one half of the heart
    elif sides == 'onesided':
      temp_matrix = np.copy(matrix)
      temp_matrix = createHeartCuts(temp_matrix, margin, line_start=line_start, sides='blank')
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

image = cv.imread("canvas_output.png", cv.IMREAD_UNCHANGED)  # Load as is (including alpha)

matrix = np.array(image)

print(matrix.shape)

padded_array, canvas_extended_width = padArray (matrix, 40, 130)

final_output_array = createHeartCuts(padded_array, 10, canvas_extended_width, 'onesided')

print(padded_array.shape)  # (height, width, channels)

# second input is background color user chooses from MainWindow wip
rgba_image = makeTrans(final_output_array, [255,255,255])

# Save the 3D array as a PNG file
cv.imwrite('output_image.png', rgba_image)






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