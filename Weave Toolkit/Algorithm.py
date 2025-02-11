#from PIL import Image

import cv2 as cv

import numpy as np

# Will be called when the user presses the "Update" button

# The algorithm will be given a PNG image of the desired pattern as input and return 2 things:
# 1. A PNG image which shows the weaving pattern (to be used by CriCut for laser cutting)
# 2. Assembly instructions to guide the user in weaving the output image

def drawLines(matrix):
    line_color = [0,0,0]
    for i in range(3,8):
        matrix[i,:] = line_color
        matrix[-i,:] = line_color

    return matrix

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

padded_array = np.pad(matrix, ((6,6),(1,1),(0,0)), constant_values=255)

final_output_array = drawLines(padded_array)

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