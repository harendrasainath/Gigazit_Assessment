# Uncomment the below line to install OpenCV module
# pip install opencv-python

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for array manipulation

# Step 1: Load the image using OpenCV
image_path = 'challenge_image.png'
image = cv2.imread(image_path)  # Read the image

# Step 2: Convert to grayscale (optional)
# Uncomment the next line if you want grayscale values instead of RGB
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Extract pixel values
# Initialize an array to store pixel values
height, width, channels = image.shape  # Get the image dimensions
pixel_values = np.zeros((height, width, channels), dtype=int)  # Create an array to hold the pixel values

# Iterate over each pixel to extract the RGB values
for i in range(height):
    for j in range(width):
        pixel_values[i, j] = image[i, j]

# Step 4: Save the pixel values to a text file
output_file = 'pixel_values_opencv.txt'
# Reshape the array to 2D for saving to text
reshaped_pixel_values = pixel_values.reshape(-1, channels)
np.savetxt(output_file, reshaped_pixel_values, fmt='%d', delimiter=',', header='R,G,B', comments='')

print(f"Pixel values saved to {output_file}")
