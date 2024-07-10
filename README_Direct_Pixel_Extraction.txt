README for Direct Pixel Extraction Using OpenCV
Overview
This script demonstrates a simple approach to extract pixel values from an image using OpenCV. It reads an image, processes its pixels, and saves the RGB values of each pixel into a text file.

 Requirements
- Python 3.x
- OpenCV
- NumPy

 Installation
To install the required libraries, use the following commands:
“pip install opencv-python
pip install numpy”

Code Explanation
1. Load the Image: The image is loaded using OpenCV's `imread` function.
2. Extract Pixel Values: The script iterates over each pixel to extract its RGB values and stores them in a NumPy array.
3. Save Pixel Values: The RGB values are saved to a text file in a comma-separated format.

 Output
The output file (`pixel_values_opencv.txt`) contains the RGB values of each pixel in the format `R,G,B`.

 Advantages
- Simple and easy to implement.
- Requires minimal dependencies.

 Disadvantages
- Does not provide meaningful segmentation, just raw pixel values.
