# Uncomment the below line to install the dependencies
# pip install torch torchvision numpy matplotlib pillow

import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the transformation for the input image
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(image_path):
    """Load an image from the specified path and return it."""
    image = Image.open(image_path)
    return image

def segment_image(image):
    """Segment the image using the pre-trained DeepLabV3 model and return the segmentation map."""
    input_tensor = transform(image).unsqueeze(0)  # Transform and add batch dimension
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    seg_map = output.argmax(0).byte().cpu().numpy()
    return seg_map

def visualize_segmentation(image, seg_map):
    """Visualize the original image and the segmentation map."""
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(seg_map, cmap='jet')
    plt.title("Segmented Image")
    plt.show()

def save_segmentation(seg_map, output_path):
    """Save the segmentation map to a text file."""
    np.savetxt(output_path, seg_map, fmt='%d')

def main():
    image_path = 'challenge_image.png'
    output_path = 'challenge_image_segmented_pixels.txt'
    
    # Load the image
    image = load_image(image_path)
    
    # Segment the image
    seg_map = segment_image(image)
    
    # Visualize the segmentation
    visualize_segmentation(image, seg_map)
    
    # Save the segmented pixel values
    save_segmentation(seg_map, output_path)
    print(f"Segmented pixel values saved to {output_path}")

if __name__ == "__main__":
    main()
