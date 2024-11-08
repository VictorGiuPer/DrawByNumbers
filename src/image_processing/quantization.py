import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


plt.ioff()
def quantization(image, num_colors):
    """
    Applies k-means color quantization to reduce the number of colors in an image.
    
    Parameters:
        image (numpy.ndarray): The input image in BGR format.
        num_colors (int): The number of colors to quantize the image to.
        
    Returns:
        numpy.ndarray: The quantized image with reduced colors.
    """
    # Convert to RGB (-> Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image_rgb.reshape(-1, 3)
    pixel_values = np.float32(pixel_values)  # Convert to float for k-means

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixel_values)

    # Get centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Assign new color to pixel based on kmeans
    quantized_pixels = centers[labels]
    quantized_image = quantized_pixels.reshape(image_rgb.shape).astype(np.uint8)
    
    return quantized_image


def display_quantizion(original_image, quantized_image):
    """
    Displays the original image and the quantized image side by side for comparison.
    
    Parameters:
        original_image (numpy.ndarray): The original input image in BGR format.
        quantized_image (numpy.ndarray): The quantized image with reduced colors.
    """
    plt.figure(figsize=(10, 5))
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # Quantized image
    plt.subplot(1, 2, 2)
    plt.imshow(quantized_image)
    plt.title("Quantized Image")
    plt.axis("off")
    
    plt.show()

