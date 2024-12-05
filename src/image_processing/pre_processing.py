"""
Image Preprocessing for Color Reduction and Smoothing.

This module provides the `Preprocessor` class, which facilitates basic image preprocessing 
tasks such as reducing the color space using KMeans clustering and applying Gaussian blur 
to smooth images.

Key Features:
1. **Initial Color Reduction**:
   - Reduces the number of colors in an image using MiniBatchKMeans clustering.
   - Useful for simplifying images before further processing or analysis.

2. **Image Smoothing with Gaussian Blur**:
   - Applies Gaussian blur to reduce noise and smooth the image.
   - Helps in pre-processing tasks like edge detection or segmentation.

Class:
- `Preprocessor`: Encapsulates methods for color quantization and image blurring.

Dependencies:
- OpenCV (`cv2`) for image manipulation.
- NumPy (`np`) for array operations.
- Scikit-learn (`MiniBatchKMeans`) for clustering algorithms.
- PyTorch (`torch`) is imported but currently unused in this implementation.

Functions:
- `initial_kmeans`: Reduces the color space of the image.
- `gaussian_blur`: Applies Gaussian blur to smooth the image.
"""


import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch



class Preprocessor:

    # Initial Color Reduction
    def initial_kmeans(self, image: np.ndarray, n_colors: int = 8) -> np.ndarray:
        """
        Reduce the color space of the image using MiniBatchKMeans.
        
        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - n_colors (int): Number of colors to quantize the image to.

        Returns:
        - compressed_image (np.ndarray): Image with reduced color space.
        """

        # Reshape the image to a 2D array of pixels (N x 3)
        pixel_data = image.reshape((-1, 3))

        # Apply MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=0)
        labels = kmeans.fit_predict(pixel_data)
        centers = kmeans.cluster_centers_

        # Convert centers to integers (0-255 range)
        centers = np.uint8(centers)

        # Map each pixel to the nearest cluster center
        quantized_image = centers[labels]
        compressed_image = quantized_image.reshape(image.shape)

        return compressed_image
    
    # Apply Blur To Image
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Reduce the detail in the image by introducing gaussian blur.
        
        Parameters:
        - image (np.ndarray): Input image.
        - kernel_size (int): Blur intensity.

        Returns:
        - blurred_image (np.ndarray): Image with gaussian blur.
        """
        if image is None:
            raise ValueError("No image loaded. Load an image first.")
        
        blurred_image =  cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image