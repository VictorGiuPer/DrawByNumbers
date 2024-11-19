import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch



class ImageCompressor:

    def reduce_color_space(self, image: np.ndarray, n_colors: int = 8) -> np.ndarray:
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
    

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        if image is None:
            raise ValueError("No image loaded. Load an image first.")
        
        blurred_image =  cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image
    
    def median_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        if image is None:
            raise ValueError("No image loaded. Load an image first.")
        
        blurred_image = cv2.medianBlur(image, kernel_size)
        return blurred_image
    