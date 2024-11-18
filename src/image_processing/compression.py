import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch



class ImageCompressor:

    @staticmethod
    def reduce_color_space(self, image: np.ndarray, n_colors: int = 8) -> np.ndarray:
        """
        Reduce the color space of the image using MiniBatchKMeans.
        
        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - n_colors (int): Number of colors to quantize the image to.

        Returns:
        - compressed_image (np.ndarray): Image with reduced color space.
        """


        h, w, c = image.shape

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
    
    @staticmethod
    def reduce_color_space_gpu(image: np.ndarray, n_colors: int = 8) -> np.ndarray:
        """
        Reduce the color space using GPU-accelerated k-means with PyTorch.

        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - n_colors (int): Number of colors to quantize the image to.

        Returns:
        - compressed_image (np.ndarray): Image with reduced color space.
        """
        # Reshape and convert image to PyTorch tensor
        pixel_data = torch.tensor(image.reshape((-1, 3)), dtype=torch.float32, device='cuda')

        # Randomly initialize cluster centers
        centers = pixel_data[torch.randperm(pixel_data.size(0))[:n_colors]]

        for _ in range(10):  # Number of iterations
            # Compute distances and assign clusters
            distances = torch.cdist(pixel_data, centers)
            labels = torch.argmin(distances, dim=1)

            # Update centers
            for i in range(n_colors):
                centers[i] = pixel_data[labels == i].mean(dim=0)

        # Map each pixel to its cluster center
        quantized_image = centers[labels].cpu().numpy().astype(np.uint8)
        compressed_image = quantized_image.reshape(image.shape)

        return compressed_image