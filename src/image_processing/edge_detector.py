"""
Edge detection logic.
"""
import numpy as np
import cv2


class EdgeDetector:
    """
    A class to detect edges in an image 
    using Canny or Sobel edge detection.
    """
    def __init__(self):
        """
        Initialize EdgeDetector class.
        """
        pass

    def canny_edges(self, image: np.ndarray, min_val: int = 50, max_val: int = 150) -> np.ndarray:
        """
        Detect edges in the image using the Canny algorithm.

        Parameters:
        - image (np.ndarray): Input image in grayscale.
        - min_val (int): Minimum threshold for the Canny algorithm.
        - max_val (int): Maximum threshold for the Canny algorithm.

        Returns:
        - canny_edges (np.ndarray): Image with edges detected by the Canny algorithm.
        """

        # Ensure image is in grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply canny edge detection
        canny_edges = cv2.Canny(image, min_val, max_val)
        return canny_edges
     
    def sobel_edges(self, image: np.ndarray, ksize: int=3) -> np.ndarray:
        """
        Apply Sobel edge detection to the image.

        Parameters:
        - image (np.ndarray): Input image in grayscale.
        - ksize (int): Kernel size for the Sobel operator. Must be an odd number.

        Returns:
        - edges (np.ndarray): Image with edges detected by the Sobel algorithm.
        """

        # Ensure image is in grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply sobel edge detection
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize, scale=2)
        gray_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize, scale=2)
        sobel_edges = cv2.magnitude(grad_x, gray_y)

        # Convert to 8-bit image
        sobel_edges =  cv2.convertScaleAbs(sobel_edges)
        return sobel_edges

    def high_pass_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply a high pass filter to enhance edges in the image.

        Parameters:
        - image (np.ndarray): Input image (grayscale or BGR).
        - kernel_size (int): Size of the kernel for the Gaussian blur (must be odd).

        Returns:
        - np.ndarray: High pass filtered image, emphasizing edges.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to the image (low pass filter)
        kernel_tuple = (kernel_size, kernel_size)
        blurred = cv2.GaussianBlur(image, kernel_tuple, 0)

        # Subtract the blurred image from the original to get the high pass result
        high_pass_filter = cv2.subtract(image, blurred)

        return high_pass_filter


        
        