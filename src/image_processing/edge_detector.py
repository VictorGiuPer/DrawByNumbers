"""
Edge detection logic.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


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

    def canny_edges(self, image: np.ndarray, min_val: int = 50, max_val: int = 150, 
                    blur_kernel_size: int = 5, aperture_size: int = 3, L2gradient: bool = False) -> np.ndarray:
        """
        Detect edges in the image using the Canny algorithm with optional preprocessing.

        Parameters:
        - image (np.ndarray): Input image in grayscale.
        - min_val (int): Minimum threshold for the Canny algorithm.
        - max_val (int): Maximum threshold for the Canny algorithm.
        - blur_kernel_size (int): Size of the kernel used for Gaussian blur to reduce noise before edge detection.
        - aperture_size (int): Aperture size for the Sobel operator in the Canny edge detection.
        - L2gradient (bool): Whether to use the L2 norm for gradient calculation (default is False, uses L1 norm).

        Returns:
        - canny_edges (np.ndarray): Image with edges detected by the Canny algorithm.
        """

        # Ensure image is in grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur before edge detection to reduce noise (optional)
        if blur_kernel_size > 0:
            image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
        
        # Apply canny edge detection
        canny_edges = cv2.Canny(image, min_val, max_val, apertureSize=aperture_size, L2gradient=L2gradient)
        
        return canny_edges
     
    def sobel_edges(self, image: np.ndarray, ksize: int = 3, scale: float = 1, delta: float = 0, 
                    sobel_type: str = 'both', normalize: bool = False) -> np.ndarray:
        """
        Apply Sobel edge detection to the image.

        Parameters:
        - image (np.ndarray): Input image in grayscale.
        - ksize (int): Kernel size for the Sobel operator. Must be an odd number. (Default is 3)
        - scale (float): Scaling factor for the Sobel operator. (Default is 1)
        - delta (float): Offset added to the result of the Sobel operator. (Default is 0)
        - sobel_type (str): Type of Sobel edge detection. 'x' for horizontal, 'y' for vertical, or 'both' for combined.
        - normalize (bool): Whether to normalize the result to the range [0, 255]. (Default is False)

        Returns:
        - edges (np.ndarray): Image with edges detected by the Sobel algorithm.
        """
        
        # Ensure image is in grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Sobel edge detection based on sobel_type
        if sobel_type == 'x' or sobel_type == 'both':
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
        else:
            grad_x = np.zeros_like(image, dtype=np.float64)
        
        if sobel_type == 'y' or sobel_type == 'both':
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
        else:
            grad_y = np.zeros_like(image, dtype=np.float64)

        # Combine the gradients in both directions if necessary
        sobel_edges = cv2.magnitude(grad_x, grad_y)

        # Convert to 8-bit image
        sobel_edges = cv2.convertScaleAbs(sobel_edges)

        # Normalize the image if requested
        if normalize:
            sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)

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

    def export_edges(self, image: np.ndarray) -> np.ndarray:
            """
            Returns only the detected edges as a numpy array.
            """
            # Apply Sobel edge detection to the image
            edges = self.sobel_edges(image)
            
            # Create a binary mask (only edges, everything else is black)
            _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
            
            # Return the binary edge image as a numpy array
            return binary_edges

        
        