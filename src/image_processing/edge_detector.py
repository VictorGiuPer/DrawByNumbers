"""
The `EdgeDetector` class provides edge detection using Canny, 
with options for preprocessing (e.g., Gaussian blur). It 
supports overlaying edges on the original image and exporting 
them with a transparent background. The class also includes 
functions for skeletonizing edges.

Functions:
1. `canny_edges`: Detects edges using the Canny algorithm 
    with customizable thresholds and preprocessing.
2. `overlay_edges`: Overlays the detected edges onto the 
    original image in a specified color.

Extra/Unused Functions:
- `segment_foreground_background`
- `sobel_edges`
- `export_edges`
- `skeletonize`

Dependencies:
- `numpy`, `cv2` (OpenCV), and `matplotlib.pyplot`.
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

    # Generate Edges (Canny Algorithm)
    def canny_edges(self, image: np.ndarray, min_val: int = 50, max_val: int = 150, 
                    blur_kernel_size: int = 9, aperture_size: int = 3, L2gradient: bool = False) -> np.ndarray:
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

        # Apply Gaussian blur before edge detection to reduce noise.
        if blur_kernel_size > 0:
            image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
        
        # Apply canny edge detection
        canny_edges = cv2.Canny(image, min_val, max_val, apertureSize=aperture_size, L2gradient=L2gradient)
        
        return canny_edges
    
    # Refine Edges
    def refine_edges(self, edges: np.ndarray, kernel: int = 3, iterations: int = 1) -> np.ndarray:
        """
        Refine edges by closing gaps using morphological operations.
        """
        # Define the structuring element (kernel)
        kernel = np.ones((kernel, kernel), np.uint8)

        # Dilation followed by erosion to close small gaps in the edges
        refined_edges = cv2.dilate(edges, kernel, iterations=iterations)
        refined_edges = cv2.erode(refined_edges, kernel, iterations=iterations)

        return refined_edges

    # Overlay image with edges
    def overlay_edges(self, image: np.ndarray, edges: np.ndarray, edge_color: tuple = (0, 0, 0)) -> np.ndarray:
        """
        Replace the pixels in the original image with the edges in a specified color.

        Parameters:
        - image (np.ndarray): Original RGB image.
        - edges (np.ndarray): Binary edge map (values 0 or 255).
        - edge_color (tuple): Color to replace edges with, in RGB.

        Returns:
        - result_image (np.ndarray): RGB image with edges overlaid, replacing the original pixels.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be an RGB image (3 channels).")
        if len(edges.shape) != 2:
            raise ValueError("Edges must be a 2D binary array.")

        # Create a copy of the original image to avoid modifying it directly
        result_image = image.copy()

        # Find where the edges are (non-zero values in the binary edges)
        edge_mask = edges > 0

        # Replace the pixels in the original image with the edge color
        result_image[edge_mask] = edge_color

        return result_image
     

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

    def export_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Returns only the detected edges with a no background as a numpy array.
        """
            
        # Threshold the edges to get a binary image (255 for edges, 0 for background)
        _, binary_edges = cv2.threshold(edges, 60, 255, cv2.THRESH_BINARY)

        # Skeletonize the binary edge image to thin the edges
        refined_edges = self.skeletonize(binary_edges)

        # Create an RGBA image (4 channels: Red, Green, Blue, Alpha)
        rgba_image = np.zeros((refined_edges.shape[0], refined_edges.shape[1], 4), dtype=np.uint8)

        # Set the RGB channels to black (or any color) for the edges
        rgba_image[:, :, 0] = 0  # Red channel (black for edges)
        rgba_image[:, :, 1] = 0  # Green channel (black for edges)
        rgba_image[:, :, 2] = 0  # Blue channel (black for edges)

        # Set the Alpha channel to the refined edge mask (255 for edges, 0 for background)
        rgba_image[:, :, 3] = refined_edges

        return rgba_image

    def skeletonize(self, binary_edges: np.ndarray) -> np.ndarray:
        """
        Thins edges using skeletonization.

        Parameters:
        - binary_edges (np.ndarray): Binary edge image with edges as white (255) and background as black (0).

        Returns:
        - skeleton (np.ndarray): Skeletonized version of the edge map, where edges are thinned to 1-pixel width.
        """
        # Initialize an empty image for the final skeleton
        skeleton = np.zeros_like(binary_edges)

        # Make a copy of the input edges to work on
        temp = binary_edges.copy()

        # Define the structuring element (3x3 cross-shaped kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            # Erode the binary image to shrink the white regions (edges)
            eroded = cv2.erode(temp, kernel)
            
            # Dilate the eroded image to approximate the original size
            dilated = cv2.dilate(eroded, kernel)
            
            # Subtract the dilated image from the original to find the 'skeleton part'
            skeleton_part = cv2.subtract(temp, dilated)
            
            # Add the skeleton part to the overall skeleton
            skeleton = cv2.bitwise_or(skeleton, skeleton_part)
            
            # Update the temporary image with the eroded version for the next iteration
            temp = eroded.copy()

            # If there are no more white pixels left, stop the loop
            if cv2.countNonZero(temp) == 0:
                break

        return skeleton
    
    def segment_foreground_background(self, image: np.ndarray, threshold: int = 100) -> tuple:
        """
        Segment the foreground and background using a threshold.

        Parameters:
        - image (np.ndarray): Input grayscale image (e.g., edge map or intensity map).
        - threshold (int): Intensity value to separate background and foreground.

        Returns:
        - background_mask (np.ndarray): Binary mask for background.
        - foreground_mask (np.ndarray): Binary mask for foreground.
        """
        if len(image.shape) == 3:
            gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        _, background_mask = cv2.threshold(gray_scale, threshold, 255, cv2.THRESH_BINARY)
        foreground_mask = cv2.bitwise_not(background_mask)
        return background_mask, foreground_mask