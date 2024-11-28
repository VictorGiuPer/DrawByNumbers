"""
Image Loading, Saving, and Manipulation.

This module provides the `ImageProcessor` class, which facilitates core image processing tasks 
such as loading, saving, grayscale conversion, resizing, and basic pre-processing.

Main Features:
1. Load images from a file using OpenCV.
2. Convert images to grayscale for further analysis or processing.
3. Ensure images are in RGB format (convert from BGR if needed).
4. Resize images to a specific width, height, or both while maintaining aspect ratio.
5. Save processed images to a specified file path.

Class:
- `ImageProcessor`: A class to encapsulate common image processing operations.

Dependencies:
- OpenCV (`cv2`) for image manipulation.
- NumPy (`np`) for image array handling.
- Matplotlib (`pyplot`) for image visualization (optional use).
- `tkinter` for file dialog support (potentially extensible).
"""

import cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np


class ImageProcessor:
    """
    A class to handle core image processing tasks such as loading, saving, 
    grayscale conversion, resizing, and blurring for pre-processing.
    """
    def __init__(self, image_path: str=None):
        """
        Initialize ImageProcessor. Uses file path to load the image.
        
        Parameters:
        - image_path (str): Path to image file.
        """
        self.image = None
        
        if image_path:
            # Use load_image method
            self.load_image(image_path)

    def load_image(self, image_path: str=None) -> np.ndarray:
        """
        Load an image from a file.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - np.ndarray: The loaded image in BGR format.
        """
        # Use OpenCV to load image
        self.image = cv2.imread(image_path)

        # Raise error if image couldn't be read
        if self.image is None:
            raise ValueError(f"Could not load image at path: {image_path}")
        return self.image
    
    def convert_to_grayscale(self) -> np.ndarray:
        """
        Convert the current image to grayscale.

        Returns:
        - np.ndarray: The grayscale image.
        """
        # Raise error if no image is provided
        if self.image is None:
            raise ValueError("No image to convert. Load an image first.")
        
        # Convert to grayscale with OpenCV
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    
    def ensure_rgb_format(self) -> np.ndarray:
        """
        Ensure the image is in RGB format. Converts from BGR if necessary.

        Returns:
        - np.ndarray: The image in RGB format.
        """
        # Raise error if no image is provided
        if self.image is None:
            raise ValueError("No image loaded. Load an image first.")
        
        # Convert image to RGB with OpenCV
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = rgb_image
        return rgb_image
    
    def resize_image(self, width: int=None, height: int=None) -> np.ndarray:
        """
        Resize the image to the specified width and height, maintaining aspect ratio 
        if only one dimension is provided.

        Parameters:
        - width (int): Desired width. If None, will scale based on height.
        - height (int): Desired height. If None, will scale based on width.

        Returns:
        - np.ndarray: The resized image.
        """
        # Raise error if no image is provided
        if self.image is None:
            raise ValueError("No image loaded. Load an image first.")
        
        (h, w) = self.image.shape[:2]

        # Raise error if neither width nor height is provided
        if width is None and height is None:
            raise ValueError("Either width or height must be specified.")
        
        # Ensure aspect ratio given only one dimension
        if width is None:
            # Input height divided by original image height
            scale =  height / float(h)
            dim = (int(w * scale), height)
        elif height is None:
            # Input width divided by original image width
            scale = width / float(w)
            dim = (width, int(h * scale))
        else:
            dim = (width, height)

        # Resize image with OpenCV
        # INTER_AREA -> averaging pixel values in source image to determine color of new pixel.
        resized_image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
        return resized_image
    
    def save_image(self, output_path: str) -> None:
        """
        Save the current image to a file.

        Parameters:
        - output_path (str): Path to save the image.
        """
        if self.image is None:
            raise ValueError("No image loaded. Load an image first.")
        cv2.imwrite(output_path, self.image)