"""
This module contains utility functions for plotting images using Matplotlib.

Functions:
- plot_image: Displays a single image with an optional colormap.
- plot_images_side_by_side: Displays up to three images side by side for comparison.

These functions are intended to help visualize image processing steps in a convenient and consistent manner.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2


def plot_image(image: np.ndarray, title: str = "Image", cmap: str = None) -> None:
    """Displays a single image with an optional colormap."""
    # Default to grayscale if single-channel
    if cmap is None and len(image.shape) == 2:
        cmap = 'gray'
    
    # Plot image
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_image_3d(image: np.ndarray, downsample_factor: int = 1, colormap: str = 'viridis'):
    """
    Plot a 3D surface representation of an image.

    Parameters:
    - image (np.ndarray): The input image (grayscale or RGB).
    - downsample_factor (int): Factor to downsample the image for faster plotting.
    - colormap (str): Colormap to use for grayscale images. Ignored for RGB.

    Returns:
    None
    """
    # Downsample the image for faster processing
    if downsample_factor > 1:
        image = image[::downsample_factor, ::downsample_factor]
    
    # Check if the image is grayscale or RGB
    if len(image.shape) == 2:
        Z = image  # Pixel intensities

        # Grid of (X, Y) coordinates corresponding to the image dimensions.
        X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
        
        # Create a figure and add a 3D subplot for plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot and apply colormap
        ax.plot_surface(X, Y, Z, cmap=colormap, edgecolor='none')
        ax.set_title("3D Plot of Grayscale Image")

    elif len(image.shape) == 3 and image.shape[2] == 3:
        Z = np.mean(image, axis=2) # Pixel intensities (RGB channels averages)
        
        # Grid of (X, Y) coordinates corresponding to the image dimensions.
        X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
        
        # Create a figure and add a 3D subplot for plotting.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot and use original RGB colors
        ax.plot_surface(X, Y, Z, facecolors=image / 255, edgecolor='none')
        ax.set_title("3D Plot of RGB Image")
    
    else:
        raise ValueError("Input image must be grayscale or RGB.")


    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Pixel Intensity')
    plt.show()

def compare_images(image1: np.ndarray, image2: np.ndarray, image3: np.ndarray = None, 
                             title1: str = "Image 1", title2: str = "Image 2", title3: str = "Image 3", 
                             cmap1: str = None, cmap2: str = None, cmap3: str = None) -> None:
    """Displays up to three images side by side."""
    
    # Initialize list, titles, and colormaps
    images = [image1, image2]
    titles = [title1, title2]
    cmaps = [cmap1, cmap2]
    
    # Add image3 if provided
    if image3 is not None:
        images.append(image3)
        titles.append(title3)
        cmaps.append(cmap3)

    # Create subplots. Single row, 5x5 inches
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    # Checks if only one image was provided
    if len(images) == 1:
        axes = [axes]
        print("Warning, only one image provided.")

    # Loop over lists
    for idx, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        # Set cmap to gray if image is single channel
        if cmap is None and len(img.shape) == 2:
            cmap = 'gray'
        axes[idx].imshow(img, cmap=cmap)
        axes[idx].set_title(title)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()
