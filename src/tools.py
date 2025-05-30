"""
This module contains utility functions for plotting images using Matplotlib.

Functions:
- plot_image: Displays a single image with an optional colormap.
- plot_images_side_by_side: Displays up to three images side by side for comparison.

These functions are intended to help visualize image processing steps in a convenient and consistent manner.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans
import time



# Plotting
class PlottingTools():
    def __init__(self):
        pass

    def plot_image(self, image: np.ndarray, title: str = "Image", cmap: str = None) -> None:
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

    def plot_image_3d(self, image: np.ndarray, downsample_factor: int = 1, colormap: str = 'viridis'):
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

    def compare_images(self, image1: np.ndarray, image2: np.ndarray, image3: np.ndarray = None, 
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

# General image processing
class GeneralTools():
    def __init__(self):
        pass

    def resize_image(self, image: np.ndarray, increase: int = 10) -> np.ndarray:
        height, width = image.shape[:2]
        new_height = height + (increase / 100) * height
        new_width = width + (increase / 100) * width
        new_dim = (int(new_width), int(new_height))
        resized = cv2.resize(image, new_dim)
        return resized

    def blur_image(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        blurred_image =  cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image

    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        Save the current image to a file.

        Parameters:
        - image (np.ndarray): Image
        - output_path (str): Path to save the image.
        """
        if image is None:
            raise ValueError("No image loaded. Load an image first.")
        cv2.imwrite(output_path, image)

# Color editing tools

class ColorTools:
    # Initialize
    def __init__(self):
        """
        A class to handles color scheme selection, color 
        quantization and application to segmented regions.
        """
        pass
    
    # Pick color
    def get_colors(self, image: np.ndarray) -> tuple:
        """
        Allows the user to select a color from the input image by clicking on it.

        Opens a window displaying the input image (resized if necessary) 
        and lets the user click on a pixel to select its color. The selected color 
        is returned as an RGB tuple.

        Parameters:
        - image (np.ndarray): Input image in RGB format.

        Returns:
        - tuple: The selected color as (R, G, B).

        Workflow:
        1. Resizes the input image proportionally if its dimensions exceed a predefined maximum 
        (default max dimension is 800 pixels).
        2. Displays the resized image in an interactive OpenCV window.
        3. Captures the color of the pixel clicked by the user and closes the window.

        Notes:
        - The function maps the clicked pixel coordinates from the resized image back to the 
        original image to ensure accurate color selection.
        - The selected color is printed to the console for reference.

        Raises:
        - No exceptions are explicitly raised, but if the user does not click a pixel before closing 
        the window, an empty tuple will be returned.
        """
        selected_color = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_color
            if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
                # Map the clicked coordinates back to the original image
                original_x = int(x / scale)
                original_y = int(y / scale)

                # Get the color from the original image
                selected_color = image[original_y, original_x]
                print(f"Selected Color: {selected_color}")  # Print to console

                # Close the window after selection
                cv2.destroyAllWindows()

        # Convert image to BGR for OpenCV display
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize image to fit screen (scale to 80% of original size for example)
        height, width = image_bgr.shape[:2]
        max_dim = 800  # Max dimension for resizing (adjust this as needed)

        # Determine scaling factor
        scale = max_dim / max(height, width)

        # Resize image keeping the aspect ratio
        new_dim = (int(width * scale), int(height * scale))
        resized_image = cv2.resize(image_bgr, new_dim)

        # Create a window and set the mouse callback
        cv2.imshow("Select a Color", resized_image)
        cv2.setMouseCallback("Select a Color", mouse_callback)
        cv2.waitKey(0)

        # Return the selected color (from the resized image)
        return tuple(selected_color)
    
    # Adapt color zones with 1 color
    def refine_threshold(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Refines color zones in the image by adapting areas near a selected color based on perceptual 
        similarity in Lab color space.

        This function smooths color zones by replacing pixels near a selected color with the exact 
        selected color, based on a strength threshold that determines the perceptual similarity.

        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - strength (int, optional): Percentage threshold (0-100) for similarity to the selected color.
        Higher values increase the range of pixels affected. Default is 10.

        Returns:
        - np.ndarray: Modified image with smoothed color zones in RGB format.

        Process:
        1. Prompts the user to select a color from the image.
        2. Converts the selected color and the input image to Lab color space for perceptual calculations.
        3. Calculates Euclidean distances between each pixel in the image and the selected color in Lab space.
        4. Identifies pixels within the threshold distance to the selected color.
        5. Replaces those pixels with the selected color in the output image.

        Raises:
        - ValueError: If no color is selected.

        Notes:
        - The strength parameter controls the sensitivity of the color replacement process. A lower 
        strength value results in fewer pixels being replaced, while a higher value expands the range.
        """
        print("Threshold Refining")
        # Select Color
        selected_color = self.get_colors(image)
        while selected_color:
            # If no colors are selected (empty list), exit the loop early.
            if len(selected_color) == 0:
                raise ValueError("No Color has been selected.")
            # Convert the selected color to Lab space
            selected_color_lab = rgb2lab(np.array(selected_color, dtype=np.float32).reshape(1, 1, 3) / 255.0)

            # Maximum perceptual distance in Lab space (approximation)
            max_distance = 100.0  # Lab distances range from 0 to ~100

            # Adapt threshold based on strength percentage
            threshold = (strength / 100) * max_distance

            # Convert the input image to Lab space
            image_lab = rgb2lab(image.astype(np.float32) / 255.0)

            # Flatten the image and selected color arrays for distance calculation
            image_lab_flat = image_lab.reshape((-1, 3))
            selected_color_lab_flat = selected_color_lab.reshape(1, 3)

            # Calculate Euclidean distances from the selected color for all pixels in the image (in Lab space)
            distance = cdist(image_lab_flat.astype(np.float32),
                            selected_color_lab_flat.astype(np.float32),
                            metric='euclidean').flatten()

            # Create a mask for pixels that are within the threshold distance to the selected color
            mask = distance < threshold

            # Reshape the mask back to the image shape (height, width)
            mask = mask.reshape(image.shape[0], image.shape[1])

            # Create a copy of the original image to modify
            smoothed_image = image.copy()

            # Apply the selected color to the pixels that match the mask
            smoothed_image[mask] = selected_color
            image = smoothed_image
            selected_color = self.get_colors(image)

        return image
    
    # Adapt color zones with 2 colors (perceptual)
    def midpoint_perceptual(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Refines the image by combining two selected colors perceptually, replacing nearby pixels 
        with their midpoint in Lab color space.

        This function calculates the perceptual midpoint between two selected colors (c1, c2) in Lab 
        space and modifies the image by replacing pixels near these colors with the calculated midpoint 
        color, based on a strength threshold.

        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - strength (int, optional): Percentage threshold (0-100) determining the perceptual distance 
        within which pixels are considered close to the selected colors. Default is 10.

        Returns:
        - np.ndarray: Modified image in RGB format with refined colors.

        Process:
        1. Converts the selected colors (c1, c2) from RGB to Lab color space.
        2. Calculates their midpoint in Lab space and converts it back to RGB.
        3. Converts the entire image to Lab space and computes perceptual distances to c1 and c2.
        4. Identifies pixels within the threshold distance from either color.
        5. Replaces matching pixels in the image with the perceptual midpoint color.
        
        Notes:
        - The function ensures all modified pixel values remain within valid RGB range (0-255).
        """
        print("Midpoint Refining")
        c1 = self.get_colors(image)
        c2 = self.get_colors(image)
        while c1 and c2:
            # Convert RGB colors to Lab (scaling RGB values to 0-1 for Lab conversion)
            c1_lab = rgb2lab(np.array(c1, dtype=np.float32).reshape(1, 1, 3) / 255.0)
            c2_lab = rgb2lab(np.array(c2, dtype=np.float32).reshape(1, 1, 3) / 255.0)
            
            # Calculate lab midpoint and convert to rgb
            midpoint_lab = (c1_lab + c2_lab) / 2
            midpoint_rgb = tuple(lab2rgb(midpoint_lab).squeeze() * 255)
            print(f"Midpoint Color (RGB): {midpoint_rgb}")

            # Calculate threshold based on strength percentage
            max_distance = 100
            threshold = max_distance * (strength / 100)

            # Convert image to Lab space for perceptual distance calculations
            image_lab =  rgb2lab(image.astype(np.float32) / 255.00)

            # Calculate distances from both selected colors in lab space
            c1_lab_flat = c1_lab.reshape(1, 3)
            c2_lab_flat = c2_lab.reshape(1, 3)
            image_lab_flat = image_lab.reshape((-1, 3))
            distances_c1 = cdist(image_lab_flat, c1_lab_flat, metric="euclidean").flatten()
            distances_c2 = cdist(image_lab_flat, c2_lab_flat, metric="euclidean").flatten()

            # Create a mask for pixels within the threshold for either selected color
            mask = (distances_c1 < threshold) | (distances_c2 < threshold)
            
            # Modify the image: set all matching pixels to the perceptual midpoint
            modified_lab = image_lab_flat.copy()
            modified_lab[mask] = midpoint_lab.flatten()

            modified_image_lab = modified_lab.reshape(image_lab.shape)
            modified_image_rgb = lab2rgb(modified_image_lab) * 255

            # Ensure the values are in valid range and convert to uint8
            modified_image_rgb = np.clip(modified_image_rgb, 0, 255).astype(np.uint8)
            image = modified_image_rgb
            c1 = self.get_colors(image)
            c2 = self.get_colors(image)

        return image

    # Select Box For box_color_replacement
    def box_select(self, image: np.ndarray) -> tuple:
        """
        Enables the user to interactively select a rectangular region in the image by clicking and dragging with the mouse.

        The user clicks and drags on the displayed image to define the rectangle's coordinates. The function then captures 
        the top-left and bottom-right corners of the selected rectangle.

        Parameters:
        - image (np.ndarray): Input image in RGB format, displayed for user interaction.

        Returns:
        - tuple: Coordinates of the selected rectangle in the format 
        ((x1, y1), (x2, y2)), where:
            - (x1, y1): Top-left corner.
            - (x2, y2): Bottom-right corner.
        
        Raises:
        - ValueError: If the rectangle is not properly defined (e.g., if the user fails to complete the rectangle).
        """

        print("Box Selection") 
        box_coords = []

        def draw_rectangle(event, x, y, flags, param):
            nonlocal box_coords
            if event == cv2.EVENT_LBUTTONDOWN:
                # Start point of the rectangle
                box_coords = [(x, y)]
            elif event == cv2.EVENT_LBUTTONUP:
                # End point of the rectangle
                box_coords.append((x, y))
                cv2.destroyAllWindows()

        # Display the image for user interaction
        temp_image = image.copy()
        cv2.imshow("Draw a rectangle (drag with mouse)", cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR))
        cv2.setMouseCallback("Draw a rectangle (drag with mouse)", draw_rectangle)
        cv2.waitKey(0)

        if len(box_coords) != 2:
            raise ValueError("Rectangle not defined properly.")

        # Extract and sort rectangle coordinates
        (x1, y1), (x2, y2) = box_coords
        return (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))

    # Refine color zones: Manual color replacement
    def box_color_replacement(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Replace a selected color with another within a user-defined rectangular region.

        This function allows the user to select a rectangular area in an image and replaces 
        all pixels similar to a specified source color (`c1`) with a target color (`c2`) 
        based on a perceptual similarity threshold.

        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - strength (int): Threshold for color similarity (0-100). Higher values allow a 
        broader range of colors similar to `c1` to be replaced. (Default: 10)

        Returns:
        - np.ndarray: The modified image with the selected color replaced within the specified rectangle.
        """
        print("Box Color Replacement") 
        c1 = self.get_colors(image)
        c2 = self.get_colors(image)
        while c1 and c2:
            time.sleep(1)
            rect = self.box_select(image)
            (x1, y1), (x2, y2) = rect

            # Crop the defined rectangle
            region = image[y1:y2, x1:x2]

            # Calculate the threshold in RGB space
            max_distance = 441.67  # Max possible Euclidean distance in RGB
            threshold = (strength / 100) * max_distance

            # Flatten the region for easier color comparison
            region_flat = region.reshape((-1, 3))

            # Calculate the Euclidean distance from c1
            distances = cdist(region_flat.astype(np.float32), np.array([c1], dtype=np.float32)).flatten()

            # Create a mask for pixels matching the color within the threshold
            mask = distances < threshold
            mask = mask.reshape(region.shape[:2])

            # Replace the matching pixels with c2
            region[mask] = c2

            # Put the modified region back into the original image
            modified_image = image.copy()
            modified_image[y1:y2, x1:x2] = region
            image = modified_image
            c1 = self.get_colors(image)
            c2 = self.get_colors(image)

        return modified_image

    # Reduce detail in final image
    def localized_pruning(self, image: np.ndarray, n_colors: int = 5) -> np.ndarray:
        """
        Allows the user to select a region of the image for localized pruning 
        to reduce detail in that area.
        
        Args:
            image (np.ndarray): The input image.
            n_colors (int): Number of colors to reduce the selected region to.
            
        Returns:
            np.ndarray: The updated image with reduced detail in the selected area.
        """
        # Allow the user to select a region of interest (ROI)
        print("Select the region you want to prune and press ENTER.")
        roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

        if roi == (0, 0, 0, 0):  # If no ROI is selected
            print("No region selected. Returning the original image.")
            return image

        x, y, w, h = map(int, roi)
        roi_image = image[y:y+h, x:x+w]

        # Reshape the ROI for clustering
        pixels = roi_image.reshape(-1, 3)

        # Apply KMeans clustering to reduce detail
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
        clustered_pixels = kmeans.cluster_centers_[kmeans.labels_].reshape(roi_image.shape)
        clustered_pixels = np.uint8(clustered_pixels)

        # Replace the ROI in the original image with the clustered result
        pruned_image = image.copy()
        pruned_image[y:y+h, x:x+w] = clustered_pixels

        return pruned_image

