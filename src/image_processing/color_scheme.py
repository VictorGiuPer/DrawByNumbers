"""
Applying color schemes to images. Assign a color to each region.
"""
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2lab, lab2rgb


class ColorSchemeCreator:
    def __init__(self):
        """
        A class to handles color scheme selection, color 
        quantization and application to segmented regions.
        """
        pass
    
    # Pick color
    def get_colors(self, image: np.ndarray) -> tuple:
        """
        Opens a window to allow the user to select a color by clicking on the image.

        Parameters:
        - image (np.ndarray): The input image (RGB).

        Returns:
        - tuple: Selected color as (R, G, B).
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

        # Wait for the user to select a color
        print("Click on the image to select a color...")
        cv2.waitKey(0)

        # Return the selected color (from the resized image)
        return tuple(selected_color)
    
    # Adapt color zones with 1 color
    def color_zones(self, image: np.ndarray, selected_color: tuple, strength: int = 10) -> np.ndarray:
        selected_color = self.get_colors(image)
        # Convert the selected color to Lab space
        selected_color_lab = rgb2lab(np.array(selected_color, dtype=np.float32).reshape(1, 1, 3) / 255.0)

        # Maximum perceptual distance in Lab space (approximation)
        max_distance = 100.0  # Lab distances range from 0 to ~100

        # Adapt threshold based on strength percentage
        threshold = (strength / 100) * max_distance
        print(f"Threshold (Perceptual Distance): {threshold}")

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

        return smoothed_image
    
    # Adapt color zones with 2 colors (perceptual)
    def midpoint_perceptual(self, image: np.ndarray, c1: tuple, c2: tuple, strength: int = 10) -> np.ndarray:
        """
        Combines two colors perceptually by calculating their midpoint in Lab color space.
        
        Parameters:
        - c1, c2: Tuples representing RGB colors (R, G, B).
        
        Returns:
        - Tuple representing the midpoint color in RGB.
        """
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
        print(f"Threshold (Perceptual Distance): {threshold}")

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

        return modified_image_rgb

    def kmeans_color_replacement(self, image: np.ndarray, colors: list, 
                                 strength: int = 10, n_colors: int = 10) -> np.ndarray:
        image_lab = rgb2lab(image.astype(np.float32) / 255.0)
        # Maximum perceptual distance in Lab space
        max_distance = 100.0

        # Adapt threshold based on strength percentage
        threshold = (strength / 100) * max_distance
        
        # Create masks for each selected color
        masks = []
        for color in colors:
            # Convert the current color to Lab
            color_lab = rgb2lab(np.array(color, dtype=np.float32).reshape(1, 1, 3) / 255.0)

            # Flatten image Lab for distance calculation
            image_lab_flat = image_lab.reshape((-1, 3))
            color_lab_flat = color_lab.reshape(1, 3)

            # Calculate Euclidean distance in Lab space
            distances = cdist(image_lab_flat, color_lab_flat, metric='euclidean').flatten()

            # Create a mask for pixels within the threshold
            mask = (distances < threshold).reshape(image.shape[0], image.shape[1])
            masks.append(mask)

        # Save masks (pixel locations)
        saved_regions = [image[mask] for mask in masks]

        # Flatten the image for K-Means clustering
        image_flat = image.reshape((-1, 3))

        # Perform K-Means clustering
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(image_flat)
        clustered_flat = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)

        # Reshape back to image dimensions
        clustered_image = clustered_flat.reshape(image.shape)

        # Restore original colors in the masked regions
        for mask, region in zip(masks, saved_regions):
            clustered_image[mask] = region

        return clustered_image

    def box_select(image: np.ndarray) -> tuple:
        """
        Allows the user to draw a rectangle on the image and returns the rectangle's coordinates.

        Parameters:
        - image (np.ndarray): Input image in RGB format.

        Returns:
        - tuple: ((x1, y1), (x2, y2)) coordinates of the rectangle.
        """
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
        print("Drag with the mouse to select a rectangular zone.")
        cv2.waitKey(0)

        if len(box_coords) != 2:
            raise ValueError("Rectangle not defined properly.")

        # Extract and sort rectangle coordinates
        (x1, y1), (x2, y2) = box_coords
        return (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))

    def box_color_replacement(image: np.ndarray, c1: tuple, c2: tuple, 
                              rect: tuple, strength: int = 10) -> np.ndarray:
        """
        Replaces color c1 with c2 inside a specified rectangular zone in the image.

        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - c1 (tuple): The original color (R, G, B) to be replaced.
        - c2 (tuple): The target replacement color (R, G, B).
        - rect (tuple): ((x1, y1), (x2, y2)) coordinates of the rectangle.
        - strength (int): Threshold for similarity to c1 (in percentage).

        Returns:
        - np.ndarray: Modified image with replacements applied inside the defined rectangle.
        """
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

        return modified_image