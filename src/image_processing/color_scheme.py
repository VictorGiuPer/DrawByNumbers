"""
Applying color schemes to images. Assign a color to each region.
"""
import cv2
import numpy as np
from scipy.spatial.distance import cdist


class ColorSchemeCreator:
    
    def __init__(self):
        """
        A class to handles color scheme selection, color 
        quantization and application to segmented regions.
        """
        pass

    def select_color_from_image(self, image: np.ndarray) -> tuple:
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
                # Get the color at the clicked pixel
                selected_color = image[y, x]
                print(f"Selected Color: {selected_color}")  # Print to console

                # Close the window after selection
                cv2.destroyAllWindows()

        # Convert image to BGR for OpenCV display
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create a window and set the mouse callback
        cv2.imshow("Select a Color", image_bgr)
        cv2.setMouseCallback("Select a Color", mouse_callback)

        # Wait for the user to select a color
        print("Click on the image to select a color...")
        cv2.waitKey(0)

        # If no color was selected (user closed window without clicking)
        if selected_color is None:
            raise ValueError("No color was selected.")
        
        selection = tuple(selected_color)
        return selection
    
    def color_zones(self, image: np.ndarray, selected_color: tuple, strength: int) -> np.ndarray:
        
        # Maximum Euclidean distance for RGB images (approximation)
        max_distance = 441.67

        # Adapt threshold based on strength percentage
        threshold = (strength / 100) * max_distance
        
        # Flatten the image to (num_pixels, 3) for easier distance calculation
        image_flat = image.reshape((-1, 3))
        
        # Convert the selected color to a 2D array (1, 3)
        selected_color_array = np.array(selected_color).reshape(1, 3)

        # Calculate the Euclidean distances from the selected color for all pixels in the image
        distance = cdist(image_flat.astype(np.float32), 
                         selected_color_array.astype(np.float32), 
                         metric='euclidean').flatten()

        # Create a mask for pixels that are within the threshold distance to the selected color
        mask = distance < threshold

        # Reshape the mask back to the image shape (height, width)
        mask = mask.reshape(image.shape[0], image.shape[1])

        # Create a copy of the original image to apply the color smoothing
        smoothed_image = image.copy()

        # Apply the selected color to the pixels that match the mask
        smoothed_image[mask] = selected_color

        return smoothed_image