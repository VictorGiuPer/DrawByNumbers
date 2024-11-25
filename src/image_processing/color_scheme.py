"""
Applying color schemes to images. Assign a color to each region.
"""
import cv2
import numpy as np

class ColorScheme:
    
    def __init__(self, num_colors=8, custom_palette=None):
        """
        A class to handles color scheme selection, color 
        quantization and application to segmented regions.
        """
    def select_color_from_image(self, image: np.ndarray):
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
        if not selected_color:
            raise ValueError("No color was selected.")

        return tuple(selected_color[::-1])  # Convert BGR to RGB