import cv2
import numpy as np

class Borders():
    def __init__(self):
        pass

    def detect_borders(self, image: np.ndarray) -> tuple[list, np.ndarray]:
        """
        Detect borders between facets using Canny edge detection.
        """
        print("Border Detection")
        # Convert the image to grayscale if it's not already
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply a Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Find contours on the edges image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours (borders) on the original image for visualization with dark grey color
        border_image = image.copy()
        cv2.drawContours(border_image, contours, -1, (169, 169, 169), 1)  # Dark grey borders
        
        # Blank Contour Image
        contour_image = np.zeros_like(edges)
        # Draw the contours with 1-pixel thickness
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

        return contours, border_image, contour_image


    def segment_borders(self, image: np.ndarray, contours: list) -> np.ndarray:
        """
        Segment the detected borders into distinct regions based on contours.
        """
        print("Border Segmentation")
        # Create a blank mask to segment the borders
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw filled contours (regions) on the mask
        cv2.drawContours(mask, contours, -1, 255, thickness=1)

        # Perform morphological operations to clean the mask (optional)
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Use the mask to segment the regions from the original image
        segmented_image = cv2.bitwise_and(image, image, mask=cleaned_mask)

        # Highlight the borders in a separate color (red)
        image[cleaned_mask == 255] = [200, 0, 0]  # Mark the borders in red

        return segmented_image
