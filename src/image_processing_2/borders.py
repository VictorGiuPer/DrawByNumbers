import cv2
import numpy as np

class Borders():
    def __init__(self):
        pass

    def detect_borders(self, image: np.ndarray, sensitivity_low: int = 30, 
                       sensistivity_high: int = 50) -> tuple[list, np.ndarray]:
        """
        Detect borders between facets using Canny edge detection.
        """
        print("Border Detection")
        # Convert the image to grayscale if it's not already
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply a Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform Canny edge detection
        border = cv2.Canny(blurred, threshold1=sensitivity_low, threshold2=sensistivity_high)
        
        return border

    def overlay(self, image: np.ndarray, borders: np.ndarray) -> list[np.ndarray, np.ndarray, np.ndarray]:
        # Find contours on the edges image
        contours, _ = cv2.findContours(borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours (borders) on the original image for visualization with dark grey color
        border_image = image.copy()
        cv2.drawContours(border_image, contours, -1, (169, 0, 0), 1)  # Dark grey borders
        
        # Blank Contour Image
        contour_image = np.zeros_like(borders)
        # Draw the contours with 1-pixel thickness
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

        return border_image, contour_image

    def export_outlines(self, edges: np.ndarray) -> np.ndarray:     
        """
        Returns only the detected outlines with a white background as a numpy array.
        """
        # Threshold the outlines to get a binary image (255 for edges, 0 for background)
        _, outlines = cv2.threshold(edges, 60, 255, cv2.THRESH_BINARY)

        # Skeletonize the binary outlines image to thin the edges
        # outlines = self.skeletonize(outlines)

        # Create a white background image (BGR: [255, 255, 255])
        white_background = np.ones((outlines.shape[0], outlines.shape[1], 3), dtype=np.uint8) * 255

        # Set the edges to black ([0, 0, 0]) on the white background
        white_background[outlines == 255] = [155, 155, 155]  # Black for edges

        return white_background
    
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