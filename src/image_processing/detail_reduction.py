import cv2
import numpy as np

class DetailReductor:
    def __init__(self):
        """
        Initializes the DetailReductor with the given image.

        Parameters:
        - image (np.ndarray): The input image in BGR format.
        """
        pass
    def brush_paint_merge(self, image: np.ndarray, brush_size: int = 100) -> np.ndarray:
        """
        Allows the user to paint over an image with a selected color, merging regions as they paint.

        Parameters:
        - image (np.ndarray): The input image in RGB format.
        - brush_size (int, optional): The size of the brush (default is 20).

        Returns:
        - np.ndarray: The modified image with painted regions.
        """
        painted_image = image.copy()

        selected_color = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_color, painted_image
            if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
                selected_color = image[y, x]  # Capture the color under the brush
                selected_color = tuple(selected_color)
            print("Selected Color:", selected_color)
            print("Type of selected_color:", type(selected_color))


            if event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_LBUTTONDOWN):
                # Draw a circle (or any shape) with the selected color
                cv2.circle(painted_image, (x, y), brush_size, tuple(selected_color), -1)

            # Show the current state of the image
            cv2.imshow("Paint with Brush", painted_image)

        # Create a window and set the mouse callback
        cv2.imshow("Paint with Brush", painted_image)
        cv2.setMouseCallback("Paint with Brush", mouse_callback)

        # Wait for a key press (press 'Esc' to exit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return painted_image
    
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
    
    def zone_merge(self, image: np.ndarray, min_size: int = 50) -> np.ndarray:
        """
        Merge small connected regions in the selected area based on a minimum size threshold.

        Parameters:
        - image (np.ndarray): The input image to process.
        - min_size (int): Minimum size of the regions to retain, small regions will be merged.

        Returns:
        - np.ndarray: The processed image with merged regions.
        """
        # Step 1: Select the region using the box_select method
        (x1, y1), (x2, y2) = self.box_select(image)
        print(f"Selected region: {(x1, y1)} to {(x2, y2)}")

        # Step 2: Extract the selected region from the image
        selected_region = image[y1:y2, x1:x2]

        # Step 3: Convert to grayscale and detect edges (Canny edge detection)
        gray_image = cv2.cvtColor(selected_region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)

        # Step 4: Find connected components in the selected area
        num_labels, labels = cv2.connectedComponents(edges)

        # Step 5: Create a new label map where small regions are merged
        new_labels = np.zeros_like(labels)
        next_label = 1  # Start labeling from 1 (0 is background)

        for label in range(1, num_labels):
            component_mask = (labels == label)
            component_size = np.sum(component_mask)

            if component_size < min_size:
                # Find the neighboring labels and merge with the largest neighboring region
                dilated_mask = cv2.dilate(component_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                neighbor_labels = labels[dilated_mask > 0]

                # Find the most common neighboring label
                if neighbor_labels.size > 0:
                    most_common_label = np.bincount(neighbor_labels[neighbor_labels > 0]).argmax()
                    new_labels[component_mask] = most_common_label
            else:
                new_labels[component_mask] = next_label
                next_label += 1

        # Step 6: Rebuild the edges image using the new labels (merged regions)
        merged_edges = np.zeros_like(edges)
        for label in range(1, next_label):
            merged_edges[new_labels == label] = 255  # Assign merged label region to white (or any other color)

        # Step 7: Convert merged region back to RGB (since we only processed the edges region in grayscale)
        merged_region_rgb = cv2.cvtColor(merged_edges, cv2.COLOR_GRAY2RGB)

        # Step 8: Replace the merged region back into the original image (keeping the RGB color)
        image[y1:y2, x1:x2] = merged_region_rgb

        return image