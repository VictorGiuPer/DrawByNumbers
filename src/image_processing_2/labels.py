import numpy as np
import cv2



class Labels():
    def __init__(self):
        pass

    def white_canvas(self, image: np.ndarray, scale_factor: float):
        # Get original image dimensions
        height, width = image.shape[:2]

        # Scale dimensions
        enlarged_width = int(width * scale_factor)
        enlarged_height = int(height * scale_factor)

        # Create a blank white canvas
        canvas = np.ones((enlarged_height, enlarged_width, 3), dtype=np.uint8) * 255
        return canvas

    def color_label_map(self, clustered_image: np.ndarray) -> dict:
        """
        Create a mapping from unique colors in the clustered image to unique labels.
        """
        # Find unique colors in the clustered image
        unique_colors = np.unique(clustered_image.reshape(-1, clustered_image.shape[-1]), axis=0)
        # Map each unique color to a unique label starting from 1
        color_map = {tuple(color): idx + 1 for idx, color in enumerate(unique_colors)}
        print(color_map)
        return color_map

    def get_label_positions(self, clustered_image: np.ndarray, color_map: dict) -> dict:
        """
        Get the positions (centers) for each color region based on the color map.
        """
        label_positions = {}  # This needs to be a dictionary

        for color, label in color_map.items():
            # Create a binary mask for the current color region
            mask = np.all(clustered_image == color, axis=-1)
            
            # Find the coordinates of the region where the mask is True
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue  # Skip if no region is found for this color

            # Compute the center of the region by averaging coordinates
            center_y, center_x = coords.mean(axis=0).astype(int)

            # Store the label and its position (as a tuple) in the dictionary
            label_positions[label] = (center_x, center_y)
        
        print("Label Positions:", label_positions)  # Debugging output
        return label_positions

    def place_labels(self, image: np.ndarray, label_positions: dict) -> np.ndarray:
        """
        Place labels on the original image at the specified positions.
        """
        print(label_positions)
        # Copy the original image for labeling
        labeled_image = image.copy()

        # Step 1: Draw the labels on the image
        for label, (x, y) in label_positions.items():
            # Place the label at the computed center of the color region
            cv2.putText(labeled_image, str(label), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)  # Grey color text
        
        return labeled_image

    def process_image(self, image: np.ndarray, clustered_image: np.ndarray) -> np.ndarray:
        """
        Main function to process the image and return a labeled version.
        """
        # Step 1: Create a mapping of color to unique labels
        color_map = self.color_label_map(clustered_image)
        print(color_map)

        # Step 2: Get positions for each label (based on color regions)
        label_positions = self.get_label_positions(clustered_image, color_map)
        print(label_positions)

        # Step 3: Place labels on the original image
        labeled_image = self.place_labels(image, label_positions)

        return labeled_image