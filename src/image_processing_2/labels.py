import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont


class Labels():
    def __init__(self):
        pass

    # KMeans Implementation
    def kmeans_clustering(self, image: np.ndarray, num_clusters=10) -> np.ndarray:
        """
        Perform KMeans clustering on the image to reduce the number of colors.
        """
        print("KMeans Clustering")
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
        clustered_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
        return np.uint8(clustered_image)

    def get_region(self, clustered_image: np.ndarray, x: int, 
                   y: int, color: tuple, visited: np.ndarray) -> list:
        """
        Find all pixels of the same color in a region starting from (x, y).
        This version considers neighbors' neighbors to ensure we get the whole region.
        """
        region = []  # List to store the coordinates of all pixels in the same color region
        stack = [(x, y)]  # Stack for DFS, starting with the given pixel

        while stack:
            cx, cy = stack.pop()  # Get the current pixel coordinates
            if visited[cy, cx]:
                continue  # Skip if this pixel has already been visited
            if tuple(clustered_image[cy, cx]) == color:
                visited[cy, cx] = True  # Mark the pixel as visited
                region.append((cx, cy))  # Add the pixel to the region

                # Check neighboring pixels (all 8 directions)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        # Skip the current pixel itself
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        # Ensure the neighbor is within bounds and hasn't been visited yet
                        if 0 <= nx < clustered_image.shape[1] and 0 <= ny < clustered_image.shape[0] and not visited[ny, nx]:
                            stack.append((nx, ny))  # Add the neighbor to the stack for further exploration
        return region


    def place_labels(self, image: np.ndarray, outline_image: np.ndarray, 
                     clustered_image: np.ndarray, min_size: int = 60) -> np.ndarray:
        """
        Place labels on the image for each distinct region of color, 
        only labeling regions that are above the specified size threshold.
        """
        visited = np.zeros(clustered_image.shape[:2], dtype=bool)
        labeled_image = image.copy()
        labeled_template = np.uint8(outline_image.copy())

        color_to_label = {}  # Map colors to labels
        label = 1  # Start label from 1

        # Convert the OpenCV images to PIL for text rendering
        pil_labeled_image = Image.fromarray(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
        pil_labeled_template = Image.fromarray(cv2.cvtColor(labeled_template, cv2.COLOR_BGR2RGB))

        # Create drawing objects for PIL
        draw_labeled_image = ImageDraw.Draw(pil_labeled_image)
        draw_labeled_template = ImageDraw.Draw(pil_labeled_template)

        # Attempt to use a custom font (ensure the font path is correct for your OS)
        try:
            font_path = "C:/Windows/Fonts/Arial.ttf"
            font = ImageFont.truetype(font_path, 10)
        except IOError:
            print("default")
            font = ImageFont.load_default()  # Fallback to default if font is not found

        for y in range(clustered_image.shape[0]):
            for x in range(clustered_image.shape[1]):
                if not visited[y, x]:  # If the pixel hasn't been visited
                    color = tuple(clustered_image[y, x])  # Get the color at this pixel

                    # Find the entire region of this color (flood fill)
                    region = self.get_region(clustered_image, x, y, color, visited)

                    # Only label regions larger than min_size
                    if len(region) >= min_size:
                        # If this color hasn't been assigned a label, do so
                        if color not in color_to_label:
                            color_to_label[color] = label
                            label += 1  # Increment label for the next color
                        
                        # Label all pixels of this region with the same label
                        region_label = color_to_label[color]

                        # Calculate the center of the region for label placement
                        center_x = int(np.mean([px[0] for px in region]))
                        center_y = int(np.mean([px[1] for px in region]))

                        # Calculate text size for correct positioning (if needed)
                        text = str(region_label)

                        # Use textbbox to get text bounding box for positioning
                        bbox = draw_labeled_image.textbbox((center_x, center_y), text, font=font)
                        text_width = bbox[2] - bbox[0]  # width of the text
                        text_height = bbox[3] - bbox[1]  # height of the text

                        text_position = (center_x - text_width // 2, center_y - text_height // 2)

                        # Draw the label on the PIL images
                        draw_labeled_image.text(text_position, text, font=font, fill=(0, 0, 0))
                        draw_labeled_template.text(text_position, text, font=font, fill=(0, 0, 0))

        # Convert back to OpenCV format
        labeled_image = cv2.cvtColor(np.array(pil_labeled_image), cv2.COLOR_RGB2BGR)
        labeled_template = cv2.cvtColor(np.array(pil_labeled_template), cv2.COLOR_RGB2BGR)

        return labeled_image, labeled_template


    def labelling(self, image: np.ndarray, outline_image: np.ndarray, 
                  n_colors: int = 10, min_size = 100) -> np.ndarray:
        """
        Main function to process the image and return a labeled version.
        """
        # Apply KMeans clustering to get a clustered image
        clustered_image = self.kmeans_clustering(image, n_colors)
        # Place labels on the image based on color regions
        labeled_image, labeled_template = self.place_labels(image, outline_image, clustered_image, min_size=min_size)
        return labeled_image, labeled_template
    








    def place_labels_r(self, image: np.ndarray, outline_image: np.ndarray, 
                     clustered_image: np.ndarray, min_size: int = 80) -> np.ndarray:
        """
        Place labels on the image for each distinct region of color, 
        only labeling regions that are above the specified size threshold.
        """
        visited = np.zeros(clustered_image.shape[:2], dtype=bool)
        labeled_image = image.copy()
        labeled_template = np.uint8(outline_image.copy())

        color_to_label = {}  # Map colors to labels
        label = 1  # Start label from 1

        for y in range(clustered_image.shape[0]):
            for x in range(clustered_image.shape[1]):
                if not visited[y, x]:  # If the pixel hasn't been visited
                    color = tuple(clustered_image[y, x])  # Get the color at this pixel

                    # Find the entire region of this color (flood fill)
                    region = self.get_region(clustered_image, x, y, color, visited)

                    # Only label regions larger than min_size
                    if len(region) >= min_size:
                        # If this color hasn't been assigned a label, do so
                        if color not in color_to_label:
                            color_to_label[color] = label
                            label += 1  # Increment label for the next color
                        
                        # Label all pixels of this region with the same label
                        region_label = color_to_label[color]

                        # Calculate the center of the region for label placement
                        center_x = int(np.mean([px[0] for px in region]))
                        center_y = int(np.mean([px[1] for px in region]))

                        # Draw the label at the center of the region
                        cv2.putText(labeled_image, str(region_label), (center_x, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(labeled_template, str(region_label), (center_x, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)

        return labeled_image, labeled_template