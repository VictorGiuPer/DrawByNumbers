import cv2
import numpy as np
from sklearn.cluster import KMeans

class DetailReductor:
    def __init__(self):
        """
        Initializes the DetailReductor with the given image.

        Parameters:
        - image (np.ndarray): The input image in BGR format.
        """
        pass
    
    # WORKS BUT WRONG SIZE
    def brush_paint_merge(self, image: np.ndarray, brush_size: int = 10) -> np.ndarray:
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
        brush_size = 20

        # Scale factor for the display window
        scale_factor = 0.8  # Scale the window to 80% of the original size

        # Resize the image for the display window
        height, width = image.shape[:2]
        scaled_width = int(width * scale_factor)
        scaled_height = int(height * scale_factor)
        scaled_image = cv2.resize(image, (scaled_width, scaled_height))
        scaled_painted_image = cv2.resize(painted_image, (scaled_width, scaled_height))

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_color, scaled_painted_image

            # Map scaled coordinates to original image coordinates
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)

            if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
                if 0 <= orig_x < width and 0 <= orig_y < height:
                    selected_color = image[orig_y, orig_x]  # Capture color from the original image
                    selected_color = tuple(map(int, selected_color))  # Convert np.uint8 to int
                    print("Selected Color:", selected_color)
                    print("Type of selected_color:", type(selected_color))

            if event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_LBUTTONDOWN):
                if 0 <= orig_x < width and 0 <= orig_y < height:
                    # Paint on the scaled image
                    cv2.circle(scaled_painted_image, (x, y), int(brush_size * scale_factor), selected_color, -1)

            # Show the current state of the scaled painted image
            cv2.imshow("Paint with Brush", scaled_painted_image)

        # Create a window and set the mouse callback
        cv2.imshow("Paint with Brush", scaled_painted_image)
        cv2.setMouseCallback("Paint with Brush", mouse_callback)

        # Wait for a key press (press 'Esc' to exit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Resize the painted image back to the original size for further processing
        painted_image = cv2.resize(scaled_painted_image, (width, height))

        return painted_image
    
    # Get Clusters and Centers for Facet Pruning
    def clusters_and_centers(self, image: np.ndarray, n_colors: int = 10, n_colors_select: int = 3):
        """
        Recalculate the cluster centers and labels after manually adding new colors.
        This method will cluster the image into a fixed number of colors using KMeans.

        Parameters:
        - image (np.ndarray): The image with manually added colors.
        - n_colors (int): The number of clusters you want.
        - n_colors_select (int): The number of colors retained prior.

        Returns:
        - centers (np.ndarray): The new cluster centers.
        - labels (np.ndarray): The labels for each pixel indicating which cluster it belongs to.
        """
        # Reshape image to a list of pixels (each pixel is an RGB value)
        # Step 1: Flatten the image to create a list of pixels (RGB values)
        image_flat = image.reshape((-1, 3))

        # Step 2: Apply KMeans clustering to reduce the color palette
        kmeans = KMeans(n_clusters = n_colors + n_colors_select, random_state=42)
        kmeans.fit(image_flat)

        # Step 3: Get the cluster labels (which color each pixel belongs to)
        labels = kmeans.labels_.reshape(image.shape[:2])

        # Step 4: The cluster centers are the new representative colors
        centers = kmeans.cluster_centers_

        # Convert to uint8 (since cluster centers are float values)
        centers = np.round(centers).astype(np.uint8)

        print(f"Number of clusters: {len(centers)}")
        print(f"Cluster centers:\n{centers}")

        return centers, labels

    def color_facet_pruning(self, image: np.ndarray, labels: np.ndarray, cluster_centers: np.ndarray, min_size: int = 100) -> np.ndarray:
        """
        Perform facet pruning directly on the original image using segmentation labels.

        Parameters:
        - image (np.ndarray): Original image (RGB).
        - labels (np.ndarray): 2D array of cluster labels for each pixel.
        - cluster_centers (np.ndarray): Cluster center colors from K-Means.
        - min_size (int): Minimum facet size to retain.

        Returns:
        - pruned_image (np.ndarray): Original image with pruned facets.
        """
        print("Facet Pruning")
        # Step 1: Copy labels to modify during pruning
        pruned_labels = labels.copy()
        # Step 2: Perform connected component analysis on the labels
        unique_labels = np.unique(labels)
        h, w = labels.shape
        for label in unique_labels:
            # Create a binary mask for the current label
            mask = (labels == label).astype(np.uint8)

            # Find connected components
            num_components, components = cv2.connectedComponents(mask)

            # Iterate through each component
            for component in range(1, num_components):  # Ignore background (0)
                # Get the size of the component
                component_mask = (components == component)
                component_size = np.sum(component_mask)

                # If the component is too small, reassign it
                if component_size < min_size:
                    # Find the neighboring labels for the small region
                    dilated = cv2.dilate(component_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                    neighbor_mask = dilated & ~component_mask
                    neighbor_labels = labels[neighbor_mask]

                    # Ensure neighbor_labels is 1D
                    neighbor_labels = neighbor_labels.flatten()

                    # Check if there are valid neighbor labels
                    if neighbor_labels.size > 0:
                        # Reassign to the most common neighboring label
                        new_label = np.bincount(neighbor_labels).argmax()
                        pruned_labels[component_mask] = new_label

        # Step 3: Map the refined labels back to the original colors
        pruned_image = cluster_centers[pruned_labels].astype(np.uint8)

        return pruned_image
    
    # NOT WORKING
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