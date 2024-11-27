"""
Applying color schemes to images. Assign a color to each region.
"""
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans


class ColorSchemeCreator:
    
    def __init__(self):
        """
        A class to handles color scheme selection, color 
        quantization and application to segmented regions.
        """
        pass

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
                # Get the color at the clicked pixel
                selected_color = image[y, x]
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
    
    def reduce_color_space_2(self, image: np.ndarray, n_colors: int = 10) -> np.ndarray:
        """
        Reduce the color space of the image using MiniBatchKMeans.
        
        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - n_colors (int): Number of colors to quantize the image to.

        Returns:
        - compressed_image (np.ndarray): Image with reduced color space.
        """

        # Reshape the image to a 2D array of pixels (N x 3)
        pixel_data = image.reshape((-1, 3))

        # Apply MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=0)
        labels = kmeans.fit_predict(pixel_data)
        centers = kmeans.cluster_centers_

        # Convert centers to integers (0-255 range)
        centers = np.uint8(centers)

        # Map each pixel to the nearest cluster center
        quantized_image = centers[labels]
        compressed_image = quantized_image.reshape(image.shape)

        return compressed_image

    def get_kmeans_colors(self, image: np.ndarray, num_colors: int = 3) -> tuple:
        """
        Opens a window to allow the user to select multiple colors by clicking on the image.

        Parameters:
        - image (np.ndarray): The input image (RGB).
        - num_colors (int): Number of colors the user can select.

        Returns:
        - list[tuple]: List of selected colors as (R, G, B) tuples.
        """
        selected_colors = []

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_colors
            if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
                # Get the color at the clicked pixel
                color = image[y, x].tolist()  # Convert to list for tuple conversion later
                selected_colors.append(tuple(color))
                print(f"Selected Color {len(selected_colors)}: {color}")

                # Draw a marker on the image at the selected point
                cv2.circle(resized_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select a Color", resized_image)

                # Close the window if the required number of colors has been selected
                if len(selected_colors) >= num_colors:
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

        # Wait for the user to select colors
        print(f"Click on the image to select {num_colors} colors...")
        cv2.waitKey(0)

        # If fewer than the required number of colors were selected
        if len(selected_colors) < num_colors:
            raise ValueError(f"Only {len(selected_colors)} colors were selected. Please select {num_colors}.")

        return selected_colors

    def custom_kmeans(self, image: np.ndarray, user_colors: list[tuple],
                       num_clusters: int = 10, max_iter: int = 100, 
                       tolerance: float = 1e-4) -> np.ndarray:
        """
        Implements K-means clustering with user-selected colors as fixed initial centroids.

        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - user_colors (list[tuple]): List of user-selected colors as (R, G, B) tuples.
        - num_clusters (int): Total number of clusters.
        - max_iter (int): Maximum number of iterations for convergence.
        - tol (float): Tolerance for centroid movement to determine convergence.

        Returns:
        - clustered_image (np.ndarray): Image where each pixel is replaced by its cluster's centroid.
        """
        if len(user_colors) >= num_clusters:
            raise ValueError("Number of user-selected colors must be less than the total number of clusters.")
        
        # Flatten the image to a (num_pixels, 3) array
        pixels = image.reshape((-1, 3)).astype(np.float32)

        # Initialize centroids: user-selection and random selection
        user_colors_np = np.array(user_colors, dtype=np.float32)
        random_indices = np.random.choice(pixels.shape[0], 
                                          num_clusters - len(user_colors), 
                                          replace=False)
        initial_centroids = np.vstack([user_colors_np, pixels[random_indices]])
        
        # Debug print statement
        print(initial_centroids)

        centroids = initial_centroids.copy()
        prev_centroids = np.zeros_like(centroids)
        labels = np.zeros(pixels.shape[0], dtype=np.int32)


        # K-means loop
        for iteration in range(max_iter):
            # Compute distances between each pixel and centroids
            distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)

            # Assign each pixel to the closest centroid
            labels = np.argmin(distances, axis=1)

            # Update centroids with mean of assigned pixels
            for i in range(num_clusters):
                cluster_points = pixels[labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = cluster_points.mean(axis=0)
            
            # Check for convergence (movement below tolerance)
            centroid_shift = np.linalg.norm(centroids - prev_centroids, axis=1)
            if centroid_shift < tolerance:
                print(f"Convergence reached after {iteration} iterations.")
                break

            prev_centroids = centroids.copy()
            clustered_pixels = centroids[labels].astype(np.uint8)
            clustered_image = clustered_pixels.reshape(image.shape)

            return clustered_image