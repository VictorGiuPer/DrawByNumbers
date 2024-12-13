import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
import time
from scipy.ndimage import label

class ColorEditing():
    def __init__(self):
        pass
    
    # Standard KMeans
    def kmeans_clustering(self, image, num_clusters=10):
        """
        Perform KMeans clustering on the image to reduce the number of colors.
        """
        print("KMeans Clustering")
       # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
        
        # Get the cluster labels for each pixel
        labels = kmeans.labels_.reshape(image.shape[:2])  # Reshape to 2D
        
        # Convert cluster centers to uint8 and assign them back to pixels
        cluster_centers = np.uint8(kmeans.cluster_centers_)
        clustered_image = cluster_centers[labels]
        
        return clustered_image, labels
    
    # Pick color
    def get_colors(self, image: np.ndarray) -> tuple:
        """
        Allows the user to select a color from the input image by clicking on it.

        Opens a window displaying the input image (resized if necessary) 
        and lets the user click on a pixel to select its color. The selected color 
        is returned as an RGB tuple.

        Parameters:
        - image (np.ndarray): Input image in RGB format.

        Returns:
        - tuple: The selected color as (R, G, B).

        Workflow:
        1. Resizes the input image proportionally if its dimensions exceed a predefined maximum 
        (default max dimension is 800 pixels).
        2. Displays the resized image in an interactive OpenCV window.
        3. Captures the color of the pixel clicked by the user and closes the window.

        Notes:
        - The function maps the clicked pixel coordinates from the resized image back to the 
        original image to ensure accurate color selection.
        - The selected color is printed to the console for reference.

        Raises:
        - No exceptions are explicitly raised, but if the user does not click a pixel before closing 
        the window, an empty tuple will be returned.
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
        cv2.waitKey(0)

        # Return the selected color (from the resized image)
        return tuple(selected_color)

    # Custom kmeans algorithm
    def kmeans_color_replacement(self, image: np.ndarray, 
                                 choose: int = 3, 
                                 strength: int = 25, 
                                 colors: int = 10) -> np.ndarray:
        """
        Replace colors in an image using KMeans clustering with selective color preservation.

        This method applies KMeans clustering to reduce the color palette of the image, 
        while preserving regions close to specific selected colors based on perceptual 
        similarity in the Lab color space.

        Parameters:
        - image (np.ndarray): Input image in RGB format.
        - strength (int): Threshold strength (0-100) for preserving colors based on Lab 
        perceptual distance. Higher values preserve more colors. (Default: 10)
        - n_colors (int): Number of clusters for KMeans quantization. (Default: 10)

        Returns:
        - np.ndarray: The resulting image with reduced color space and preserved regions.
        """

        print("Custom Kmeans")
        kmeans_colors = []
        for i in range(choose):
            kmeans_c = self.get_colors(image)
            kmeans_colors.append(kmeans_c)
        
        image_lab = rgb2lab(image.astype(np.float32) / 255.0)
        # Maximum perceptual distance in Lab space
        max_distance = 100.0

        # Adapt threshold based on strength percentage
        threshold = (strength / 100) * max_distance
        
        # Create masks for each selected color
        masks = []
        for color in kmeans_colors:
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
        kmeans = KMeans(n_clusters=colors, random_state=42)
        kmeans.fit(image_flat)
        labels = kmeans.labels_.reshape(image.shape[:2])
        clustered_flat = kmeans.cluster_centers_[labels].astype(np.uint8)

        # Reshape back to image dimensions
        clustered_image = clustered_flat.reshape(image.shape)

        # Restore original colors in the masked regions
        for mask, region in zip(masks, saved_regions):
            clustered_image[mask] = region

        return clustered_image
    
