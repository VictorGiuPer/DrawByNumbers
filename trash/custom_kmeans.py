    # Custom kmeans color implementation
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

         # Convert user-selected colors to numpy array
        user_colors_np = np.array(user_colors, dtype=np.float32)
        num_fixed_centroids = len(user_colors)

        # Initialize the label array (-1 means unassigned)
        labels = -1 * np.ones(pixels.shape[0], dtype=np.int32)

        # Assign pixels to fixed centroids first
        distances_to_fixed = np.linalg.norm(pixels[:, None] - user_colors_np[None, :], axis=2)
        fixed_labels = np.argmin(distances_to_fixed, axis=1)
        fixed_mask = np.min(distances_to_fixed, axis=1) < tolerance  # Threshold for assigning to fixed centroids

        # Update labels with fixed centroids
        labels[fixed_mask] = fixed_labels[fixed_mask]

        # Extract remaining pixels (not assigned to fixed centroids
        remaining_pixels = pixels[labels == -1]
        
        # Initialize centroids for remaining clusters
        num_remaining_clusters = num_clusters - num_fixed_centroids
        random_indices = np.random.choice(remaining_pixels.shape[0],
                                          num_remaining_clusters,
                                          replace=False)
        remaining_centroids = remaining_pixels[random_indices]
        prev_centroids = remaining_centroids.copy()

        # K-means loop
        for iteration in range(max_iter):
            # Compute distances between each pixel and centroids
            remaining_distances = np.linalg.norm(remaining_pixels[:, None] - remaining_centroids[None, :], axis=2)

            # Assign each pixel to the closest centroid
            remaining_labels = np.argmin(remaining_distances, axis=1)

            # Update centroids for remaining clusters
            for i in range(num_remaining_clusters):
                cluster_points = remaining_pixels[remaining_labels == i]
                if len(cluster_points) > 0:
                    remaining_centroids[i] = cluster_points.mean(axis=0)

            # Check for convergence
            centroid_shift = np.linalg.norm(remaining_centroids - prev_centroids, axis=1).max()
            if centroid_shift < tolerance:
                print(f"Convergence reached after {iteration} iterations.")
                break

            # Update previous centroids
            prev_centroids = remaining_centroids.copy()

        # Ensure that the remaining labels map correctly back to the original pixels
        remaining_labels_mapped = np.zeros(pixels.shape[0], dtype=np.int32)  # Array to store the final label assignments
        remaining_labels_mapped[labels == -1] = remaining_labels  # Update only the unassigned pixels with the remaining labels

        # Now combine fixed labels and the updated labels for remaining pixels
        labels[labels == -1] = remaining_labels_mapped[labels == -1]

        # Create the clustered image
        final_centroids = np.vstack([user_colors_np, remaining_centroids])
        clustered_pixels = final_centroids[labels].astype(np.uint8)
        clustered_image = clustered_pixels.reshape(image.shape)

        return clustered_image
    