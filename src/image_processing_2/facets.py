import numpy as np

class Facets():
    def __init__(self):
        pass
    def build_facets(self, image: np.ndarray) -> np.ndarray:
        """
        Build facets from clustered image regions.
        """
        print("Facet Building")
        # Initialization
        facets = np.zeros_like(image[:, :, 0], dtype=np.int32)  # Facet labels
        label = 1

        # Function to perform flood fill
        def flood_fill(x, y, color):
            stack = [(x, y)]
            # Loop over every pixel
            while stack:
                cx, cy = stack.pop()
                # Check bounds, preexisting label, match with the target color
                if (0 <= cx < image.shape[0] and 0 <= cy < image.shape[1] and
                    facets[cx, cy] == 0 and np.all(image[cx, cy] == color)):
                    facets[cx, cy] = label
                    stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

        # Iterate through all pixels
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                # If not labeled
                if facets[x, y] == 0:
                    flood_fill(x, y, image[x, y])
                    label += 1

        return facets

    def prune_small_facets(self, facets: np.ndarray, original_image: np.ndarray, min_size: int = 1) -> np.ndarray:
        """
        Remove small facets below a certain size and return an ndarray image with colors.
        The small facets will inherit colors from neighboring regions instead of being assigned black.
        """
        print("Small Facet Pruning")

        # Identify large facets
        unique_labels, counts = np.unique(facets, return_counts=True)
        large_labels = unique_labels[counts >= min_size]

        # Create a pruned facets mask, marking small facets as 0
        pruned_facets = np.zeros_like(facets)
        for label in large_labels:
            pruned_facets[facets == label] = label

        # Initialize output image
        pruned_image = np.zeros_like(original_image)

        # Loop over each pixel of a pruned (small facet)
        for x in range(facets.shape[0]):
            for y in range(facets.shape[1]):
                if pruned_facets[x, y] == 0: 
                    # Get the 4 neighbors (find neighbouring large facets with label 1)
                    neighboring_labels = set()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < facets.shape[0] and 0 <= ny < facets.shape[1]:
                            neighbor_label = pruned_facets[nx, ny]
                            # Only consider non-zero labels (non-pruned)
                            if neighbor_label != 0:
                                neighboring_labels.add(neighbor_label)

                    # If there are neighboring large labels
                    if neighboring_labels:
                        # Find the most common neighboring label based on the number of pixels in each region
                        # MAYBE DO THIS BASED ON THE VICINITY
                        most_common_label = max(neighboring_labels, key=lambda label: np.sum(facets == label))
                        # Assign color of the neighboring region to small facet
                        pruned_image[x, y] = np.mean(original_image[facets == most_common_label], axis=0)
                    else:
                        # Fallback to the color of the central pixel
                        pruned_image[x, y] = original_image[x, y]

                else:  # For large facets, use the color from the original image
                    pruned_image[x, y] = np.mean(original_image[facets == pruned_facets[x, y]], axis=0)

        return np.uint8(pruned_image)
