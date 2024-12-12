import numpy as np

class Facets():
    def __init__(self):
        pass
    def build_facets(self, image: np.ndarray) -> np.ndarray:
        """
        Build facets from clustered image regions.
        """
        print("Facet Building")
        facets = np.zeros_like(image[:, :, 0], dtype=np.int32)  # Facet labels
        label = 1

        # Function to perform flood fill
        def flood_fill(x, y, color):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if (0 <= cx < image.shape[0] and 0 <= cy < image.shape[1] and
                    facets[cx, cy] == 0 and np.all(image[cx, cy] == color)):
                    facets[cx, cy] = label
                    stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

        # Iterate through all pixels
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if facets[x, y] == 0:  # Not yet labeled
                    flood_fill(x, y, image[x, y])
                    label += 1

        return facets

    def prune_small_facets(self, facets: np.ndarray, original_image: np.ndarray, min_size: int = 1) -> np.ndarray:
        """
        Remove small facets below a certain size and return an ndarray image with colors.
        """
        print("Small Facet Pruning")
        unique_labels, counts = np.unique(facets, return_counts=True)
        large_labels = unique_labels[counts >= min_size]

        # Create a pruned facets mask
        pruned_facets = np.zeros_like(facets)
        for label in large_labels:
            pruned_facets[facets == label] = label

        # Generate an output image with colors corresponding to pruned facets
        pruned_image = np.zeros_like(original_image)
        for label in large_labels:
            pruned_image[pruned_facets == label] = np.mean(original_image[facets == label], axis=0)

        return np.uint8(pruned_image)