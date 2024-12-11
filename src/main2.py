from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
from image_processing.load import ImageProcessor
from image_processing.edge_detector import EdgeDetector
from tools import plot_image, compare_images

# Function to load and process the image
def load_image(image_path: str):
    """
    Loads and processes the image (ensure RGB format, grayscale conversion, resize, blur).
    """
    processor = ImageProcessor(image_path)

    load_dict = {}

    # Image Processing Pipeline
    loaded_image = processor.ensure_rgb_format()
    gray_scale_image = processor.convert_to_grayscale()
    resized_image = processor.resize_image(width=500)

    # Save to output dictionary
    load_dict["loaded_img"] = loaded_image
    load_dict["gray_scale_img"] = gray_scale_image
    load_dict["resized_img"] = resized_image

    return load_dict

# Placeholder functions for the steps
def kmeans_clustering(image, num_clusters=10):
    """
    Perform KMeans clustering on the image to reduce the number of colors.
    """
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    clustered_image = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
    return np.uint8(clustered_image)

def build_facets(image):
    """
    Build facets from clustered image regions.
    """
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

def prune_small_facets(facets, original_image, min_size=1):
    """
    Remove small facets below a certain size and return an ndarray image with colors.
    """
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


def detect_borders(image):
    """
    Detect borders between facets using Canny edge detection.
    """
    # Convert the image to grayscale if it's not already
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Find contours on the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours (borders) on the original image for visualization with dark grey color
    border_image = image.copy()
    cv2.drawContours(border_image, contours, -1, (169, 169, 169), 1)  # Dark grey borders

    return contours, border_image


def segment_borders(image, contours):
    """
    Segment the detected borders into distinct regions based on contours.
    """
    # Create a blank mask to segment the borders
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw filled contours (regions) on the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Perform morphological operations to clean the mask (optional)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Use the mask to segment the regions from the original image
    segmented_image = cv2.bitwise_and(image, image, mask=cleaned_mask)

    # Highlight the borders in a separate color (red)
    image[cleaned_mask == 255] = [0, 0, 255]  # Mark the borders in red

    return segmented_image


def place_labels(image, clustered_image, scale_factor=1.5):
    """
    Place labels for each color region on a white canvas with the final image,
    ensuring labels are inside the regions and the canvas is enlarged.
    """
    # Create a white canvas with an enlarged size
    height, width = image.shape[:2]
    enlarged_width = int(width * scale_factor)
    enlarged_height = int(height * scale_factor)
    labeled_image = np.ones((enlarged_height, enlarged_width, 3), dtype=np.uint8) * 255  # White canvas

    # Create a mapping of color to unique labels
    unique_colors = np.unique(clustered_image.reshape(-1, clustered_image.shape[-1]), axis=0)
    color_to_label = {tuple(color): idx + 1 for idx, color in enumerate(unique_colors)}

    # Iterate through each pixel in the original clustered image
    for y in range(clustered_image.shape[0]):
        for x in range(clustered_image.shape[1]):
            # Get the label corresponding to the color of this pixel
            color = tuple(clustered_image[y, x])  # Convert to a tuple for easy lookup
            label = color_to_label[color]

            # Rescale coordinates to fit enlarged canvas
            new_x = int(x * scale_factor)
            new_y = int(y * scale_factor)

            # Draw the corresponding region's color onto the enlarged canvas
            labeled_image[new_y, new_x] = clustered_image[y, x]

            # Add labels to the region
            if new_x - 10 >= 0 and new_y - 10 >= 0:  # Make sure label is not out of bounds
                cv2.putText(labeled_image, str(label), (new_x - 10, new_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text for label

    return labeled_image

# Main function that coordinates the entire process
def start_application(image_path: str):
    print("Application Started.")

    # Load the image
    print("Loading Image")
    load_dict = load_image(image_path)

    # Preprocessing
    print("KMeans Clustering")
    clustered_image = kmeans_clustering(load_dict["resized_img"])
    plot_image(clustered_image)

    print("Facet Building")
    facets = build_facets(clustered_image)

    print("Small Facet Pruning")
    pruned_image = prune_small_facets(facets, load_dict["resized_img"])
    plot_image(pruned_image)

    print("Border Detection")
    contours, edge_image = detect_borders(pruned_image)
    plot_image(edge_image)

    print("Border Segmentation")
    segmented_image = segment_borders(edge_image, contours)
    plot_image(segmented_image)  # Display the segmented image

    clustered_image_2 = kmeans_clustering(pruned_image)
    print("Label Placement")
    labeled_image = place_labels(segmented_image, clustered_image_2, scale_factor=1.5)
    plot_image(labeled_image)

# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    # Example image paths:
    # image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    image_path = "C:\Victor\DrawByNumbers\TestImages\mickey-mouse-cinderella-castle-1024x683.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine/20240815_172047.jpg"
    start_application(image_path)  # Run the app