from PIL import Image
import numpy as np
import cv2

from image_processing.load import ImageProcessor
from image_processing_2.color_editing import ColorEditing
from image_processing_2.facets import Facets
from image_processing_2.borders import Borders
from image_processing_2.labels import Labels
from tools import plot_image, compare_images

# Function to load and process the image
def load_image(image_path: str) -> dict:
    """
    Loads and processes the image (ensure RGB format, grayscale conversion, resize, blur).
    """
    print("Loading Image")
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
def KMeans(image: np.ndarray) -> np.ndarray:
    """
    Perform KMeans clustering on the image to reduce the number of colors.
    """ 
    color_editor = ColorEditing()
    # Possibly Replace the colors at the end
    custom_kmeans = color_editor.kmeans_color_replacement(image, choose=0, strength=20, colors=10)
    return custom_kmeans

def create_facets(image: np.ndarray) -> np.ndarray:
    """
    Build and prune facets.
    """
    facet_creator = Facets()
    facets = facet_creator.build_facets(image)
    facet_pruning = facet_creator.prune_small_facets(facets=facets, original_image=image, min_size=1)
    return facet_pruning

def create_borders(image: np.ndarray) -> np.ndarray:
    border_creator = Borders()
    borders, border_image, contour_image = border_creator.detect_borders(image)
    segmented_image = border_creator.segment_borders(border_image, borders)
    return border_image, segmented_image, contour_image

def create_labels(image: np.ndarray) -> np.ndarray:
    # Get Clustered Image NEED TO RUN WITH AMOUNT OF COLORS IN THE IMAGE
    clustered_image = ColorEditing.kmeans_clustering(ColorEditing, image, 10)
    label_creator = Labels()
    label_image = label_creator.place_labels(image, clustered_image)
    return label_image

# Main function that coordinates the entire process
def start_application(image_path: str):
    print("Application Started.")

    # Load the image
    load_dict = load_image(image_path)

    # Initial KMeans
    clustered_image = KMeans(load_dict["resized_img"])
    # plot_image(clustered_image)

    # Facet building and pruning
    pruned_image = create_facets(clustered_image)
    # plot_image(pruned_image)

    # Detecting borders and segmentation
    border_image, segment_image, outline_image = create_borders(pruned_image)
    # compare_images(border_image, segment_image, outline_image, title1="Borders", title2="Segments", title3="Outlines")

    # ADD MORE DETAIL WITH MORE TRANSPARENCY
    # Label the image
    labeled_image = create_labels(border_image)
    plot_image(labeled_image)

# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    # Example image paths:
    # image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    image_path = "C:\Victor\DrawByNumbers\TestImages\mickey-mouse-cinderella-castle-1024x683.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine/20240815_172047.jpg"
    start_application(image_path)  # Run the app