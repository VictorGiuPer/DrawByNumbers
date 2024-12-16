# Import Libraries
from PIL import Image
import numpy as np
import cv2

# Import Classes
from image_processing.load import ImageProcessor
from image_processing_2.color_editing import ColorEditing
from image_processing_2.facets import Facets
from image_processing_2.borders import Borders
from image_processing_2.labels import Labels
from tools import Plotter, ColorTools

# Load and initial processing of the image
def load(image_path: str) -> dict:
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

# Initial custom KMeans (keep specific colors)
def KMeans(image: np.ndarray) -> np.ndarray:
    """
    Perform KMeans clustering on the image to reduce the number of colors.
    """ 
    color_editor = ColorEditing()
    # Possibly Replace the colors at the end
    k_colors = 10
    choose = 0
    f_colors = k_colors + choose
    custom_kmeans = color_editor.kmeans_color_replacement(image, choose=choose, strength=20, colors=k_colors)
    return custom_kmeans, f_colors

# Create and prune facets
def facets(image: np.ndarray) -> np.ndarray:
    """
    Build and prune facets.
    """
    facet_creator = Facets()
    facets = facet_creator.build_facets(image)
    facet_pruning = facet_creator.prune_small_facets(facets=facets, original_image=image, min_size=500)
    return facet_pruning

    # Segment and Create

# Create borders and segment image
def borders(image: np.ndarray) -> np.ndarray:    
    border_creator = Borders()
    borders, border_image, contour_image = border_creator.detect_borders(image)
    segmented_image = border_creator.segment_borders(border_image, borders)
    return border_image, segmented_image, contour_image

# Create and place labels
def labels(image: np.ndarray, n_colors: int = 10) -> np.ndarray:
    # Get Clustered Image NEED TO RUN WITH AMOUNT OF COLORS IN THE IMAGE
    label_creator = Labels()
    label_image = label_creator.labelling(image)
    # label_creator = Labels2()
    # label_image = label_creator.process_image(image)
    return label_image

# Main function coordinating the entire process
def start_application(image_path: str):
    print("Application Started.")
    plotter = Plotter()
    color_tools = ColorTools()

    # Load and initial processing of the image
    load_dict = load(image_path)

    # Initial custom KMeans (keep specific colors)
    clustered_image, n_colors = KMeans(load_dict["resized_img"])
    # plot_image(clustered_image)

    # Use tools to improve color reduction
    

    # Create and prune facets
    pruned_image = facets(clustered_image)
    plotter.plot_image(pruned_image)

    # pruned_image = cv2.cvtColor(cv2.imread("C:\Victor\DrawByNumbers\TestOutput\PRUNING_OUTPUT.png"), cv2.COLOR_BGR2RGB)

    # Create borders and segment image
    border_image, segment_image, outline_image = borders(pruned_image)
    plotter.compare_images(border_image, segment_image, outline_image, title1="Borders", title2="Segments", title3="Outlines")

    # ADD MORE DETAIL WITH MORE TRANSPARENCY
    # Create and place labels
    labeled_image = labels(pruned_image, n_colors)
    plotter.plot_image(labeled_image)

# Run app only when main.py is executed directly
if __name__ == "__main__":
    # Example image paths:
    # image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    image_path = "C:\Victor\DrawByNumbers\TestImages\mickey-mouse-cinderella-castle-1024x683.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine/20240815_172047.jpg"
    start_application(image_path)  # Run the app