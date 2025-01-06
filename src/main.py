# Import Libraries
from PIL import Image
import numpy as np
import cv2

# Import Classes
from image_processing_2.load import Loader
from image_processing_2.color_editing import ColorEditing
from image_processing_2.facets import Facets
from image_processing_2.borders import Borders
from image_processing_2.labels import Labels
from tools import PlottingTools, ColorTools, GeneralTools

# Load and initial processing of the image
def load(image_path: str) -> dict:
    """
    Loads and processes the image (ensure RGB format, grayscale conversion, resize, blur).
    """
    print("Loading Image")
    processor = Loader(image_path)

    load_dict = {}

    # Image Processing Pipeline
    loaded_image = processor.ensure_rgb_format()
    gray_scale_image = processor.convert_to_grayscale()
    resized_image = processor.resize_image(width=400)

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
    k_colors = 24
    choose = 0
    f_colors = k_colors + choose
    custom_kmeans = color_editor.kmeans_color_replacement(image, choose=choose, strength=5, colors=k_colors)
    return custom_kmeans, f_colors

# Create and prune facets
def facets(image: np.ndarray) -> np.ndarray:
    """
    Build and prune facets.
    """
    facet_creator = Facets()
    facets = facet_creator.build_facets(image)
    facet_pruning = facet_creator.prune_small_facets(facets=facets, original_image=image, min_size=700)
    return facet_pruning

    # Segment and Create

# Create borders and segment image
def borders(image: np.ndarray) -> np.ndarray:    
    border_creator = Borders()
    outlines = border_creator.detect_borders(image, 20)
    border_image, outline_black = border_creator.overlay(image, outlines)
    outline_template = border_creator.export_outlines(outlines) 
    return border_image, outline_black, outline_template

# Create and place labels
def labels(image: np.ndarray, outline_image: np.ndarray, n_colors: int = 10) -> np.ndarray:
    # Get Clustered Image NEED TO RUN WITH AMOUNT OF COLORS IN THE IMAGE
    label_creator = Labels()
    label_image, labeled_template = label_creator.labelling(image, outline_image, min_size=50)
    return label_image, labeled_template

# Main function coordinating the entire process
def start_application(image_path: str):
    print("Application Started.")
    pl_tools = PlottingTools()
    co_tools = ColorTools()
    ge_tools = GeneralTools()

    # Load and initial processing of the image
    load_dict = load(image_path)


    # Initial custom KMeans (keep specific colors)
    clustered_image, n_colors = KMeans(load_dict["resized_img"])
    # plotter.plot_image(clustered_image)

    # Use tools to improve color reduction

    # co_tool_image = co_tools.refine_threshold(clustered_image, strength=5)
    # co_tool_image = co_tools.midpoint_perceptual(co_tool_image, strength=5)
    # co_tool_image = co_tools.box_color_replacement(co_tool_image)
    # pl_tools.plot_image(co_tool_image)
    # ge_tools.save_image("C:\Victor\DrawByNumbers\DrawByNumbers\\tests\output", co_tool_image)
    co_tool_image = clustered_image


    # Create and prune facets
    # pruned_image = facets(co_tool_image)
    # pl_tools.plot_image(pruned_image)

    pruned_image = cv2.cvtColor(cv2.imread("C:\Victor\DrawByNumbers\TestOutput\\NEW PICTURE.png"), cv2.COLOR_BGR2RGB)
    n_colors = 20

    # Create borders and segment image
    again = True
    while again:
        border_image, outline_black, outline_template = borders(pruned_image)
        pl_tools.compare_images(border_image, outline_black, outline_template, 
                                title1="Borders", title2="Outline Black", title3="Outline Template")
        # Create and place labels
        labeled_image, labeled_template = labels(pruned_image, outline_template, n_colors)
        pl_tools.compare_images(labeled_image, labeled_template)
        again = bool(input("Again: "))
        pruned_image = co_tools.localized_pruning(pruned_image, 3)

# Run app only when main.py is executed directly
if __name__ == "__main__":
    # Example image paths:
    # image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    # image_path = "C:\Victor\DrawByNumbers\TestImages\mickey-mouse-cinderella-castle-1024x683.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine/20240815_172047.jpg"
    image_path = "C:\Victor\Photo & Video\\Nadine\\20240816_214217.jpg"
    start_application(image_path)  # Run the app