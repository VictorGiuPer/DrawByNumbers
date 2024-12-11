from tools import plot_image, compare_images, plot_image_3d, resize_image, blur_image
from image_processing.load import ImageProcessor
from image_processing.pre_processing import Preprocessor
from image_processing.edge_detector import EdgeDetector
from image_processing.color_scheme import ColorSchemeCreator
from image_processing.detail_reduction import DetailReductor

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

# Function to pre-process the image for better edge detection
def pre_processing(load_dict: dict):
    """
    Prepare the image for edge detection.
    - Step 1: Reduce the color space
    - Step 3: Enhance contrast using histogram equalization
    - Step 4: Apply high-pass filter to emphasize edges
    - Step 5: Thresholding for binarization or enhancement

    """
    # Load necessary images
    loaded_img = load_dict["loaded_img"]
    pre_processing_dict = {}

    # Initialize pre_processor and pre_processed)image
    pre_processor = Preprocessor()
    pre_processed_img = loaded_img

    # Step 1: Reduce the color space
    pre_processed_img = pre_processor.initial_kmeans(pre_processed_img, 30)
    pre_processing_dict["color_reduced_img"] = pre_processed_img

    # Step 2: Apply blur
    pre_processed_img = pre_processor.gaussian_blur(pre_processed_img, 7)
    pre_processing_dict["cr_blurred_img"] = pre_processed_img

    # compare_images(loaded_img, pre_processed_img)

    return pre_processing_dict

# Function to create color scheme
def color_scheme(load_dict: dict, preprocess_img):
    """
    Creates the color scheme for the paint-by-numbers format.

    Parameters:
    - load_dict (dict): A dictionary containing additional parameters or settings (not directly used in this function).
    - edge_img (np.ndarray): The edge-detected image used as the base for color scheme creation.

    Returns:
    - np.ndarray: The final image with the generated color scheme applied.
    """
    # Instantiate ColorSchemeCreator
    cs_creator = ColorSchemeCreator()
    color_images = {}

    # Initialize color_zone_image
    color_zone_img = preprocess_img
    color_images["PreProcess"] = color_zone_img

    rd = DetailReductor()

    centers = rd.clusters_and_centers(color_zone_img, n_colors=20)
    facet = rd.color_facet_pruning(color_zone_img, )


    # Refine color zones: Threshold with selected colors.
    for i in range(7):
        color_zone_img = cs_creator.color_zones(color_zone_img, 10)
    color_images["ThresholdMerge"] = color_zone_img
    compare_images(color_images["PreProcess"], color_images["ThresholdMerge"])

    # Refine color zones: Color midpoint
    for i in range(3):
        color_zone_img = cs_creator.midpoint_perceptual(color_zone_img, 2)
    color_images["MidPoint"] = color_zone_img
    plot_image(color_images["MidPoint"])
 
    # Refine color zones: Manual color replacement
    for i in range(3):
        color_zone_img = cs_creator.box_color_replacement(color_zone_img)
    color_images["ManualReplacement"] = color_zone_img
    plot_image(color_images["ManualReplacement"])

    # Custom kmeans algorithm
    color_zone_img = cs_creator.kmeans_color_replacement(color_zone_img, 10, 10)
    color_images["KMeans2"] = color_zone_img
    compare_images(color_images["ManualReplacement"], color_images["KMeans2"])

    return color_zone_img

# Function to reduce the detail in the image
def reduce_detail(color_zone_img):

    detail_reductor = DetailReductor()
    # Brush Merging
    color_zone_img = detail_reductor.brush_paint_merge(color_zone_img, 1)
    plot_image(color_zone_img)

    # Facet Pruning
    # Recompute clusters and centers
    centers, labels = detail_reductor.clusters_and_centers(color_zone_img)
    color_zone_img = detail_reductor.color_facet_pruning(color_zone_img, labels, centers, 100)
    plot_image(color_zone_img) 
    """
    detail_reductor = DetailReductor()
    reduce_detail_img = detail_reductor.zone_merge(color_zone_img)

    return reduce_detail_img  """

# Function to perform edge detection and comparison
def detect_edges(load_dict: dict, color_zone_img):
    """
    Performs edge detection using Sobel and other methods, then compares the results.
    """
    edge_detector = EdgeDetector()

    # Resize Image
    color_zone_img = resize_image(color_zone_img)
    
    # Blur Image
    color_zone_img = blur_image(color_zone_img, 7)

    # Apply Canny edge detection
    canny_edges = edge_detector.canny_edges(color_zone_img, 
                                            min_val=10, max_val=40)

    # Export and visualize canny edges
    binary_canny_edges = edge_detector.export_edges(canny_edges)
    compare_images(canny_edges, binary_canny_edges, title1="Canny", title2="Binary Canny")

    # Refine Edges
    refined_canny_edges = edge_detector.refine_edges(canny_edges)
    binary_refined = edge_detector.export_edges(refined_canny_edges)
    plot_image(binary_refined)

    # Overlay image with edges
    edge_img = edge_detector.overlay_edges(color_zone_img, canny_edges)
    edge_img_refined = edge_detector.overlay_edges(color_zone_img, refined_canny_edges)
    plot_image(edge_img_refined)
    # compare_images(edge_img, edge_img_refined, title1="Image with Edges", title2="Image with refined Edges")
    return edge_img

# Main function that coordinates the entire process
def start_application(image_path: str):
    print("Application Started.")
    
    # Load the image
    print("Loading Image")
    load_dict = load_image(image_path)
    
    # Preprocessing
    print("Preprocessing")
    # pre_processing_dict = pre_processing(load_dict)

    # Applying color reduction and creating color scheme for 
    print("Creating Color Scheme")
    # color_zone_img = color_scheme(load_dict, pre_processing_dict["cr_blurred_img"])

    import cv2
    new_load = cv2.imread("C:\Victor\DrawByNumbers\TestOutput\FURTHER_REDUCTION_NADINE_COMO.png")
    color_zone_img = cv2.cvtColor(new_load, cv2.COLOR_BGR2RGB)

    # Reducing Detail
    reduced_detail_img = reduce_detail(color_zone_img)

    print("Detecting Edges")
    # edge_img = detect_edges(load_dict, color_zone_img)


# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    # image_path = "C:\Victor\Photo & Video\\2024_06_Italy\P1122131.JPG"
    # image_path = "C:\Victor\DrawByNumbers\TestImages\mickey-mouse-cinderella-castle-1024x683.jpg"
    image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine//20240815_172047.jpg"
    # image_path = "C:\Victor\Photo & Video\\Nadine\P1132388.JPG"
    # image_path = "C:\Victor\Photo & Video\\Nadine\IMG-20240721-WA0009.jpg"
    start_application(image_path)  # Run the app
