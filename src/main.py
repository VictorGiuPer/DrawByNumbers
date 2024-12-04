from plot_utils import plot_image, compare_images, plot_image_3d
from image_processing.load import ImageProcessor
from image_processing.pre_processing import Preprocessor
from image_processing.edge_detector import EdgeDetector
from image_processing.color_scheme import ColorSchemeCreator

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
def pre_processing(load_dict: dict, blur_type: str = "gaussian"):
    # EXTEND FUNCTIONS (STEPS 4 AND 5)
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
    pre_processed_img = pre_processor.cs_red_k1(pre_processed_img, 30)
    pre_processing_dict["color_reduced_img"] = pre_processed_img

    # Step 2: Apply blur
    if blur_type.lower() == "median":
        pre_processed_img = pre_processor.median_blur(pre_processed_img)
    elif blur_type.lower() == "gaussian":
        pre_processed_img = pre_processor.gaussian_blur(pre_processed_img)
    else:
        raise ValueError("Not a valid blur method. Choose (gaussian | median).")
    pre_processing_dict["cr_blurred_img"] = pre_processed_img

    # compare_images(loaded_img, pre_processed_img)

    return pre_processing_dict

# Function to create color scheme
def color_scheme(load_dict: dict, edge_img):
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

    # Initialize color_zone_image
    color_zone_img = edge_img

    # Refine color zones based on selected colors
    for i in range(5):
        selected_color = cs_creator.get_colors(color_zone_img)

        # If no colors are selected (empty list), exit the loop early.
        if len(selected_color) == 0:
            break

        # Refine color zones within threshold with selected colors.
        color_zone_img = cs_creator.color_zones(color_zone_img, selected_color, 10)

    compare_images(edge_img, color_zone_img)

    # Refine color zones based on color midpoint
    for i in range(3):
        selected_color_1 = cs_creator.get_colors(color_zone_img)
        selected_color_2 = cs_creator.get_colors(color_zone_img)

        # Apply midpoint color within treshold
        color_zone_img = cs_creator.midpoint_perceptual(color_zone_img, 
                                                          selected_color_1, 
                                                          selected_color_2, 2)
        
    plot_image(color_zone_img)

    # Custom kmeans algorithm
    kmeans_colors = []
    for i in range(3):
        kmeans_c = cs_creator.get_colors(color_zone_img)
        kmeans_colors.append(kmeans_c)
    color_zone_reduced_img = cs_creator.kmeans_color_replacement(color_zone_img, kmeans_colors, 10, 10)
    compare_images(color_zone_img, color_zone_reduced_img)


    return color_zone_img

# Function to perform edge detection and comparison
def detect_edges(load_dict: dict, color_zone_img):
    """
    Performs edge detection using Sobel and other methods, then compares the results.
    """
    edge_detector = EdgeDetector()
    
    # Apply Canny edge detection
    canny_edges = edge_detector.canny_edges(color_zone_img, 
                                            min_val=10, max_val=40)

    # Export and visualize canny edges
    binary_canny_edges = edge_detector.export_edges(canny_edges)
    # compare_images(canny_edges, binary_canny_edges, title1="Canny", title2="Binary Canny")

    edge_img = edge_detector.overlay_edges(color_zone_img, canny_edges)
    
    # Plot image with edges
    plot_image(edge_img, title="Image with Edges")
    return edge_img

# Main function that coordinates the entire process
def start_application(image_path: str):
    print("Application Started.")
    
    # Load the image
    print("Loading Image")
    load_dict = load_image(image_path)
    
    # Preprocessing
    print("Preprocessing")
    pre_processing_dict = pre_processing(load_dict)

    # Applying color reduction and creating color scheme for 
    print("Creating Color Scheme")
    color_zone_img = color_scheme(load_dict, pre_processing_dict["cr_blurred_img"])

    # Perform edge detection and visualize results
    print("Detecting Edges")
    # edge_img = detect_edges(load_dict, color_zone_img)



# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    # image_path = "C:/Victor/DrawByNumbers/TestImages/flowers_name_in_english.jpg"
    image_path = "C:\Victor\DrawByNumbers\TestImages\mickey-mouse-cinderella-castle-1024x683.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine//20240815_172047.jpg"
    start_application(image_path)  # Run the app
