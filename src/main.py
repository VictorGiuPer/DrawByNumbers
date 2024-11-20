from plot_utils import plot_image, compare_images, plot_image_3d
from image_processing.load import ImageProcessor
from image_processing.pre_processing import Preprocessor
from image_processing.edge_detector import EdgeDetector

# Function to load and process the image
def load_image(image_path: str):
    """
    Loads and processes the image (ensure RGB format, grayscale conversion, resize, blur).
    """
    processor = ImageProcessor(image_path)
    

    load_dict = {}

    # Image Processing Pipeline
    print("Loading Image")
    loaded_image = processor.ensure_rgb_format()
    gray_scale_image = processor.convert_to_grayscale()
    resized_image = processor.resize_image(width=500)

    # Save to output dictionary
    load_dict["loaded_image"] = loaded_image    
    load_dict["gray_scale_image"] = gray_scale_image
    load_dict["resized_image"] = resized_image

    return load_dict

def pre_processing(load_dict: dict, blur_type: str = "gaussian"):
    # EXTEND FUNCTIONS (STEPS 4 AND 5)
    """
    Prepare the image for edge detection.
    - Step 1: Reduce the color space
    - Step 2: Apply blur (gaussian/median) to smooth the image
    - Step 3: Enhance contrast using histogram equalization
    - Step 4: Apply high-pass filter to emphasize edges
    - Step 5: Thresholding for binarization or enhancement

    """
    # Load necessary images
    loaded_image = load_dict["loaded_image"]
    gray_scale_image = load_dict["gray_scale_image"]
    pre_processing_dict = {}

    # Initialize pre_processor
    pre_processor = Preprocessor()

    # Step 1: Reduce the color space
    pre_processed_image = pre_processor.reduce_color_space(loaded_image, 200)
    pre_processing_dict["color_reduced_image"] = pre_processed_image

    # Step 2: Apply blur
    if blur_type.lower() == "median":
        pre_processed_image = pre_processor.median_blur(pre_processed_image)
    elif blur_type.lower() == "gaussian":
        pre_processed_image = pre_processor.gaussian_blur(pre_processed_image)
    else:
        raise ValueError("Not a valid blur method. Choose (gaussian | median).")
    pre_processing_dict["cr_blurred_image"] = pre_processed_image

    # Step 3: Enhance contrast using historgram equalization
    pre_processed_image =  pre_processor.histogram_equalization(pre_processed_image)
    pre_processing_dict["cr_bl_equalized_image"] = pre_processed_image

    # Visualize progression
    compare_images(pre_processing_dict["color_reduced_image"], pre_processing_dict["cr_blurred_image"], 
                   pre_processing_dict["cr_bl_equalized_image"], title1="Color Reduced", 
                   title2="Color Reduced & Blurred", title3 = "Color Reduced, Blurred & Equalized")

    return pre_processing_dict

# Function to perform edge detection and comparison
def detect_edges(load_dict: dict, pre_processing_dict: dict):
    # CLEANUP NEEDED
    """
    Performs edge detection using Sobel and other methods, then compares the results.
    """
    edge_detection = {}
    edge_detector = EdgeDetector()
    
    # Apply Sobel edge detection
    print("Detecting Edges")
    canny_edges = edge_detector.canny_edges(load_dict["gray_scale_image"], 
                                            min_val=10, max_val=40)
    sobel_edges = edge_detector.sobel_edges(load_dict["gray_scale_image"], scale=0.3)

    # Save edges to dictionary
    edge_detection["canny_edges"] = canny_edges
    edge_detection["sobel_edges"] = sobel_edges
    
    # Compare edge detections
    compare_images(sobel_edges, canny_edges)

    # Export and visualize canny edges
    binary_canny_edges = edge_detector.export_edges(canny_edges)
    plot_image(binary_canny_edges)
    compare_images(load_dict["color_reduced_blurred_image"], binary_canny_edges)


    # Export and visualize sobel edges
    binary_sobel_edges = edge_detector.export_edges(sobel_edges)
    plot_image(binary_sobel_edges)
    compare_images(load_dict["color_reduced_blurred_image"], binary_sobel_edges)

    # Save binary edges to dictionary
    edge_detection["binary_canny_edges"] = canny_edges
    edge_detection["binary_sobel_edges"] = sobel_edges

    return edge_detection
    

# Main function that coordinates the entire process
def start_application(image_path: str):
    print("Application Started.")
    
    # Load the image
    print("Loading Image")
    load_dict = load_image(image_path)
    
    # Preprocessing
    print("Preprocessing Image")
    pre_processing(load_dict)

    # Perform edge detection and visualize results
    print("Detecting Edges")
    edge_detection = detect_edges(load_dict)

# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    # image_path = "C:/Victor/DrawByNumbers/TestImages/flowers_name_in_english.jpg"
    image_path = "C:\Victor\DrawByNumbers\TestImages\mickey-mouse-cinderella-castle-1024x683.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    start_application(image_path)  # Run the app
