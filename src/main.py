from plot_utils import plot_image, compare_images, plot_image_3d
from image_processing.initial_processing import ImageProcessor
from image_processing.compression import ImageCompressor
from image_processing.edge_detector import EdgeDetector

# Function to load and process the image
def load_and_compress_image(image_path: str):
    """
    Loads and processes the image (ensure RGB format, grayscale conversion, resize, blur).
    """
    processor = ImageProcessor(image_path)
    

    initial_processing = {}

    # Image Processing Pipeline
    print("Loading Image")
    loaded_image = processor.ensure_rgb_format()
    gray_scale_image = processor.convert_to_grayscale()
    resized_image = processor.resize_image(width=500)

    # Save to output dictionary
    initial_processing["loaded_image"] = loaded_image    
    initial_processing["gray_scale_image"] = gray_scale_image
    initial_processing["resized_image"] = resized_image

    # Image Compression
    print("Compressing Image")
    compressor = ImageCompressor()
    color_reduced_image = compressor.reduce_color_space(loaded_image, 100)
    print("Constructing Image")
    # plot_image(color_reduced_image)
    # compare_images(loaded_image, color_reduced_image)

    # Plot 3D visualization of image
    # print("Constructing 3d")
    # plot_image_3d(color_reduced_image)

    # Blur color_reduced_image
    color_reduced_blurred_image = compressor.general_blur(color_reduced_image)

    # Save to output dictionary
    initial_processing["color_reduced_blurred_image"] = color_reduced_blurred_image

    
    return initial_processing

# Function to perform edge detection and comparison
def detect_edges(initial_processing: dict):
    """
    Performs edge detection using Sobel and other methods, then compares the results.
    """
    edge_detection = {}
    edge_detector = EdgeDetector()
    
    # Apply Sobel edge detection
    print("Detecting Edges")
    canny_edges = edge_detector.canny_edges(initial_processing["gray_scale_image"], 
                                            min_val=10, max_val=40)
    sobel_edges = edge_detector.sobel_edges(initial_processing["gray_scale_image"], scale=0.3)

    # Save edges to dictionary
    edge_detection["canny_edges"] = canny_edges
    edge_detection["sobel_edges"] = sobel_edges
    
    # Compare edge detections
    compare_images(sobel_edges, canny_edges)

    # Export and visualize canny edges
    binary_canny_edges = edge_detector.export_edges(canny_edges)
    plot_image(binary_canny_edges)
    compare_images(initial_processing["color_reduced_blurred_image"], binary_canny_edges)


    # Export and visualize sobel edges
    binary_sobel_edges = edge_detector.export_edges(sobel_edges)
    plot_image(binary_sobel_edges)
    compare_images(initial_processing["color_reduced_blurred_image"], binary_sobel_edges)

    # Save binary edges to dictionary
    edge_detection["binary_canny_edges"] = canny_edges
    edge_detection["binary_sobel_edges"] = sobel_edges

    return edge_detection
    

# Main function that coordinates the entire process
def start_application(image_path: str):
    print("Application Started.")
    
    # Load and process the image
    initial_processing = load_and_compress_image(image_path)
    
    # Perform edge detection and visualize results
    edge_detection = detect_edges(initial_processing)

# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    # image_path = "C:/Victor/DrawByNumbers/TestImages/flowers_name_in_english.jpg"
    image_path = "C:\Victor\DrawByNumbers\TestImages\mickey-mouse-cinderella-castle-1024x683.jpg"
    # image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    start_application(image_path)  # Run the app
