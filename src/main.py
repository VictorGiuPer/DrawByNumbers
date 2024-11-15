from plot_utils import plot_image, compare_images
from image_processing.initial_processing import ImageProcessor
from image_processing.edge_detector import EdgeDetector

# Function to load and process the image
def load_and_process_image(image_path: str):
    """
    Loads and processes the image (ensure RGB format, grayscale conversion, resize, blur).
    """
    processor = ImageProcessor(image_path)
    
    # Image Processing Pipeline
    loaded_image = processor.ensure_rgb_format()
    gray_scale_image = processor.convert_to_grayscale()
    resized_image = processor.resize_image(width=500)
    blurred_image = processor.apply_blur(kernel_size=11)
    
    return loaded_image, gray_scale_image, resized_image, blurred_image

# Function to perform edge detection and comparison
def detect_and_compare_edges(gray_scale_image, loaded_image):
    """
    Performs edge detection using Sobel and other methods, then compares the results.
    """
    edge_detector = EdgeDetector()
    
    # Apply Sobel edge detection
    sobel_edges = edge_detector.sobel_edges(gray_scale_image)
    
    # Apply blur to the image and detect edges on the blurred image
    blurred_grayscale = edge_detector.general_blur(gray_scale_image)
    sobel_edges_blurred = edge_detector.sobel_edges(blurred_grayscale)
    
    # Export the edges and compare
    sobel_blurred_exported = edge_detector.export_edges(sobel_edges_blurred)
    compare_images(loaded_image, sobel_blurred_exported, title1="Original", title2="Detected Edges")
    
    return sobel_edges, sobel_edges_blurred

# Main function that coordinates the entire process
def start_application(image_path: str):
    print("Application Started.")
    
    # Load and process the image
    loaded_image, gray_scale_image, resized_image, blurred_image = load_and_process_image(image_path)
    
    # Visualize the processed image
    plot_image(blurred_image, title="Blurred Image")
    
    # Perform edge detection and visualize results
    sobel_edges, sobel_edges_blurred = detect_and_compare_edges(gray_scale_image, loaded_image)
    
    # You can expand this section with other edge detection or processing methods

# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    image_path = "C:/Victor/DrawByNumbers/TestImages/flowers_name_in_english.jpg"
    start_application(image_path)  # Run the app
