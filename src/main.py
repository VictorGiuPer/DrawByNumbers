"""
Main entry point of the program. Calls functions from the other .py files.
"""
from plot_utils import plot_image, compare_images
from image_processing.initial_processing import ImageProcessor
from image_processing.edge_detector import EdgeDetector

def start_application():
    print("Application Started.")
    # Later takes input from UI

    # Initial processing
    # Other picture: "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
    # processor = ImageProcessor("C:/Victor/Photo & Video/Nadine/_DSC0283.jpg")
    processor = ImageProcessor("C:\Victor\DrawByNumbers\TestImages\\flowers_name_in_english.jpg")
    loaded_image = processor.ensure_rgb_format()
    #plot_image(loaded_image)
    gray_scale_image = processor.convert_to_grayscale()
    resized_image = processor.resize_image(width=500)
    blurred_image = processor.apply_blur(kernel_size=11)
    # processor.save_image("C:\Victor\DrawByNumbers\TestOutput\output.jpg")
    # compare_images(loaded_image, gray_scale_image, resized_image)


    # Individual Edge detection approach
    edge_detector = EdgeDetector()
    canny_edges = edge_detector.canny_edges(gray_scale_image)  
    sobel_edges = edge_detector.sobel_edges(gray_scale_image)
    high_pass_filter = edge_detector.high_pass_filter(gray_scale_image)
    compare_images(canny_edges, sobel_edges, high_pass_filter, 
                   title1="Canny Edges", title2="Sobel Edges", title3="High Pass Filter")
    
     # Individual Edge detection approach. Parameters tuned'.
    canny_edges_t = edge_detector.canny_edges(gray_scale_image, min_val=140, max_val=200)  
    sobel_edges_t = edge_detector.sobel_edges(gray_scale_image)
    high_pass_filter_t = edge_detector.high_pass_filter(gray_scale_image)
    compare_images(canny_edges_t, sobel_edges_t, high_pass_filter_t, 
                   title1="Canny Edges Tuned", title2="Sobel Edges Tuned", title3="High Pass Filter Tuned")

    """ # Combined Edge Detection Appraoch
    high_pass_filter_2 = edge_detector.high_pass_filter(gray_scale_image, blurred_image)
    sobel_edges_2 = edge_detector.sobel_edges(high_pass_filter_2)
    canny_edges_2 = edge_detector.canny_edges(high_pass_filter_2)  
    compare_images( high_pass_filter_2, sobel_edges_2, canny_edges_2, 
                   title1="High Pass Filter", title2="Sobel Edges", title3="Canny Edges") """


# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    start_application()  # Run the app