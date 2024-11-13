"""
Main entry point of the program. Calls functions from the other .py files.
"""
from image_processing.initial_processing import ImageProcessor
from plot_utils import plot_image, compare_images

def start_application():
    print("Application Started.")
    # Later takes input from UI
    processor = ImageProcessor("C:\Victor\DrawByNumbers\TestImages\\flowers_name_in_english.jpg")
    loaded_image = processor.ensure_rgb_format()
    plot_image(loaded_image)
    gray_scale_image = processor.convert_to_grayscale()
    resized_image = processor.resize_image(width=500)
    blurred_image = processor.apply_blur(kernel_size=11)
    # processor.save_image("C:\Victor\DrawByNumbers\TestOutput\output.jpg")
    compare_images(loaded_image, gray_scale_image, resized_image)


# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    start_application()  # Run the app