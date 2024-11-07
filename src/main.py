from image_processing.load_display import load_image, display_image

def start_application():
    image = load_image()  # Let the user select an image
    if image is not None:
        display_image(image, "Selected Image")  # Display the selected image

# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    start_application()  # Run the app
