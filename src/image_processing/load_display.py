"""

"""

import cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt

def load_image():
    # Create Tkinter root window
    root = tk.Tk()
    root.withdraw() # Hide the root window

    # Open window to select file
    image_path = filedialog.askopenfilename(
        title="Select an Image File", 

        # Limit File Types
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )

    # If the user canceled return None
    if not image_path:
        print("No image selected.")
        return None

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image.")
        return None
    
    print(f"Image loaded from: {image_path}")
    return image

# Function to display an image
def display_image(image, window_name="Image"):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes for clarity
    plt.show()
