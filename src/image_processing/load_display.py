"""
This script provides basic functionality to load and display an image using OpenCV and Tkinter.
- The `load_image` function opens a file dialog for the user to select an image file (JPEG, PNG, BMP, or GIF).
- The selected image is loaded using OpenCV and returned as a NumPy array.
- The `display_image` function displays the image using Matplotlib with RGB color space.

Dependencies:
- OpenCV
- Tkinter
- Matplotlib
"""

import cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt

plt.ioff()

def load_image():
    # Placeholder for Development
    image = cv2.imread("C:/Victor/Photo & Video/Nadine/_DSC0283.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

    """ # Create Tkinter root window
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
    return image """

# Function to display an image
def display_image(image, window_name="Image", cmap=None):
    """
    Prepare an image for plotting without displaying it immediately.
    Allows specifying a colormap (default is None).
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    fig, ax = plt.subplots()  # Create a new figure
    ax.imshow(image_rgb, cmap=cmap)  # Display the image with the specified colormap
    ax.axis('off')  # Hide axes for clarity
    ax.set_title(window_name)  # Set the title
    return fig  # Return the figure object, no plt.show() yet.