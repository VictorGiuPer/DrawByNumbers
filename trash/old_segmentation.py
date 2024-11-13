"""
Edge detection and segmentation logic.
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np

def segment_image_by_contours(image, edges):
    """
    Segments the image based on detected contours and fills each region with a unique color.

    Parameters:
    - image (numpy.ndarray): The original image (BGR format).
    - edges (numpy.ndarray): The edge-detected image (binary).

    Returns:
    - segmented_image (numpy.ndarray): Image with regions filled with unique colors.
    - contours (list): List of detected contours.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    segmented_image = np.zeros_like(image)

    colors = [
        (255, 0, 0),  # Blue
        (0, 255, 0),  # Green
        (0, 0, 255),  # Red
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (255, 255, 0), # Cyan
        (128, 0, 128), # Purple
        (0, 128, 128)  # Teal
    ]
    
    # For each contour, assign a region filled with a unique color
    for i, contour in enumerate(contours):
        # Cycle through predefined colors if there are more contours than colors
        color = colors[i % len(colors)]  # This ensures color repeats if more than 8 segments
        cv2.drawContours(segmented_image, [contour], -1, color, thickness=cv2.FILLED)
    
    return segmented_image, contours

def display_segments(pre_image, segmented_image):
    """
    Displays the original and segmented images side by side.

    Parameters:
    - original_image (numpy.ndarray): The original image (BGR format).
    - segmented_image (numpy.ndarray): The segmented image with unique colors for each region.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB))
    plt.title("Pre Segmentation Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Image")
    plt.axis("off")

    plt.show()


def extract_larger_segments(image, edges, num_segments=3, dilation_kernel_size=(5, 5), dilation_iterations=2):
    """
    Extract larger segments by dilating the detected edges and highlighting the biggest contours.

    Parameters:
    - image (numpy.ndarray): The original image (BGR format).
    - edges (numpy.ndarray): The edge-detected image (binary).
    - num_segments (int): The number of largest segments to extract and display.
    - dilation_kernel_size (tuple): The size of the kernel used for dilation.
    - dilation_iterations (int): The number of times dilation is applied.

    Returns:
    - None: Displays the largest segments on a white background.
    """
    # Dilate the edges to make the contours thicker and more visible
    kernel = np.ones(dilation_kernel_size, np.uint8)  # Structuring element
    dilated_edges = cv2.dilate(edges, kernel, iterations=dilation_iterations)
    
    # Find contours in the dilated edge image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order and take the top 'num_segments'
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_segments]
    
    # Create a white background image
    white_background = np.ones_like(image) * 255  # White background (BGR format)
    
    # Generate a unique color for each of the largest segments
    colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(num_segments)]
    
    # Draw the largest segments on the white background with unique colors
    for i, contour in enumerate(contours):
        color = colors[i]
        cv2.drawContours(white_background, [contour], -1, color, thickness=cv2.FILLED)
    
    # Convert to RGB (for display with plt)
    white_background_rgb = cv2.cvtColor(white_background, cv2.COLOR_BGR2RGB)
    
    # Plot the result
    plt.figure(figsize=(6, 6))
    plt.imshow(white_background_rgb)
    plt.title(f"Top {num_segments} Largest Segments (Dilated Edges)")
    plt.axis('off')  # Hide axes
    plt.show()