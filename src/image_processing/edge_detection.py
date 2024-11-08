import cv2
from matplotlib import pyplot as plt


def detect_edges(image, low_threshold=100, high_threshold=200, cmap=None):
    """
    Applies Canny edge detection to an image.

    Parameters:
        image (numpy array): The input image on which to perform edge detection.
        low_threshold (int): Lower bound for edge detection threshold.
        high_threshold (int): Upper bound for edge detection threshold.

    Returns:
        numpy array: The binary edge-detected image.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if cmap == "RGB":
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    else:
        image_gray = image
    
    # Apply Canny edge detection
    edges = cv2.Canny(image_gray, low_threshold, high_threshold)
    return edges

# Function to display an edge-detected image
def display_edges(pre_image, edges, window_name="Image with Edges", cmap='gray'):
    """
    Displays the original image and the edge-detected image side by side.
    
    Parameters:
    - edges (numpy.ndarray): The edge-detected image (binary).
    - original_image (numpy.ndarray): The original image (BGR format).
    - window_name (str, optional): The title for the edge-detected image subplot.
    - cmap (str, optional): Colormap to apply (default is 'gray' for edge detection).
    
    Returns:
    - None
    """
    plt.figure(figsize=(10, 5))

    # Display the original image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(pre_image)
    plt.title("Pre Edges Images")
    plt.axis("off")

    # Display the edge-detected image on the right
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap=cmap)
    plt.title(window_name)
    plt.axis("off")

    plt.show()


