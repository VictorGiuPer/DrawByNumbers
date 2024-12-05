import matplotlib.pyplot as plt
import cv2

def display_side_by_side(fig1, fig2):
    """
    Display two figure objects as subplots in one window.
    The function will preserve the colormap used in the individual figure functions.
    """
    plt.figure(figsize=(10, 5))  # Set the overall size of the output figure
    
    # First subplot: Get the image and title from the first figure
    ax1 = fig1.get_axes()[0]
    img1 = ax1.get_images()[0]
    plt.subplot(1, 2, 1)
    plt.imshow(img1.get_array(), cmap=img1.get_cmap())  # Respect the colormap
    plt.axis('off')  # Hide axes for clarity
    plt.title(ax1.get_title())  # Set the title
    
    # Second subplot: Get the image and title from the second figure
    ax2 = fig2.get_axes()[0]
    img2 = ax2.get_images()[0]
    plt.subplot(1, 2, 2)
    plt.imshow(img2.get_array(), cmap=img2.get_cmap())  # Respect the colormap
    plt.axis('off')  # Hide axes for clarity
    plt.title(ax2.get_title())  # Set the title
    
    plt.show()  # Finally, display the combined plot


def ensure_rgb(image):
    """
    Ensures the image is in RGB format. If it's already in RGB, it is returned as is;
    if it's in BGR or grayscale, it is converted to RGB.

    Parameters:
    - image (numpy.ndarray): Input image in BGR, RGB, or grayscale format.

    Returns:
    - numpy.ndarray: Image in RGB format.
    """
    # If the image has 3 channels and is likely already RGB, return as is
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if it's already in RGB by looking at pixel ordering.
        # OpenCV defaults to BGR, so if this is BGR, we convert it to RGB
        # Here, we check a quick pixel sample to determine if it's RGB or BGR
        # by inspecting the channels' typical order
        if image[0, 0, 0] < image[0, 0, 2]:  # Assuming BGR order if B > R
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return image  # Likely already RGB
    elif len(image.shape) == 2:  # Grayscale image with only one channel
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Otherwise, it’s a single-channel grayscale image
    return image