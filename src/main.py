from helper_functions import display_side_by_side, ensure_rgb
from image_processing.load_display import load_image, display_image
from image_processing.edge_detection import detect_edges, display_edges
from image_processing.quantization import quantization, display_quantizion
from image_processing.edge_segmentation import segment_image_by_contours, display_segments, extract_larger_segments

def start_application():
    image = load_image()  # Let the user select an image
    if image is not None:
        # Store plot as figure
        # fig_original = display_image(image)  # Display the selected image

        # Detect Edges
        print("Detecting Edges")
        # Store plot as figure
        edges = detect_edges(image=image)
        display_edges(image, edges)
        # display_side_by_side(fig_original, fig_edges)

        # Segment Picture
        segmented_image, contours = segment_image_by_contours(image=image, edges=edges)
        fig_segmentation = display_segments(pre_image=edges, 
                                            segmented_image=segmented_image)
        
        extract_larger_segments(image=image, edges=edges)




        num_colors = input("Quantization colors: ")
        quantized_image = quantization(image, num_colors=int(num_colors))
        # display_quantizion(original_image=image, quantized_image=quantized_image)

        quanitzed_egdes = detect_edges(quantized_image, cmap="tab10")
        display_edges(ensure_rgb(quantized_image), quanitzed_egdes, cmap=None)


    

# This ensures the app only runs when main.py is executed directly
if __name__ == "__main__":
    start_application()  # Run the app
