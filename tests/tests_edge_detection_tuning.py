import sys
import os
import unittest
import cv2
import numpy as np

# Add directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Custom Functions Imports
from image_processing.edge_detector import EdgeDetector
from plot_utils import compare_images
from image_processing.initial_processing import ImageProcessor



class TestEdgeDetection(unittest.TestCase):
    def setUp(self):
        """
        Setup before each test case. Load a sample image and prepare the edge detector.
        """
        # Set the image path
        self.image_path = "C:/Victor/Photo & Video/Nadine/_DSC0283.jpg"
        
        # Initialize ImageProcessor with the test image path
        self.processor = ImageProcessor(self.image_path)
        
        # Initialize EdgeDetector
        self.edge_detector = EdgeDetector()

        # Process the image
        self.loaded_image = self.processor.ensure_rgb_format()
        self.gray_scale_image = self.processor.convert_to_grayscale()
        self.resized_image = self.processor.resize_image(width=500)
        self.blurred_image = self.processor.apply_blur(kernel_size=11)

        self.blurred_gray_image = self.edge_detector.general_blur(self.gray_scale_image, kernel_size=7)
    
    @unittest.skip("Skipping other tests, only running Sobel edge test")
    def test_canny_edges(self):
        """
        Test the Canny edge detection with different parameter values, including blur kernel size, aperture size, and L2 gradient.
        """
        # Apply Canny edge detection with default parameters (no blur, default aperture size, L1 gradient)
        canny_default = self.edge_detector.canny_edges(self.gray_scale_image)
        
        # Apply Canny edge detection with tuned parameters (smaller thresholds, larger aperture, and blur)
        canny_tuned = self.edge_detector.canny_edges(self.gray_scale_image, 
                                                    min_val=50, max_val=100, 
                                                    blur_kernel_size=5, 
                                                    aperture_size=5, 
                                                    L2gradient=True)

        # Visual comparison of the two results
        compare_images(canny_default, canny_tuned, 
                    title1="Default Canny Edges", title2="Tuned Canny Edges")
        
        # Test: Check if the resulting images have the correct dimensions
        self.assertEqual(canny_default.shape, self.gray_scale_image.shape)
        self.assertEqual(canny_tuned.shape, self.gray_scale_image.shape)
        
        # Test: Ensure that the edge detection results differ when using different parameters
        self.assertFalse(np.array_equal(canny_default, canny_tuned), 
                        "The edge detection results should be different with tuned parameters.")

        # Additional Test: Apply Canny edge detection with no blur (default), but larger aperture and L2gradient
        canny_large_aperture = self.edge_detector.canny_edges(self.gray_scale_image, 
                                                            min_val=50, max_val=150, 
                                                            aperture_size=7, L2gradient=True)

        # Visual comparison of the original image with the large aperture edges
        compare_images(canny_default, canny_large_aperture, 
                    title1="Default Canny Edges", title2="Canny Edges with Large Aperture Size")

        # Test: Ensure that the results differ between the default and large aperture versions
        self.assertFalse(np.array_equal(canny_default, canny_large_aperture), 
                        "The edge detection results should differ with different aperture sizes.")
        
        # Optional: Apply Gaussian blur but keep aperture size default (to see the effect of blur only)
        canny_with_blur = self.edge_detector.canny_edges(self.gray_scale_image, 
                                                        min_val=50, max_val=150, 
                                                        blur_kernel_size=5, 
                                                        aperture_size=3, 
                                                        L2gradient=False)

        # Visual comparison of the blurred image with the default canny edges
        compare_images(canny_default, canny_with_blur, 
                    title1="Default Canny Edges", title2="Canny Edges with Gaussian Blur")
        
        # Test: Ensure the results are different between blurred and non-blurred
        self.assertFalse(np.array_equal(canny_default, canny_with_blur), 
                        "The edge detection results should differ with blur applied.")


    def test_sobel_edges(self):
        """
        Test Sobel edge detection with different parameter values.
        """
        # Apply Sobel edge detection with the default kernel size (3x3) and default scale/delta
        sobel_default = self.edge_detector.sobel_edges(self.gray_scale_image)
        
        # Apply Sobel edge detection with a larger kernel size (5x5), scaled by 2, and delta of 10
        sobel_tuned = self.edge_detector.sobel_edges(self.gray_scale_image, ksize=5, scale=2, delta=10)
        
        # Apply Sobel edge detection with only horizontal gradients (sobel_type = 'x')
        sobel_x = self.edge_detector.sobel_edges(self.gray_scale_image, sobel_type='x')

        # Apply Sobel edge detection with normalization enabled
        sobel_normalized = self.edge_detector.sobel_edges(self.gray_scale_image, normalize=True)
        
        # Visual comparison of the original image with different edge detection results
        compare_images(sobel_default, sobel_x, sobel_normalized,
                    title1="Default Sobel Edges", title2="Sobel Edges (X only)", title3="Normalized Sobel Edges")
        
        # Test: Ensure the images have the correct shape
        self.assertEqual(sobel_default.shape, self.gray_scale_image.shape)
        self.assertEqual(sobel_tuned.shape, self.gray_scale_image.shape)
        self.assertEqual(sobel_x.shape, self.gray_scale_image.shape)
        self.assertEqual(sobel_normalized.shape, self.gray_scale_image.shape)
        
        # Test: Ensure that the results differ with different kernel sizes or parameters
        self.assertFalse(np.array_equal(sobel_default, sobel_tuned), 
                        "The Sobel edge detection results should differ with different parameters.")
        self.assertFalse(np.array_equal(sobel_default, sobel_x), 
                        "The Sobel edge detection results should differ with sobel_type='x'.")
    

    @unittest.skip("Skipping other tests, only running Sobel edge test")
    def test_high_pass_filter(self):
        """
        Test the High Pass Filter edge detection with different kernel sizes.
        """
        # Apply High Pass Filter with default kernel size (3x3)
        hp_filter_default = self.edge_detector.high_pass_filter(self.gray_scale_image, kernel_size=3)
        
        # Apply High Pass Filter with a larger kernel size (7x7)
        hp_filter_large = self.edge_detector.high_pass_filter(self.gray_scale_image, kernel_size=7)
        
        # Visual comparison of the two results
        compare_images(hp_filter_default, hp_filter_large, title1="Default High Pass Filter", title2="Large Kernel High Pass Filter")
        
        # Test: Ensure the images have the correct shape
        self.assertEqual(hp_filter_default.shape, self.gray_scale_image.shape)
        self.assertEqual(hp_filter_large.shape, self.gray_scale_image.shape)
        
        # Test: Check if the results are different for different kernel sizes
        self.assertFalse(np.array_equal(hp_filter_default, hp_filter_large), "The High Pass Filter results should be different with different kernel sizes.")

if __name__ == '__main__':
    unittest.main()