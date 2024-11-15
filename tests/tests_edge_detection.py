import unittest
import cv2
import numpy as np
from src.image_processing.edge_detector import EdgeDetector
from src.plot_utils import compare_images

class TestEdgeDetection(unittest.TestCase):
    def setUP(self):
        """
        Setup before each test case. Load a sample image and prepare the edge detector.
        """
        # Load a sample image
        self.image_path = "C:\Victor\DrawByNumbers\TestImages\\flowers_name_in_english.jpg"
        self.image = cv2.imread(self.image_path)
        self.edge_detector = EdgeDetector()

        # Ensure image is loaded properly
        if self.image is None:
            raise ValueError(f"Could not load image from path: {self.image_path}")
        
        # Ensure image is in grayscale
        self.gray_scale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def test_canny_edges(self):
        """
        Test the Canny edge detection with different parameter values.
        """

        canny_default = self.edge_detector.canny_edges(self.gray_scale_image)
        canny_tuned = self.edge_detector.canny_edges(self.gray_scale_image, 
                                                     min_val=50, max_val=100)
         # Visual comparison of the two results
        compare_images(canny_default, canny_tuned, 
                       title1="Default Canny Edges", title2="Tuned Canny Edges")
        
        # Test: Check if the resulting images have the correct dimensions
        self.assertEqual(canny_default.shape, self.gray_scale_image.shape)
        self.assertEqual(canny_tuned.shape, self.gray_scale_image.shape)
        
        # Test: Check if there are differences between the two (for non-identical parameters)
        self.assertFalse(np.array_equal(canny_default, canny_tuned), 
                         "The edge detection results should be different with tuned parameters.")


    def test_sobel_edges(self):
        """
        Test Sobel edge detection with different kernel sizes.
        """
        # Apply Sobel edge detection with a small kernel size (3x3)
        sobel_small = self.edge_detector.sobel_edges(self.gray_scale_image, ksize=3)
        
        # Apply Sobel edge detection with a large kernel size (7x7)
        sobel_large = self.edge_detector.sobel_edges(self.gray_scale_image, ksize=7)
        
        # Visual comparison of the two results
        compare_images(sobel_small, sobel_large, title1="Small Kernel Sobel Edges", title2="Large Kernel Sobel Edges")
        
        # Test: Ensure the images have the correct shape
        self.assertEqual(sobel_small.shape, self.gray_scale_image.shape)
        self.assertEqual(sobel_large.shape, self.gray_scale_image.shape)
        
        # Test: Ensure that the results are different with different kernel sizes
        self.assertFalse(np.array_equal(sobel_small, sobel_large), "The Sobel edge results should differ with different kernel sizes.")
    
    
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