import sys
import os
import unittest
import cv2
import numpy as np
import time

# Add directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from image_processing.compression import ImageCompressor


class TestImageCompressor(unittest.TestCase):

    def setUp(self):
        """
        Setup before each test case. Load a sample image and prepare for testing.
        """
        # Load a test image (replace with a suitable test image path)
        self.test_image = cv2.imread("C:/Victor/DrawByNumbers/TestImages/flowers_name_in_english.jpg")
        self.test_image = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
        self.compressor = ImageCompressor()

    def test_runtime(self):
        """
        Test and compare the runtime of reduce_color_space and reduce_color_space_gpu.
        """
        n_colors = 10  # Number of colors to reduce to

        # Measure runtime of CPU-based method
        start_time_cpu = time.time()
        color_reduced_image = self.compressor.reduce_color_space(self.test_image, n_colors)
        cpu_time = time.time() - start_time_cpu

        print(f"CPU Method Runtime: {cpu_time:.4f} seconds")

if __name__ == '__main__':
    unittest.main()
