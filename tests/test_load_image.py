import unittest
import numpy as np
import sys, os

# Adjusting the path to import ImageProcessor correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from image_processing.load import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        """Set up resources for tests."""
        self.processor = ImageProcessor()
        self.valid_image_path = "C:\\Victor\\DrawByNumbers\\TestImages\\mickey-mouse-cinderella-castle-1024x683.jpg"
        self.invalid_image_path = "invalid_path.jpg"

    def test_load_image_valid(self):
        """Test loading a valid image."""
        image = self.processor.load_image(self.valid_image_path)
        self.assertIsNotNone(image, "Loaded image should not be None")
        self.assertIsInstance(image, np.ndarray, "Loaded image should be a numpy array")

    def test_load_image_invalid(self):
        """Test loading an invalid image raises ValueError."""
        with self.assertRaises(ValueError, msg="Could not load image at path:"):
            self.processor.load_image(self.invalid_image_path)

    def test_convert_to_grayscale(self):
        """Test converting an image to grayscale."""
        self.processor.load_image(self.valid_image_path)
        grayscale_image = self.processor.convert_to_grayscale()
        self.assertEqual(len(grayscale_image.shape), 2, "Grayscale image should have 2 dimensions")

    def test_ensure_rgb_format(self):
        """Test ensuring the image is in RGB format."""
        self.processor.load_image(self.valid_image_path)
        rgb_image = self.processor.ensure_rgb_format()
        self.assertEqual(rgb_image.shape[-1], 3, "RGB image should have 3 channels")

    def test_resize_image_with_width(self):
        """Test resizing the image by specifying width."""
        self.processor.load_image(self.valid_image_path)
        resized_image = self.processor.resize_image(width=200)
        self.assertEqual(resized_image.shape[1], 200, "Width should be 200")

    def test_resize_image_with_height(self):
        """Test resizing the image by specifying height."""
        self.processor.load_image(self.valid_image_path)
        resized_image = self.processor.resize_image(height=100)
        self.assertEqual(resized_image.shape[0], 100, "Height should be 100")

    def test_resize_image_with_both(self):
        """Test resizing the image by specifying both width and height."""
        self.processor.load_image(self.valid_image_path)
        resized_image = self.processor.resize_image(width=150, height=150)
        self.assertEqual(resized_image.shape[:2], (150, 150), "Image dimensions should match input")

if __name__ == "__main__":
    # Run all tests when script is executed
    unittest.main()

