    def median_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Reduce the detail in the image by introducing median blur.
        
        Parameters:
        - image (np.ndarray): Input image.
        - kernel_size (int): Blur intensity.

        Returns:
        - blurred_image (np.ndarray): Image with median blur.
        """
        if image is None:
            raise ValueError("No image loaded. Load an image first.")
        
        blurred_image = cv2.medianBlur(image, kernel_size)
        return blurred_image

    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to enhance the contrast of a grayscale or RGB image.

        Parameters:
        - image (np.ndarray): Input image (grayscale or RGB).

        Returns:
        - np.ndarray: Contrast-enhanced image.
        """
        if len(image.shape) == 2:  # Grayscale image
            return cv2.equalizeHist(image)

        elif len(image.shape) == 3:  # RGB image
            # Split the image into its R, G, B channels
            r, g, b = cv2.split(image)
            
            # Equalize each channel independently
            r_eq = cv2.equalizeHist(r)
            g_eq = cv2.equalizeHist(g)
            b_eq = cv2.equalizeHist(b)

            # Merge the channels back together
            equalized_image = cv2.merge((r_eq, g_eq, b_eq))
            return equalized_image
        else:
            raise ValueError("Input image must be either grayscale or RGB.")

    def high_pass_filter(self, image: np.ndarray, kernel_size: int = 51) -> np.ndarray:
        """
        Apply a high-pass filter to enhance edges in the image.

        Parameters:
        - image (np.ndarray): Input image (grayscale or BGR).
        - kernel_size (int): Size of the kernel for the Gaussian blur (must be odd).

        Returns:
        - np.ndarray: High-pass filtered image, emphasizing edges.
        """
        if len(image.shape) == 2:  # Grayscale image
            # Create a low-pass filtered image
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            # Get high-pass result
            high_pass_image = cv2.absdiff(image, blurred)

            return high_pass_image

        elif len(image.shape) == 3:  # RGB image
            print("RGB")
            # Split the image into its R, G, B channels
            r, g, b = cv2.split(image)
            # Apply high-pass filter to each channel

            r_high_pass = cv2.subtract(r, cv2.GaussianBlur(r, (kernel_size, kernel_size), 0))
            g_high_pass = cv2.subtract(g, cv2.GaussianBlur(g, (kernel_size, kernel_size), 0))
            b_high_pass = cv2.subtract(b, cv2.GaussianBlur(b, (kernel_size, kernel_size), 0))

            # Merge the channels back together
            high_pass_image = cv2.merge((r_high_pass, g_high_pass, b_high_pass))
            return high_pass_image
        else:
            raise ValueError("Input image must be either grayscale or RGB.")

    def thresholding(self, image: np.ndarray, threshold: int = 128, max_value: int = 255, method: str = "binary"):
        """
        Apply thresholding to convert an image to binary or enhance regions based on intensity.

        Parameters:
        - image (np.ndarray): Input image (grayscale or RGB).
        - threshold (int): Threshold value.
        - max_value (int): Maximum value to assign for binary thresholding.
        - method (str): Type of thresholding ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv').

        Returns:
        - np.ndarray: Thresholded image.
        """
        method_dict = {
            "binary": cv2.THRESH_BINARY,
            "binary_inv": cv2.THRESH_BINARY_INV,
            "trunc": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_inv": cv2.THRESH_TOZERO_INV, 
        }

        thresh_type = method_dict.get(method.lower(), cv2.THRESH_BINARY)

        # Grayscale image
        if len(image.shape) == 2:
            # Apply thresholding directly
            _, thresholded_image = cv2.threshold(image, threshold, max_value, thresh_type)
            return thresholded_image

        elif len(image.shape) == 3:
            r, g, b = cv2.split(image)

            # Apply thresholding to each channel
            _, r_thresh = cv2.threshold(r, threshold, max_value, thresh_type)
            _, g_thresh = cv2.threshold(g, threshold, max_value, thresh_type)
            _, b_thresh = cv2.threshold(b, threshold, max_value, thresh_type)

            # Merge the channels back together
            thresholded_image = cv2.merge((r_thresh, g_thresh, b_thresh))
            return thresholded_image
            
        else:
            raise ValueError("Input image must be either grayscale or RGB.")
        

    def enhance_sharpness_unsharp(image: np.ndarray, strength: float = 1.5, kernel_size: int = 5) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        high_pass = cv2.subtract(image, blurred)
        sharpened = cv2.addWeighted(image, 1.0 + strength, high_pass, -strength, 0)
        return sharpened
