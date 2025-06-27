

import cv2
from matplotlib import pyplot as plt
import numpy as np


def hsv_filter(image_clip, hue_min, hue_max):
        """
        Converts an RGB image to HSV, applies a hue threshold, and returns the color-filtered image.

        Parameters:
        - image_clip: np.ndarray, input RGB image
        - hue_min: float, minimum hue threshold (0 to 1)
        - hue_max: float, maximum hue threshold (0 to 1)

        Returns:
        - image_color_filtered: np.ndarray, filtered image with background removed
        """
        # Convert RGB to HSV
        image_clip = cv2.convertScaleAbs(image_clip, alpha=1.2, beta=50)
        plt.imshow(cv2.cvtColor(image_clip, cv2.COLOR_BGR2RGB))
        plt.show()
        hsv_image = cv2.cvtColor(image_clip, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_image[:, :, 0] /= 179.0  # Normalize hue to [0, 1]

        # Define thresholds
        channel1_min = hue_min
        channel1_max = hue_max
        channel2_min = 0.0
        channel2_max = 1.0
        channel3_min = 0.0
        channel3_max = 1.0

        # Create mask
        mask = (
            (hsv_image[:, :, 0] >= channel1_min) & (hsv_image[:, :, 0] <= channel1_max) &
            (hsv_image[:, :, 1] >= channel2_min) & (hsv_image[:, :, 1] <= channel2_max) &
            (hsv_image[:, :, 2] >= channel3_min) & (hsv_image[:, :, 2] <= channel3_max)
        )

        # Apply mask
        image_color_filtered = np.copy(image_clip)
        image_color_filtered[~mask] = 0

        
        plt.imshow(cv2.cvtColor(image_color_filtered, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return image_color_filtered