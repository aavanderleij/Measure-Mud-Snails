import cv2
import numpy as np
from .image_ruler import ImageRuler
from scipy.spatial import distance
import matplotlib.pyplot as plt

class ReferenceObject(ImageRuler):
    
    def __init__(self, reference_length_mm):
        """
        Initializes the ReferenceObject with a specified reference length in mm.

        Args:
            reference_length_mm (float): The real-world length of the reference object in millimeters.

        Returns:
            None
        """
        super().__init__()
        # Initialize the reference length in mm
        self.reference_length_mm = reference_length_mm
          

    def set_red_squire_ref_object_mask(self, image):
        """
        Create a mask for red objects in the image using HSV color filtering and morphological operations.

        Args:
            image (numpy.ndarray): The input BGR image.

        Returns:
            list: Contours found in the red mask.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Red mask (two ranges)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to clean up the mask
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def is_reference_rectangle(self, cnt, min_area=1000):
        """
        Determines if a given contour corresponds to a reference rectangle based on area and shape.

        Args:
            cnt (numpy.ndarray): The contour to check.
            min_area (float): Minimum area threshold.

        Returns:
            bool: True if the contour is a quadrilateral and large enough, False otherwise.
        """
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Approximate the contour and check if it has 4 vertices (quadrilateral)
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            return len(approx) == 4
        return False

    def process_reference_rectangle(self, cnt, image):
        """
        Processes a reference rectangle contour: draws it, computes midpoints, and annotates dimensions.

        Args:
            cnt (numpy.ndarray): The contour of the rectangle.
            image (numpy.ndarray): The image to annotate.

        Returns:
            None
        """
        self.draw_rectangle_contour(cnt, image)

    def detect_and_annotate(self, image):
        """
        Detects red reference rectangles in the image and annotates them.

        Args:
            image (numpy.ndarray): The input BGR image.

        Returns:
            None
        """
        contours = self.set_red_squire_ref_object_mask(image)
        for cnt in contours:
            if self.is_reference_rectangle(cnt):
                self.process_reference_rectangle(cnt, image)


    def calculate_pixels_per_metric(self, image):
        """
        Detects the reference rectangle and calculates pixels per metric (mm) using the smaller width.

        Args:
            image (numpy.ndarray): The input BGR image.

        Returns:
            float: The calculated pixels per metric (pixels per mm).
        """
        contours = self.set_red_squire_ref_object_mask(image)
        # Find the largest rectangle contour
        max_area = 0
        best_cnt = None
        for cnt in contours:
            if self.is_reference_rectangle(cnt):
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
                    print('Found reference rectangle with area:', area)
        
        self.show_image(image, title="Reference Rectangle Detection")
        if best_cnt is None:
            raise ValueError("No suitable reference rectangle found.")

        box = self.get_min_area_rect_box(best_cnt)
        (tl, tr, br, bl) = box
        # Compute width and height in pixels
        width = distance.euclidean(tl, tr)
        height = distance.euclidean(tr, br)
        # Use the smaller dimension as the reference width
        ref_pixels = min(width, height)
        self.pixels_per_metric = ref_pixels / self.reference_length_mm
        return self.pixels_per_metric
