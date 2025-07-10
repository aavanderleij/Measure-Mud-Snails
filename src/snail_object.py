import cv2
import imutils
from imutils import perspective
from imutils import contours
from matplotlib import pyplot as plt
import numpy as np
from .image_ruler import ImageRuler

class SnailObject(ImageRuler):
    """
    Class for detecting and measuring snail objects (or other objects of interest).
    """
    def __init__(self):
        """
        Initializes the SnailObject class for detecting and measuring snail objects.

        """
        # call the parent constructor
        super().__init__()

    def prep_image(self, image):
        """
        Preprocesses the input image by converting to grayscale, bilateral filtering, and performing edge detection.

        Args:
            image (numpy.ndarray): The input BGR image.

        Returns:
            numpy.ndarray: The edge-detected image.
        """
        # Bilateral filter
        blured = cv2.bilateralFilter(image, d=6, sigmaColor=30, sigmaSpace=50)

        # Canny edge detection
        edged = cv2.Canny(blured, 5, 30)

        # Morphological operations to clean up the edges
        # dilate contours to close gaps
        edged = cv2.dilate(edged, None, iterations=2)
        # erode to remove noise
        edged = cv2.erode(edged, None, iterations=1)

        # Show the blurred image for inspection
        self.show_image(blured, title="Blurred Image")
        return edged
    
    
    
    def clip_petri_dish(self, image):
        """
        Clips the image to the largest detected circle (petri dish).

        Args:
            image (numpy.ndarray): The input BGR image.

        Returns:
            numpy.ndarray: The clipped image containing only the petri dish.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Detect circles using Hough Transform
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=50, param2=30, minRadius=100, maxRadius=0
        )
        if circles is not None:
            
            circles = np.round(circles[0, :]).astype("int")

            # Draw all detected circles for inspection
            output = image.copy()
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            # Show the output with detected circles
            self.show_image(output, title="Detected Circles")

            # Sort circles by radius, descending
            circles_sorted = sorted(circles, key=lambda c: c[2], reverse=True)

            # Use the second largest circle if available, otherwise fallback to the largest
            # sencond largest circle is likely the inside rim of the petri dish
            if len(circles_sorted) > 1:
                x, y, r = circles_sorted[1]
            else:
                x, y, r = circles_sorted[0]
            

            # Create a mask
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)

            # Apply mask to image
            masked = cv2.bitwise_and(image, image, mask=mask)

            # Crop to bounding box
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            clipped = masked[max(y1,0):y2, max(x1,0):x2]

            # Show the clipped image
            self.show_image(clipped, title="Clipped Petri Dish")
            
            return mask
        else:
            # If no circle found, return original image
            return image

    def get_contours(self, edged, image):
        """
        Finds and draws contours on the image, highlighting each detected object.

        Args:
            edged (numpy.ndarray): The edge-detected image.
            image (numpy.ndarray): The image to annotate.

        Returns:
            None
        """
        
        #TODO this just draws contours, doest get anything
        # Find contours in the edged image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # sort contours from left to right
        (cnts, _) = imutils.contours.sort_contours(cnts)

        #TODO seperate method for drawing contours
        # draw all contours on the image
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
        print(f"Found {len(cnts)} contours in the image.")
        print(f"Number of contours with area > 1000: {sum(1 for contour in cnts if cv2.contourArea(contour) > 1000)}")
        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
                continue
            self.draw_rectangle_contour(contour, image)

        self.show_image(image, title="Detected Contours")
