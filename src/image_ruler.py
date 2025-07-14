import cv2
import imutils
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance
from imutils import perspective

class ImageRuler:
    """
    Base class for image processing and measurement.
    """
    def __init__(self):
        """
        Initializes the ImageRuler base class for image processing and measurement.

        Args:
            None

        Returns:
            None
        """
        self.pixels_per_metric = None
        self.image = None

    @staticmethod
    def midpoint(point_a, point_b):
        """
        Calculates the midpoint between two points.

        Args:
            point_a (tuple): The (x, y) coordinates of the first point.
            point_b (tuple): The (x, y) coordinates of the second point.

        Returns:
            tuple: The (x, y) coordinates of the midpoint.
        """
        return ((point_a[0] + point_b[0]) * 0.5, (point_a[1] + point_b[1]) * 0.5)
    
    def get_dimensions_in_mm(self,box_points):
        """
        Calculates the dimensions of the rectangle in millimeters based on the midpoints.

        Args:
            tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY (float): Midpoint coordinates.

        Returns:
            tuple: Dimensions in millimeters (dimA, dimB).
        """

        (top_left, top_right, _, bottom_left) = box_points
        # Calculate distances in pixels

        # Compute distances between adjacent corners
        dA = distance.euclidean(top_left, top_right) 
        dB = distance.euclidean(top_left, bottom_left) 

        
        if self.pixels_per_metric is None:
            raise ValueError("pixels_per_metric is not set. Please set it before calling get_dimensions_in_mm.")
        
        # Convert to real-world units using the reference length
        dimA = dA / self.pixels_per_metric
        dimB = dB / self.pixels_per_metric
        
        return dimA, dimB
    
    def annotate_dimensions(self, image, dimA, dimB, tltrX, tltrY, trbrX, trbrY):
        """
        Annotates the image with the measured dimensions of the detected rectangle.

        Args:
            image (numpy.ndarray): The image to annotate.
            tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY (float): Midpoint coordinates.

        Returns:
            None
        """
        

        # Annotate the dimensions on the image
        cv2.putText(image, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 255, 255), 3)
        cv2.putText(image, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 255, 255), 3)
    
    @staticmethod
    def get_min_area_rect_box(cnt):
        """
        Returns ordered box points for the minimum area rectangle of a contour.

        Args:
            cnt (numpy.ndarray): The contour for which to compute the minimum area rectangle.

        Returns:
            numpy.ndarray: Ordered box points of the rectangle.
        """
        # Get the minimum area rectangle for the contour
        rotated_rect = cv2.minAreaRect(cnt)
        # Get the box points and order them
        box_points = cv2.boxPoints(rotated_rect)
        box_points = np.array(box_points, dtype="int")
        box_points = perspective.order_points(box_points)
        return box_points


    @staticmethod
    def show_image(image, title="Image"):
        """
        Displays the image using matplotlib.

        Args:
            image (numpy.ndarray): The image to display.
            title (str): The title of the image window.

        Returns:
            None
        """
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.show()

    def draw_midpoints_and_lines(self, image, box_points):
        """
        Draws midpoints and lines between midpoints on the image.

        Args:
            image (numpy.ndarray): The image to annotate.
            tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY (float): Midpoint coordinates.

        Returns:
            None
        """

         # get midpoints
        (top_left, top_right, bottom_right, bottom_left) = box_points
        (tltrX, tltrY) = self.midpoint(top_left, top_right)
        (blbrX, blbrY) = self.midpoint(bottom_left, bottom_right)
        (tlblX, tlblY) = self.midpoint(top_left, bottom_left)
        (trbrX, trbrY) = self.midpoint(top_right, bottom_right)

        # for every midpoint, draw a circle and a line connecting the midpoints
        for (x, y) in [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)]:
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        return image, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY

    def get_contours(self, image):
        """
        Finds contours in the (edged) image and draws them on the original image.

        Args:
            image (numpy.ndarray): The edge-detected image.


        Returns:
            None
        """
        # Find contours in the edged image
        cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Sort contours from left to right
        (cnts, _) = imutils.contours.sort_contours(cnts)


        print(f"Found {len(cnts)} contours in the image.")
        
        return cnts

    def draw_rectangle_contour(self, cnt, image):
        """
        Shared logic for drawing, computing midpoints, and annotating dimensions for a rectangle contour.

        Args:
            cnt (numpy.ndarray): The contour of the rectangle.
            image (numpy.ndarray): The image to annotate.

        Returns:
            numpy.ndarray: The box points of the rectangle.
        """

        # Get the minimum area rectangle box points
        box_points = self.get_min_area_rect_box(cnt)
        # Draw the rectangle on the image
        cv2.drawContours(image, [box_points.astype("int")], -1, (0, 255, 0), 2)


        # Draw midpoints and lines
        self.draw_midpoints_and_lines(image, box_points)

        return image 