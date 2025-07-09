import cv2
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
    
    def annotate_dimensions(self, image, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY):
        """
        Annotates the image with the measured dimensions of the detected rectangle.

        Args:
            image (numpy.ndarray): The image to annotate.
            tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY (float): Midpoint coordinates.

        Returns:
            None
        """
        #TODO should be in a separate method
        # Calculate distances in pixels
        dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # Convert to real-world units using the reference length
        if self.pixels_per_metric is None:
            raise ValueError("pixels_per_metric is not set. Please set it before calling annotate_dimensions.")
            
        print(f"Calculated pixels_per_metric: {self.pixels_per_metric}")
        
        print(f"Distance A (pixels): {dA}, Distance B (pixels): {dB}")
        dimA = dA / self.pixels_per_metric
        dimB = dB / self.pixels_per_metric

        print(f"Dimension A (mm): {dimA}, Dimension B (mm): {dimB}")


        # TODO logic wrong place
        # Filter out measurements where either dimension is < 1 mm or > 10 mm
        if dimA < 1 or dimB < 1 or dimA > 10 or dimB > 10:
            print("One of the dimensions is outside the valid range (1mm - 10mm). Skipping annotation.")
            return
        

        # TODO should be in a separate method
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

    def draw_midpoints_and_lines(self, image, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY):
        """
        Draws midpoints and lines between midpoints on the image.

        Args:
            image (numpy.ndarray): The image to annotate.
            tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY (float): Midpoint coordinates.

        Returns:
            None
        """

        # for every midpoint, draw a circle and a line connecting the midpoints
        for (x, y) in [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)]:
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    def process_rectangle_contour(self, cnt, image):
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

        # get midpoints
        (top_left, top_right, bottom_right, bottom_left) = box_points
        (tltrX, tltrY) = self.midpoint(top_left, top_right)
        (blbrX, blbrY) = self.midpoint(bottom_left, bottom_right)
        (tlblX, tlblY) = self.midpoint(top_left, bottom_left)
        (trbrX, trbrY) = self.midpoint(top_right, bottom_right)

        # Draw midpoints and lines
        self.draw_midpoints_and_lines(image, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY)

        return box_points 