
from scipy.spatial import distance
from imutils import perspective
from imutils import contours
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageRuler:
    """
    Base class for image processing and measurement.
    """
    def __init__(self):
        self.pixels_per_metric = None
        self.image = None

    @staticmethod
    def midpoint(point_a, point_b):
        """
        Calculate the midpoint between two points.

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
        # Calculate distances in pixels
        dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # Convert to real-world units using the reference length
        pixels_per_metric = 78.63018599673104
        print(f"Distance A (pixels): {dA}, Distance B (pixels): {dB}")
        dimA = dA / pixels_per_metric
        dimB = dB / pixels_per_metric
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
        """
        rotated_rect = cv2.minAreaRect(cnt)
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
            None
        """
        box_points = self.get_min_area_rect_box(cnt)
        cv2.drawContours(image, [box_points.astype("int")], -1, (0, 255, 0), 2)

        (top_left, top_right, bottom_right, bottom_left) = box_points
        (tltrX, tltrY) = self.midpoint(top_left, top_right)
        (blbrX, blbrY) = self.midpoint(bottom_left, bottom_right)
        (tlblX, tlblY) = self.midpoint(top_left, bottom_left)
        (trbrX, trbrY) = self.midpoint(top_right, bottom_right)

        self.draw_midpoints_and_lines(image, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY)
        self.annotate_dimensions(image, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY)
        return box_points 

class ReferenceObject(ImageRuler):
    
    def __init__(self, reference_length_mm):
        """
        Initializes the ReferenceObject with a specified reference length in mm.

        Args:
            reference_length_mm (float): The real-world length of the reference object in millimeters.
        """
        super().__init__()
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
        self.process_rectangle_contour(cnt, image)

    def detect_and_annotate(self, image):
        """
        Detects red reference rectangles in the image and annotates them.
        """
        contours = self.set_red_squire_ref_object_mask(image)
        for cnt in contours:
            if self.is_reference_rectangle(cnt):
                self.process_reference_rectangle(cnt, image)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


class SnailObject(ImageRuler):
    """
    Class for detecting and measuring snail objects (or other objects of interest).
    """
    def __init__(self):
        super().__init__()

    def prep_image(self, image):
        """
        Preprocess the input image by converting to grayscale, blurring, and performing edge detection.
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
        plt.imshow(cv2.cvtColor(blured, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        return edged

    def get_contours(self, edged, image):
        """
        Finds and draws contours on the image, highlighting each detected object.
        """
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = imutils.contours.sort_contours(cnts)
        print(f"Found {len(cnts)} contours in the image.")
        print(f"Number of contours with area > 1000: {sum(1 for contour in cnts if cv2.contourArea(contour) > 1000)}")
        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
                continue
            self.process_rectangle_contour(contour, image)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    

    

def get_args():
    """
    Parses command-line arguments for the script.
    """
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(argparser.parse_args())
    return args

if __name__ == "__main__":
    args = get_args()


    image = cv2.imread(args["image"])
    reference_length_mm = 10.42  # Set this to the real-world length of your reference object in mm
    ref_obj = ReferenceObject(reference_length_mm=reference_length_mm)
    pixels_per_metric = ref_obj.detect_and_annotate(image.copy())

    print(f"Pixels per metric: {pixels_per_metric}")

    # Snail object detection (example usage)
    snail_obj = SnailObject()
    edged = snail_obj.prep_image(image.copy())
    snail_obj.get_contours(edged, image.copy())


    # image_color_filtered = SnailObject.hsv_filter(image, hue_min=0.05, hue_max=0.22)
    # You can call snail_obj.get_contours(edged, image.copy()) to visualize snail contours
