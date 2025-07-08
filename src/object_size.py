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
            numpy.ndarray: The box points of the rectangle.
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

        Returns:
            None
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

        Args:
            image (numpy.ndarray): The input BGR image.

        Returns:
            None
        """
        contours = self.set_red_squire_ref_object_mask(image)
        for cnt in contours:
            if self.is_reference_rectangle(cnt):
                self.process_reference_rectangle(cnt, image)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

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


class SnailObject(ImageRuler):
    """
    Class for detecting and measuring snail objects (or other objects of interest).
    """
    def __init__(self):
        """
        Initializes the SnailObject class for detecting and measuring snail objects.

        Args:
            None

        Returns:
            None
        """
        # call the parent constructor
        super().__init__()

    def prep_image(self, image):
        """
        Preprocesses the input image by converting to grayscale, blurring, and performing edge detection.

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
        plt.imshow(cv2.cvtColor(blured, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
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
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Detected Circles")
            plt.show()
            # Sort circles by radius, descending
            circles_sorted = sorted(circles, key=lambda c: c[2], reverse=True)
            # Use the second largest circle if available, otherwise fallback to the largest
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

            plt.imshow(cv2.cvtColor(clipped, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
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
        
        # Find contours in the edged image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = imutils.contours.sort_contours(cnts)
        # cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
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

    Args:
        None

    Returns:
        dict: Dictionary of parsed command-line arguments.
    """
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(argparser.parse_args())
    return args

if __name__ == "__main__":
    args = get_args()

    image = cv2.imread(args["image"])

    # Set this to the real-world length of your reference object in mm
    ref_obj = ReferenceObject(reference_length_mm=10.42)
    pixels_per_metric = ref_obj.calculate_pixels_per_metric(image.copy())

    print(f"Pixels per metric: {pixels_per_metric}")

    # Snail object detection (example usage)
    snail_obj = SnailObject()
    snail_obj.pixels_per_metric = pixels_per_metric

    # Get the petri dish mask
    petri_mask = snail_obj.clip_petri_dish(image.copy())

    # Apply the mask to the original image for contour detection
    masked_image = cv2.bitwise_and(image, image, mask=petri_mask if len(petri_mask.shape) == 2 else cv2.cvtColor(petri_mask, cv2.COLOR_BGR2GRAY))
    edged = snail_obj.prep_image(masked_image)
    
    # Annotate contours on the original image (in-place)
    annotated_image = image.copy()
    snail_obj.get_contours(edged, annotated_image)

    # Show the annotated original image at the end
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
