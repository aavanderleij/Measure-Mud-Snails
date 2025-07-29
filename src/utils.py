import cv2
from matplotlib.pyplot import imshow
import rawpy
import matplotlib.pyplot as plt 


def get_args():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--image", required=True, help="path to the input image")
    argparser.add_argument("-r", "--reference_length_mm", type=float, default=10.42, help="length of the reference object in mm")
    argparser.add_argument("-o", "--output", type=str, default="output.csv", help="path to save the output files")
    argparser.add_argument("-k", "--pos_key", type=int, default=0, help="position key for the snail measurements")
    argparser.add_argument("-sub", "--subsample", type=int, default=1, help="subsample factor for the image")
    argparser.add_argument("--round", type=bool, default=False, help="whether to round the measurements")

    args = vars(argparser.parse_args())
    return args

def get_input_img(path):

    if path.lower().endswith('.nef'):
        print(f"Reading NEF file from {path}")
        with rawpy.imread(path) as raw:
            image = raw.postprocess(rawpy.Params(use_camera_wb=True))
            # Convert RGB to BGR for OpenCV compatibility
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"Reading image file from {path}")
        image = cv2.imread(path)
    return image


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

def draw_midpoints_and_lines(image, box_points):
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
        (tltrX, tltrY) = midpoint(top_left, top_right)
        (blbrX, blbrY) = midpoint(bottom_left, bottom_right)
        (tlblX, tlblY) = midpoint(top_left, bottom_left)
        (trbrX, trbrY) = midpoint(top_right, bottom_right)

        # for every midpoint, draw a circle and a line connecting the midpoints
        for (x, y) in [(tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)]:
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        return image, tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY

def annotate_dimensions(image, name, dimA, dimB, box_points):
        """
        Annotates the image with the measured dimensions of the detected rectangle.

        Args:
            image (numpy.ndarray): The image to annotate.
            name (str): The label to annotate on the left side of the rectangle.
            dimA (float): First dimension in mm.
            dimB (float): Second dimension in mm.
            box_points (numpy.ndarray): The 4 points of the rectangle.

        Returns:
            None
        """

        # Unpack box points
        (tl, tr, br, bl) = box_points

        # Calculate midpoints using the midpoint function from utils
        tltr = midpoint(tl, tr)
        blbr = midpoint(bl, br)
        tlbl = midpoint(tl, bl)
        trbr = midpoint(tr, br)

        # Annotate the dimensions on the image
        cv2.putText(image, "{:.1f}mm".format(dimA),
                    (int(tltr[0] - 15), int(tltr[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 255, 255), 3)
        cv2.putText(image, "{:.1f}mm".format(dimB),
                    (int(trbr[0] + 10), int(trbr[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 255, 255), 3)
        # Annotate the name on the left side of the rectangle
        cv2.putText(image, name,
                    (int(tlbl[0] - 80), int(tlbl[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (255, 0, 0), 3)