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
    plt.imshow(image)
    plt.show()
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

