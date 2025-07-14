import cv2
from matplotlib.pyplot import imshow
import rawpy
import matplotlib.pyplot as plt 


def get_args():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--image", required=True, help="path to the input image")
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