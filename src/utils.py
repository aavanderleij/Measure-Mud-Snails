def get_args():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(argparser.parse_args())
    return args