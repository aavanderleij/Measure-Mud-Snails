import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageRuler():
    def __init__(self):
        # width and height of a4 paper
        self.ref_object_width = 210
        self.ref_object_height = 297
        self.img_resized = None

    def load_image(self, path, scale=0.7):
        """
        Loads an image from the specified path and resizes it by the given scale.
        """
        # Read image from file
        img = cv2.imread(path)  
        # Resize image
        self.img_resized = cv2.resize(img, (0,0), None, scale, scale)

    def show_resized_image(self):
        """
        Displays the resized image using matplotlib.
        """
        self.show_image(self.img_resized)

    def show_image(self, image):
        """
        Displays the loaded and resized image using matplotlib.
        """
        # Convert the image from BGR to RGB format for correct color display in matplotlib
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6,8))

        # Remove x-axis and y-axis ticks (axis ticks and numbers)
        plt.xticks([])
        plt.yticks([]) 
        # plot the image
        plt.imshow(img_rgb)
        # show the plot with the image
        plt.show()
    
    def preprocess_image(self, image, thresh_1=57, thresh_2=232):
        """
        Preprocess the image (convert to grayscale, blur, edge detection, dilation, closing).
        Returns the preprocessed image and a dict of intermediate steps.
        """
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray, (5, 5), 1)
        image_canny = cv2.Canny(image_blur, thresh_1, thresh_2)

        kernel = np.ones((3, 3), np.uint8)
        image_dilated = cv2.dilate(image_canny, kernel, iterations=1)
        image_closed = cv2.morphologyEx(image_dilated, cv2.MORPH_CLOSE, kernel, iterations=4)

        image_preprocessed = image_closed.copy()

        image_each_step = {
            'image_gray': image_gray,
            'image_blur': image_blur,
            'image_canny': image_canny,
            'image_dilated': image_dilated
        }

        return image_preprocessed, image_each_step
    
    def show_preprocessed_image(self, image, thresh_1=57, thresh_2=232):
        """
        Preprocess the image and display the intermediate steps.
        """
        self.preprocessed_image, steps = self.preprocess_image(image, thresh_1, thresh_2)

        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.imshow(steps['image_gray'])
        plt.title('Grayscale')
        plt.axis('off')

        plt.subplot(232)
        plt.imshow(steps['image_blur'])
        plt.title('Blurred')
        plt.axis('off')

        plt.subplot(233)
        plt.imshow(steps['image_canny'])
        plt.title('Canny Edges')
        plt.axis('off')

        plt.subplot(234)
        plt.imshow(steps['image_dilated'])
        plt.title('Dilated')
        plt.axis('off')

        plt.subplot(235)
        plt.imshow(self.preprocessed_image)
        plt.title('Preprocessed Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def find_contours(self, image=None, epsilon_param=0.04):
        """
        Finds contours in the preprocessed image.
        Returns the contours and hierarchy.
        """

        if image is None:
            if self.img_resized is None:
                raise ValueError("No image loaded. Please load an image first.")
            image = self.preprocessed_image.copy()


        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.image_paper_contour = self.img_resized.copy()
        cv2.drawContours(self.image_paper_contour, contours, -1, (203,192,255), 6)

        polygons = []

        for contour in contours:
            epsilon = epsilon_param * cv2.arcLength(curve=contour, closed=True)

            polygon = cv2.approxPolyDP(curve=contour, epsilon=epsilon, closed=True)

            print(f"Found contour with {len(polygon)} points")

            if len(polygon) != 4:
                print("Skipping contour with less than 4 points")
                continue

            polygon = polygon.reshape(4, 2)

            polygons.append(polygon)

            for point in polygon:
                self.immage_with_contours = cv2.circle(img=self.image_paper_contour, center=point,
                                     radius=8, color=(0,240,0),
                                     thickness=-1)
        
        self.show_image(self.immage_with_contours)


        return contours, hierarchy

def main():
    # Main function to load and display an image
    print("Loading image...")

    # Create an instance of ImageRuler
    ruler = ImageRuler()

    # Load and resize image
    ruler.load_image('data/snails_on_a4.jpg', scale=0.7)
    # Display the image
    ruler.show_resized_image()
    # Show preprocessed image and intermediate steps
    ruler.show_preprocessed_image(ruler.img_resized, thresh_1=57, thresh_2=232)
    ruler.find_contours()

if __name__ == "__main__":
    sys.exit(main())