from src.reference_object import ReferenceObject
from src.snail_measurer import SnailMeasurer
from src.utils import get_args, get_input_img
import cv2
import matplotlib.pyplot as plt

def main():
    # Parse command line arguments
    
    args = get_args()

    # image = cv2.imread(args["image"])
    image = get_input_img(args["image"])

    # Create a ReferenceObject instance and calculate pixels per metric
    ref_obj = ReferenceObject(reference_length_mm=10.42)
    pixels_per_metric = ref_obj.calculate_pixels_per_metric(image.copy())
    print(f"Pixels per metric: {pixels_per_metric}")

    # Create a SnailMeasurer instance and clip the petri dish
    snail_obj = SnailMeasurer()
    snail_obj.pixels_per_metric = pixels_per_metric

    # petri_mask = snail_obj.clip_petri_dish(image.copy())

    # #TODO should be in a separate method in SnailMeasurer
    # masked_image = cv2.bitwise_and(
    #     image, image,
    #     mask=petri_mask if len(petri_mask.shape) == 2 else cv2.cvtColor(petri_mask, cv2.COLOR_BGR2GRAY)
    # )
    # Prepare the image for contour detection
    edged = snail_obj.prep_image(image)
    
    annotated_image = image.copy()
    snails = snail_obj.get_snail_contours(edged, annotated_image, args["pos_key"])

    print("Detected snails:")
    for snail_id, snail in snails.items():
        print(f"ID: {snail_id}, Length: {snail.length:.2f} mm, Width: {snail.width:.2f} mm")

    # mask = snail_obj.prep_image(image)
    # separated_mask = snail_obj.separate_touching_snails(mask)
    # snail_obj.get_snail_contours(separated_mask,image.copy())

    # snail_obj.annotate_dimensions(annotated_image)

    # plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    # plt.show()

if __name__ == "__main__":
    main()