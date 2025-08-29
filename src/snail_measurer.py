"""
Module for detecting and measuring snails in images.
"""

import os
import csv
import cv2
import imutils
from imutils import perspective
from imutils import contours
from matplotlib import pyplot as plt
import numpy as np
from src.image_ruler import ImageRuler
from src.snail_object import SnailObject
from src.utils import draw_midpoints_and_lines, annotate_dimensions


class SnailMeasurer(ImageRuler):
    """
    Class for detecting and measuring snail objects (or other objects of interest).
    """

    def prep_image(self, image):
        """
        Detects both dark and light snails using bilateral filtering and edge detection.
        Returns a combined edge mask.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Bilateral filter
        blured = cv2.bilateralFilter(gray, d=9, sigmaColor=50, sigmaSpace=50)

        # Detect dark snails (edges of dark regions)
        dark_edges = cv2.Canny(blured, 30, 100)

        # Detect light snails (invert image, then edges)
        inverted = cv2.bitwise_not(blured)
        light_edges = cv2.Canny(inverted, 30, 100)

        # Combine both edge masks
        combined = cv2.bitwise_or(dark_edges, light_edges)

        # Morphological operations to clean up
        combined = cv2.dilate(combined, None, iterations=2)
        combined = cv2.erode(combined, None, iterations=1)

        # self.show_image(combined, title="Combined Snail Edges")
        return combined


    def clip_petri_dish(self, image):
        """
        Clips the image to the largest detected circle (petri dish).

        Args:
            image (numpy.ndarray): The input BGR image.

        Returns:
            numpy.ndarray: The clipped image containing only the petri dish.
            If no circle is found, returns the original image.
        """
        print("Clipping petri dish...")
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Optional: Enhance contrast
        # gray = cv2.equalizeHist(gray)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Detect circles using Hough Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=150,
            maxRadius=2000
        )
        if circles is not None:

            circles = np.round(circles[0, :]).astype("int")

            # Draw all detected circles for inspection
            output = image.copy()
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            # Show the output with detected circles
            self.show_image(output, title="Detected Circles")

            # Sort circles by radius, descending
            circles_sorted = sorted(circles, key=lambda c: c[2], reverse=True)

            # Use the second largest circle if available, otherwise fallback to the largest
            # second largest circle is likely the inside rim of the petri dish
            if len(circles_sorted) > 1:
                x, y, r = circles_sorted[0]
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

            # Show the clipped image
            self.show_image(clipped, title="Clipped Petri Dish")

            return mask
        else:
            print("No circles found in the image.")
            # If no circle found, return original image
            return image


    def bin_measuments(self, snails, attribute="length"):
        """
        Takes all measurements and groups them in bins to the closest half millimeter.
        Args:
            snails (dict): Dictionary of SnailObject instances.
            attribute (str): "length" or "width" to bin.
        Returns:
            dict: {bin_size: count}
        """
        values = [getattr(snail, attribute) for snail in snails.values()]
        # Bin to nearest 0.5 mm
        bins = [round(v * 2) / 2 for v in values]
        # Count occurrences
        bin_counts = {}
        for b in bins:
            bin_counts[b] = bin_counts.get(b, 0) + 1
        # Sort bins descending
        sorted_bins = sorted(bin_counts.items(), reverse=True)
        print("size\tN in bin")
        for size, count in sorted_bins:
            print(f"{size}\t{count}")
        print(dict(sorted_bins))
        return dict(sorted_bins)


    def export_to_csv(self, snails):
        """
        Exports snail measurements to a CSV file.

        Args:
            snails (dict): Dictionary of SnailObject instances.
        """
        # Create a directory for the CSV files if it doesn't exist
        os.makedirs("snail_measurements", exist_ok=True)

        for _, snail_obj in snails.items():
            self.write_snail_to_csv(snail_obj)

    def write_snail_to_csv(self, snail_obj):
        """
        Writes the snail's name, length, and width to a CSV file.
        #TODO discuss if we want a mega file with all measurements of separate csv per image

        Args:
            name (str): Snail identifier.
            length (float): Length in mm.
            width (float): Width in mm.
            filename (str): CSV file name.
        """

        file_exists = os.path.isfile(f"snail_measurements_{snail_obj.pos_key}.csv")
        with open(f"snail_measurements_{snail_obj.pos_key}.csv",
                    mode='a', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Name", "PosKey", "Length (mm)", "Width (mm)"])
            writer.writerow([snail_obj.snail_id, snail_obj.pos_key,
                             snail_obj.length, snail_obj.width])

    def draw_snails(self, image, snails, draw_contours=True,
                    draw_measurements=True, draw_bounding_box=True):
        """
        Draws all detected snails on the image.

        Args:
            image (numpy.ndarray): The image to annotate.
            snails (dict): Dictionary of SnailObject instances.

        Returns:
            numpy.ndarray: The annotated image.
        """
        annotated_image = image.copy()
        for snail in snails.values():
            annotated_image = self.draw_single_snail(annotated_image, snail,
                                                     draw_contours=draw_contours,
                                                      draw_measurements=draw_measurements,
                                                      draw_bounding_box=draw_bounding_box)

        return annotated_image

    def get_snail_contours(self, edged, image, draw_contours_all=True,
                            draw_measurements=True, draw_bounding_box=True):
        """
        Finds and draws contours on the image, highlighting each detected object.

        Args:
            edged (numpy.ndarray): The edge-detected image.
            image (numpy.ndarray): The image to annotate.

        Returns:
            None
        """

        cnts = self.get_contours(edged)

        # draw all contours on the image
        if draw_contours_all:
            cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
        print(f"Number of contours with area > 1000: \
              {sum(1 for contour in cnts if cv2.contourArea(contour) > 1000)}")

        # set ID for the contour
        snails = {}
        snail_id = 1
        for contour in cnts:
            # Skip small contours
            if cv2.contourArea(contour) < 1000:
                continue

            else:
                box_points = self.get_min_area_rect_box(contour)
                # get mesurements in mm
                dim_a, dim_b = self.get_dimensions_in_mm(box_points)

                # Filter out measurements where either dimension is < 1 mm or > 10 mm
                if dim_a < 1 or dim_b < 1 or dim_a > 10 or dim_b > 10:
                    continue

                else:
                    name = f"S{snail_id}"
                    # draw rectangle contour
                    if draw_bounding_box:
                        cv2.drawContours(image, [box_points.astype("int")], -1, (0, 255, 0), 2)
                    # draw midpoints and lines
                    image, _, _, _, _, _, _, _, _ = draw_midpoints_and_lines(image, box_points)
                    # Annotate the dimensions on the image
                    if draw_measurements:
                        annotate_dimensions(image, name, dim_a, dim_b, box_points)
                    # add the id to the countour

                    # write the snail measurements to csv
                    # if dim_a is greater than dim_b, DimA is the length and DimB is the width
                    if dim_a > dim_b:
                        length, width = dim_a, dim_b
                    else:
                        length, width = dim_b, dim_a

                    snail = SnailObject(
                        snail_id=name,
                        length=length,
                        width=width,
                        contour=contour,
                        bounding_box=box_points
                    )

                    snails[name] = snail
                    snail_id += 1

        return snails

    def draw_single_snail(self, image, snail, draw_contours=True,
                          draw_measurements=True, draw_bounding_box=True):
        """
        Draws a single snail's contour and dimensions on the image.

        Args:
            image (numpy.ndarray): The image to annotate.
            snail (SnailObject): The snail object containing its details.

        Returns:
            numpy.ndarray: The annotated image.
        """
        box_points = snail.bounding_box
        dim_a, dim_b = self.get_dimensions_in_mm(box_points)

        if draw_contours:
            # Draw the contour of the snail
            cv2.drawContours(image, [snail.contour], -1, (0, 255, 0), 2)
        # Draw rectangle contour
        if draw_bounding_box:
            cv2.drawContours(image, [box_points.astype("int")], -1, (0, 255, 0), 2)
        # Draw midpoints and lines
        image, _, _, _, _, _, _, _, _ = draw_midpoints_and_lines(image, box_points)
        # Annotate the dimensions on the image
        annotate_dimensions(image, snail.snail_id, dim_a, dim_b,
                            box_points, draw_measurements=draw_measurements, draw_id=True)

        return image
