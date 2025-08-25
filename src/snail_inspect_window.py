"""
This module has functions that have to do with inspecting induvirual snails
"""
import cv2


class SnailInspectorCore:
    def __init__(self, image, detected_snails):
        self.image = image
        self.detected_snails = detected_snails
        self.current_snail_idx = 0
        self.deleted_snails = []

    def get_snail_keys(self):
        """
        return keys of detected snails
        """
        return list(self.detected_snails.keys())

    def get_current_snail(self):
        """
        return snail with the id stored in self.current_snail_idx
        """
        keys = self.get_snail_keys()
        if not keys:
            return None
        idx = max(0, min(self.current_snail_idx, len(keys) - 1))
        return self.detected_snails[keys[idx]], keys[idx]


    def goto_snail(self, idx_or_id):
        """
        Return the snail with the index
        """
        keys = self.get_snail_keys()
        if not keys:
            return

        if isinstance(idx_or_id, int):
            # Wrap around using modulo for both forward and backward
            idx = idx_or_id % len(keys)
        else:
            try:
                idx = keys.index(idx_or_id)
            except ValueError:
                idx = 0
        self.current_snail_idx = idx

    def next_snail(self):
        """
        To the snail with the snail index one higher than the current snail index
        """
        self.goto_snail(self.current_snail_idx + 1)

    def prev_snail(self):
        """
        To the snail with the snail index one lower than the current snail index
        """
        self.goto_snail(self.current_snail_idx - 1)

    def get_annotated_image(self, draw_func=None):
        """
        Returns an annotated version of the current image with drawing from draw_func

        args:
            draw_func (function): function used for drawing
        
        return:
            annotated_image (np.ndarray): The annotated image
            snail_id (str): The identifier of the current snail

        """
        snail, snail_id = self.get_current_snail()
        if snail is None:
            return None, None
        annotated_image = self.image.copy()
        if draw_func:
            annotated_image = draw_func(annotated_image, snail)
        else:
            # fallback: just draw contour
            if hasattr(snail, "contour"):
                cv2.drawContours(annotated_image, [snail.contour], -1, (0,255,0), 2)
        return annotated_image, snail_id


