"""
This module has functions that have to do with inspecting induvirual snails
"""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
from src.utils import annotate_dimensions


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


class SnailInspectWindow:
    def __init__(self, parent, image, detected_snails):
        """
        Opens a new window to inspect a single snail.
        Args:
            parent: The parent Tkinter root or window.
            image: The full image (numpy array, BGR).
            detected_snails: dict of snails {id: SnailObject}
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Snail Inspector")
        self.detected_snails = detected_snails
        self.current_snail_idx = tk.IntVar(value=0)
        self.image = image
        self.core = SnailInspectorCore(image, detected_snails)  # Core logic
        self.setup_window()

    def setup_window(self):
        """
        Setup for the pannel where the display where the annotad image is shown.
        also contains a navigation bar at the bottom.
        """
        # Right panel for processed image
        self.center_frame = ttk.LabelFrame(self.window, text="Processed Image")
        self.center_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        self.processed_label = tk.Label(self.center_frame, text="Processed image")
        self.processed_label.pack(expand=True)

        # Navigation interface
        self.nav_frame = ttk.Frame(self.center_frame)
        self.nav_frame.pack(side="bottom", pady=10)

        self.prev_btn = ttk.Button(self.nav_frame, text="<-", width=3, command=self.prev_snail)
        self.prev_btn.pack(side="left")

        self.snail_id_entry = tk.Entry(self.nav_frame, width=5, justify="center")
        self.snail_id_entry.pack(side="left", padx=2)

        self.next_btn = ttk.Button(self.nav_frame, text="->", width=3, command=self.next_snail)
        self.next_btn.pack(side="left")

        self.goto_btn = ttk.Button(self.nav_frame, text="Go", width=3, command=self.goto_snail)
        self.goto_btn.pack(side="left", padx=2)

        # Show the first snail on startup
        self.update_snail_display()

    def draw_single_snail(self, snail):
        """
        Draws a single snail on the image and updates the label.
        """
        annotated_image = self.image.copy()
        # Try to use draw_single_snail from SnailMeasurer if available
        # Otherwise, just draw the contour
        if hasattr(snail, "contour"):
            cv2.drawContours(annotated_image, [snail.contour], -1, (0,255,0), 2)

        if hasattr(snail, "bounding_box"):
            box_points = snail.bounding_box
            cv2.polylines(annotated_image, [box_points.astype("int")], True, (0, 255, 0), 2)

        # Annotate dimensions
        if hasattr(snail, "length") and hasattr(snail, "width"):
            annotate_dimensions(annotated_image, snail.snail_id,
                                snail.length, snail.width, box_points)
        # If you have a SnailMeasurer instance, you can use its method here


        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_image_rgb)
        img_pil = img_pil.resize((400, 300), resample=Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.processed_label.config(image=img_tk, text="")
        self.processed_label.image = img_tk

    def update_snail_display(self):
        """
        Shows the currently selected snail based on the current index.
        """
        snail_keys = list(self.detected_snails.keys())
        if not snail_keys:
            messagebox.showerror("Error", "No snails detected! Run 'Find Snails' first.")
            return
        idx = self.current_snail_idx.get()
        if idx < 0:
            idx = 0
        if idx >= len(snail_keys):
            idx = len(snail_keys) - 1
        self.current_snail_idx.set(idx)
        snail = self.detected_snails[snail_keys[idx]]
        self.draw_single_snail(snail)
        self.snail_id_entry.delete(0, tk.END)
        self.snail_id_entry.insert(0, str(snail_keys[idx]))

    def prev_snail(self):
        """ 
        get the snail with the current index -1 and update the display
        """
        if self.detected_snails:
            self.current_snail_idx.set(max(0, self.current_snail_idx.get() - 1))
            self.update_snail_display()

    def next_snail(self):
        """ 
        get the snail with the current index +1 and update the display
        """
        if self.detected_snails:
            self.current_snail_idx.set(min(len(self.detected_snails) - 1,
                                           self.current_snail_idx.get() + 1))
            self.update_snail_display()

    def goto_snail(self):
        """ 
        get the snail with the id the entry field and update the display
        """
        if self.detected_snails:
            snail_keys = list(self.detected_snails.keys())
            val = self.snail_id_entry.get()
            try:
                idx = int(val)
                if 0 <= idx < len(snail_keys):
                    self.current_snail_idx.set(idx)
                    self.update_snail_display()
                    return
            except ValueError:
                pass
            if val in snail_keys:
                idx = snail_keys.index(val)
                self.current_snail_idx.set(idx)
                self.update_snail_display()
            else:
                messagebox.showerror("Error", f"Snail '{val}' not found.")
