import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from src.reference_object import ReferenceObject
from src.snail_measurer import SnailMeasurer
from src.snail_inspect_window import SnailInspectWindow

class SnailGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Snail Measurement GUI")
        self.root.geometry("1000x700")
        self.root.configure(bg="#d0e7f9")

        self.original_loaded_image = None
        self.detected_snails = {}
        self.snail_obj = None
        self.current_snail_idx = tk.IntVar(value=0)
        self.reference_obj_width_mm = 10.42


        self.setup_layout()

    def setup_layout(self):
        """"
        Sets up the GUI layout with left and right panels, image display, and navigation controls.
        """
        # Left panel
        self.left_frame = ttk.Frame(self.root, width=250)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Original Image section
        self.original_image_frame = ttk.LabelFrame(self.left_frame, text="Image Sample", width=250, height=150)
        self.original_image_frame.pack(fill="x", pady=10)
        self.image_label = tk.Label(self.original_image_frame, text="original image")
        self.image_label.pack(expand=True)

        # Tools/Filters section
        self.tools_frame = ttk.LabelFrame(self.left_frame, text="Select between various tools or filters", width=250, height=200)
        self.tools_frame.pack(fill="x", pady=10)

        # Checkbuttons for contour options
        self.draw_contours_var = tk.BooleanVar(value=True)
        self.draw_measurements_var = tk.BooleanVar(value=True)
        self.draw_bounding_box_var = tk.BooleanVar(value=True)

        self.cb_draw_contours = tk.Checkbutton(self.tools_frame, text="Draw All Contours", variable=self.draw_contours_var, command=self.update_processed_image)
        self.cb_draw_contours.pack(anchor="w")
        self.cb_draw_measurements = tk.Checkbutton(self.tools_frame, text="Draw Measurements", variable=self.draw_measurements_var, command=self.update_processed_image)
        self.cb_draw_measurements.pack(anchor="w")
        self.cb_draw_bounding_box = tk.Checkbutton(self.tools_frame, text="Draw Bounding Box", variable=self.draw_bounding_box_var, command=self.update_processed_image)
        self.cb_draw_bounding_box.pack(anchor="w")

        # Right panel for processed image
        self.right_frame = ttk.LabelFrame(self.root, text="Processed Image")
        self.right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        self.processed_label = tk.Label(self.right_frame, text="Processed image")
        self.processed_label.pack(expand=True)


        # Add buttons
        self.select_img_btn = ttk.Button(self.left_frame, text="Select Image", command=self.select_image)
        self.select_img_btn.pack(pady=10)

        self.process_btn = ttk.Button(self.left_frame, text="Find Snails", command=self.measure_snails)
        self.process_btn.pack(pady=10)

        self.view_snail_btn = ttk.Button(self.left_frame, text="View Single Snail", command=self.view_single_snail)
        self.view_snail_btn.pack(pady=10)

    def select_image(self):
        """
        Opens a file dialog to select an image and displays it in the GUI in the original image frame.
        """
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )

        if file_path:
            try:
                img = Image.open(file_path)
                # set original_loaded_image to the full unedited resolution image
                self.original_loaded_image = cv2.imread(file_path)
                img = img.resize((300, 200), resample=Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk, text="")
                self.image_label.image = img_tk

                try:
                    self.measure_snails()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process image:\n check your picture setup\n{e}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def measure_snails(self):
        """
        Processes the loaded image to detect snails and display the results in the processed image frame.
        """
        if self.original_loaded_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Calculate pixels per metric using the reference object
        #TODO show check to user if the reference object is correct
        #TODO check if this should be done here or in the SnailMeasurer class
        ref_obj = ReferenceObject(reference_length_mm=self.reference_obj_width_mm)
        pixels_per_metric = ref_obj.calculate_pixels_per_metric(self.original_loaded_image.copy())

        self.snail_obj = SnailMeasurer()
        self.snail_obj.pixels_per_metric = pixels_per_metric

        edged = self.snail_obj.prep_image(self.original_loaded_image)
        annotated_image = self.original_loaded_image.copy()
        self.detected_snails = self.snail_obj.get_snail_contours(
            edged, annotated_image,
            draw_contours_all=self.draw_contours_var.get(),
            draw_measurements=self.draw_measurements_var.get(),
            draw_bounding_box=self.draw_bounding_box_var.get()
        )

        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_image_rgb)
        img_pil = img_pil.resize((600, 400), resample=Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.processed_label.config(image=img_tk, text="")
        self.processed_label.image = img_tk

    def inspect_single_snail(self):
        """
        Opens the inspect window for a single snail.
        """
        SnailInspectWindow(self.root, self.original_loaded_image, detected_snails=self.detected_snails)

    def view_single_snail(self):
        """
        Displays the first detected snail in a new window.
        """
        if not self.detected_snails:
            messagebox.showerror("Error", "No snails detected! Run 'Find Snails' first.")
            return
        first_snail = list(self.detected_snails.values())[0]
        self.inspect_single_snail()



    def update_processed_image(self):
        """
        Updates the processed image display when a Checkbutton is changed.
        """
        if self.original_loaded_image is None or self.snail_obj is None:
            return
        annotated_image = self.original_loaded_image.copy()
        edged = self.snail_obj.prep_image(self.original_loaded_image)
        self.detected_snails = self.snail_obj.get_snail_contours(
            edged, annotated_image,
            draw_contours_all=self.draw_contours_var.get(),
            draw_measurements=self.draw_measurements_var.get(),
            draw_bounding_box=self.draw_bounding_box_var.get()
        )
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_image_rgb)
        img_pil = img_pil.resize((600, 400), resample=Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.processed_label.config(image=img_tk, text="")
        self.processed_label.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = SnailGUI(root)
    root.mainloop()
