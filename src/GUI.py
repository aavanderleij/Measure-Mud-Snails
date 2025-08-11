"""
A GUI application for measuring mud snails using image processing.
This application allows users to select an image, process it to detect snails,
and display measurements. It also provides functionality to save measurements to a CSV file.
It uses the tkinter library for the GUI, PIL for image handling, and OpenCV for image processing.
It includes features for navigating through detected snails, deleting snails, and updating the
display based on user input.
It also includes input fields for sample data such as position key, subsample, project, species
"""
import csv
from datetime import datetime
import glob
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from src.reference_object import ReferenceObject
from src.snail_measurer import SnailMeasurer
from src.snail_inspect_window import SnailInspectorCore
from src.digi_cam_control_python import Camera

class SnailGUI:
    """
    A GUI application for measuring mud snails using image processing.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Snail Measurement GUI")
        self.root.geometry("1440x900")
        self.root.configure(bg="#d0e7f9")

        # Camera
        self.camera = Camera()
        self.output_path = os.path.join(os.getcwd(), "output")

        self.file_path = None
        self.original_loaded_image = None
        self.annotated_image_rgb = None
        self.detected_snails = {}
        self.snail_measurer = None
        self.current_snail_idx = tk.IntVar(value=0)
        self.reference_obj_width_mm = 10
        self.pos_key = None
        # Typing delay for input fields (in milliseconds)
        self.typing_delay = 500
        self.after_id = None
        self.inspector = None  # Will be set after detection
        self.deleted_snails = []

        self.species = None
        self.subsample = None
        self.lab_method_code = None
        self.analyst = None
        self.project = None
        self.year = None

        self.setup_layout()

    def setup_layout(self):
        """"
        Sets up the GUI layout with left and right panels, image display, and navigation controls.
        """
        # Left panel
        self.left_frame = ttk.Frame(self.root, width=250)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Original Image section
        self.original_image_frame = ttk.LabelFrame(self.left_frame,
                                                   text="Image Sample",
                                                    width=250, height=150)
        self.original_image_frame.pack(fill="x", pady=10)
        self.image_label = tk.Label(self.original_image_frame, text="original image")
        self.image_label.pack(expand=True)

        # Tools/Filters section
        self.tools_frame = ttk.LabelFrame(self.left_frame,
                                          text="Select between various tools or filters",
                                            width=250, height=200)
        self.tools_frame.pack(fill="x", pady=10)

        # Sample data
        self.input_data_frame = ttk.LabelFrame(self.left_frame, text="Sample data",
                                               width=250, height=100)
        self.input_data_frame.pack(fill="x", pady=10)

        # Checkbuttons for contour options
        self.draw_contours_var = tk.BooleanVar(value=True)
        self.draw_measurements_var = tk.BooleanVar(value=True)
        self.draw_bounding_box_var = tk.BooleanVar(value=True)

        self.cb_draw_contours = tk.Checkbutton(self.tools_frame, text="Draw All Contours",
                                                variable=self.draw_contours_var,
                                                command=self.update_processed_image)
        self.cb_draw_contours.pack(anchor="w")
        self.cb_draw_measurements = tk.Checkbutton(self.tools_frame, text="Draw Measurements",
                                                    variable=self.draw_measurements_var,
                                                    command=self.update_processed_image)
        self.cb_draw_measurements.pack(anchor="w")
        self.cb_draw_bounding_box = tk.Checkbutton(self.tools_frame, text="Draw Bounding Box",
                                                    variable=self.draw_bounding_box_var,
                                                    command=self.update_processed_image)
        self.cb_draw_bounding_box.pack(anchor="w")

        # Right panel for processed image
        self.right_frame = ttk.LabelFrame(self.root, text="Processed Image")
        self.right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        self.processed_label = tk.Label(self.right_frame, text="Processed image")
        self.processed_label.pack(expand=True)

        # Navigation interface for single snail
        self.nav_frame = ttk.Frame(self.right_frame)
        self.nav_frame.pack(side="bottom", pady=10)

        self.prev_btn = ttk.Button(self.nav_frame, text="<-", width=3, command=self.prev_snail)
        self.prev_btn.pack(side="left")

        self.snail_id_entry = tk.Entry(self.nav_frame, width=5, justify="center")
        self.snail_id_entry.pack(side="left", padx=2)

        self.next_btn = ttk.Button(self.nav_frame, text="->", width=3, command=self.next_snail)
        self.next_btn.pack(side="left")

        self.goto_btn = ttk.Button(self.nav_frame, text="Go", width=3, command=self.goto_snail)
        self.goto_btn.pack(side="left", padx=2)

        self.delete_btn = ttk.Button(self.nav_frame, text="Delete snail", width=12,
                                      command=self.delete_snail)
        self.delete_btn.pack(side="left", padx=2)

        # Input fields for Sample data
        self.pos_key_label = tk.Label(self.input_data_frame, text="Pos Key:")
        self.pos_key_label.grid(row=0, column=0, sticky="w")
        self.pos_key_entry = tk.Entry(self.input_data_frame)
        self.pos_key_entry.bind("<KeyRelease>", self.get_pos_key)
        self.pos_key_entry.grid(row=0, column=1, sticky="ew")

        self.subsample_label = tk.Label(self.input_data_frame, text="Subsample:")
        self.subsample_label.grid(row=1, column=0, sticky="w")
        self.subsample_entry = tk.Entry(self.input_data_frame)
        self.subsample_entry.bind("<KeyRelease>", self.get_subsample)
        self.subsample_entry.grid(row=1, column=1, sticky="ew")

        self.project_label = tk.Label(self.input_data_frame, text="Project:")
        self.project_label.grid(row=2, column=0, sticky="w")
        self.project_entry = tk.Entry(self.input_data_frame)
        self.project_entry.bind("<KeyRelease>", self.get_project)
        self.project_entry.insert(0, "SIBES")  # Default
        self.project_entry.grid(row=2, column=1, sticky="ew")

        self.species_label = tk.Label(self.input_data_frame, text="Species:")
        self.species_label.grid(row=3, column=0, sticky="w")
        self.species_entry = tk.Entry(self.input_data_frame)
        self.species_entry.bind("<KeyRelease>", self.get_species)
        self.species_entry.insert(0, "Peringia")  # Default
        self.species_entry.grid(row=3, column=1, sticky="ew")


        self.analyst_label = tk.Label(self.input_data_frame, text="Analyst:")
        self.analyst_label.grid(row=4, column=0, sticky="w")
        self.analyst_entry = tk.Entry(self.input_data_frame)
        self.analyst_entry.bind("<KeyRelease>", self.get_analyst)
        self.analyst_entry.grid(row=4, column=1, sticky="ew")

        self.year_label = tk.Label(self.input_data_frame, text="Year:")
        self.year_label.grid(row=5, column=0, sticky="w")
        self.year_entry = tk.Entry(self.input_data_frame)
        self.year_entry.bind("<KeyRelease>", self.get_year)
        self.year_entry.grid(row=5, column=1, sticky="ew")

        self.lab_method_code_label = tk.Label(self.input_data_frame, text="Lab Method Code:")
        self.lab_method_code_label.grid(row=6, column=0, sticky="w")
        self.lab_method_code_entry = tk.Entry(self.input_data_frame)
        self.lab_method_code_entry.bind("<KeyRelease>", self.get_lab_method_code)
        self.lab_method_code_entry.insert(0, 19) # Default
        self.lab_method_code_entry.grid(row=6, column=1, sticky="ew")

        # Add buttons
        self.take_photo_btn = ttk.Button(self.left_frame, text="Take Photo",
                                         command=self.capture)
        self.take_photo_btn.pack(pady=10)

        self.select_img_btn = ttk.Button(self.left_frame, text="Select Image",
                                         command=self.select_image)
        self.select_img_btn.pack(pady=10)

        self.process_btn = ttk.Button(self.left_frame, text="Select output folder",
                                      command=self.select_output_folder)
        self.process_btn.pack(pady=10)

        self.save_btn = ttk.Button(self.left_frame, text="Save Measurements",
                                   command=self.write_measurements_to_csv)
        self.save_btn.pack(pady=10)

        self.plot_btn = ttk.Button(self.left_frame, text="View as Plot",
                                   command=self.view_image_as_plot)
        self.plot_btn.pack(pady=10)

    def select_image(self):
        """
        Opens a file dialog to select an image and displays it in the GUI in 
        the original image frame.
        """
        self.file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )

        if self.file_path:
            print(f"Selected file: {self.file_path}")
            try:
                img = Image.open(self.file_path)
                # set original_loaded_image to the full unedited resolution image
                self.original_loaded_image = cv2.imread(self.file_path)
                img = img.resize((300, 200), resample=Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk, text="")
                self.image_label.image = img_tk

                try:
                    self.measure_snails()
                except Exception as e:
                    messagebox.showerror("Error",
                                        f"Failed to process image:\n check your picture setup\n{e}")

            except Exception as e:
                messagebox.showerror("Error",
                                     f"Failed to open image:\n{e}")

    def select_latest_image(self):
        """
        Opens a file dialog to select the latest image
        and displays it in the GUI in the original image frame.
        """
        folder_path = os.path.dirname(self.file_path)
        list_of_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        latest_file = max(list_of_files, key=os.path.getctime)

        self.file_path = latest_file
        print(f"Selected latest file: {self.file_path}")
        try:
            img = Image.open(self.file_path)
            # set original_loaded_image to the full unedited resolution image
            self.original_loaded_image = cv2.imread(self.file_path)
            img = img.resize((300, 200), resample=Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk

            try:
                self.measure_snails()
            except Exception as e:
                messagebox.showerror("Error",
                                     f"Failed to process image:\n check your picture setup\n{e}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def capture(self):
        """
        activate digiCamControl and take photo
        """

        self.camera.set_folder(self.output_path)

        self.camera.capture()

        print("smile :)")

    def select_output_folder(self):
        """
        Opens a file dialog to select the output folder for saving images.
        """
        self.output_path = filedialog.askdirectory(
            title="Select Output Folder"
        )
        if self.output_path:
            print(f"Selected output folder: {self.output_path}")

    def measure_snails(self):
        """
        Processes the loaded image to detect snails and display the results in
         the processed image frame.
        """
        if self.original_loaded_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Calculate pixels per metric using the reference object
        #TODO show check to user if the reference object is correct
        #TODO check if this should be done here or in the SnailMeasurer class


        # reset the detected snails
        self.detected_snails = None
        self.snail_measurer = None
        self.inspector = None
        self.deleted_snails = []

        ref_obj = ReferenceObject(reference_length_mm=self.reference_obj_width_mm)
        pixels_per_metric = ref_obj.calculate_pixels_per_metric(self.original_loaded_image.copy())

        # get the Pixels Per Metric form the image
        self.snail_measurer = SnailMeasurer()
        self.snail_measurer.pixels_per_metric = pixels_per_metric

        edged = self.snail_measurer.prep_image(self.original_loaded_image)
        annotated_image = self.original_loaded_image.copy()
        self.detected_snails = self.snail_measurer.get_snail_contours(
            edged, annotated_image,
            draw_contours_all=self.draw_contours_var.get(),
            draw_measurements=self.draw_measurements_var.get(),
            draw_bounding_box=self.draw_bounding_box_var.get()
        )

        self.annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(self.annotated_image_rgb)
        img_pil = img_pil.resize((800, 500), resample=Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.processed_label.config(image=img_tk, text="")
        self.processed_label.image = img_tk

        self.inspector = SnailInspectorCore(self.original_loaded_image, self.detected_snails)

    def update_single_snail_display(self):
        """
        Updates the display for a single detected snail.
        """
        if not self.inspector or not self.detected_snails:
            return
        annotated_image, snail_id = self.inspector.get_annotated_image(
            draw_func=self.snail_measurer.draw_single_snail if self.snail_measurer else None
        )
        if annotated_image is None:
            return
        self.annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(self.annotated_image_rgb)
        img_pil = img_pil.resize((800, 500), resample=Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.processed_label.config(image=img_tk, text="")
        self.processed_label.image = img_tk
        self.snail_id_entry.delete(0, tk.END)
        self.snail_id_entry.insert(0, str(snail_id))

    def view_image_as_plot(self):
        """
        funtion for vieuw as plot button.
        Allowes the user to open the processed image as a matplotlib plot for
        easier zoom functionality
        """
        self.snail_measurer.show_image(self.annotated_image_rgb)

    def prev_snail(self):
        """
        Function for button to move the index of the visible snail backward,
        allowing for cyleing throug the diffent snails
        """
        if self.inspector:
            self.inspector.prev_snail()
            self.update_single_snail_display()

    def next_snail(self):
        """
        Function for button to move the index of the visible snail forward,
        allowing for cyleing throug the diffent snails
        """
        if self.inspector:
            self.inspector.next_snail()
            self.update_single_snail_display()

    def goto_snail(self):
        """
        funtion for entry field of naviation plane.
        allowes user to enter specific id of snails
        """
        if self.inspector:
            val = self.snail_id_entry.get()
            try:
                idx = int(val)
            except ValueError:
                idx = val
            self.inspector.goto_snail(idx)
            self.update_single_snail_display()

    def delete_snail(self):
        """
        Deletes the currently displayed snail from the detected snails.
        """
        if self.inspector:
            snail_id = self.snail_id_entry.get()
            if snail_id.isdigit():
                snail_id = int(snail_id)
            if snail_id in self.detected_snails:
                self.deleted_snails.append(snail_id)
                del self.detected_snails[snail_id]
                # self.inspector.delete_snail(snail_id)
                self.update_single_snail_display()
                print(f"deleted_snails: {self.deleted_snails}")
            else:
                messagebox.showwarning("Warning", f"Snail ID {snail_id} not found.")

    def update_processed_image(self):
        """
        Updates the processed image display when a Checkbutton is changed.
        """
        if self.original_loaded_image is None or self.snail_measurer is None:
            return
        annotated_image = self.snail_measurer.draw_snails(
            self.original_loaded_image, self.detected_snails,
            draw_contours=self.draw_contours_var.get(),
            draw_measurements=self.draw_measurements_var.get(),
            draw_bounding_box=self.draw_bounding_box_var.get()
        )
        self.annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(self.annotated_image_rgb)
        img_pil = img_pil.resize((800, 500), resample=Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.processed_label.config(image=img_tk, text="")
        self.processed_label.image = img_tk

    def get_pos_key(self, _event=None):
        """
        Returns the position key from the input field.
        """
        # if pos_key is not empty
        if self.pos_key_entry.get():

            # strip trailing whitespace
            poskey = self.pos_key_entry.get().strip()
            # check if poskey is a valid number
            if not poskey.isdigit():
                self.pos_key_entry.config(bg="red")
                return None
            # check if poskey is 4 digits long
            elif len(poskey) > 4:
                self.pos_key_entry.config(bg="red")

                return None
            else:
                self.pos_key_entry.config(bg="white")
                self.pos_key = poskey

        return poskey


    def delay_subsample_field_key_release(self, _event=None):
        """
        Delays the execution of get_subsample to avoid excessive calls while typing.
        """

        # if self.after_id exists
        if self.after_id:
            # cancel the previous after_id
            self.root.after_cancel(self.after_id)
        # set a new after_id with the function and typing delay
        self.after_id = self.root.after(self.typing_delay, self.get_subsample)


    def get_subsample(self, _event=None):
        """
        Returns the subsample from the input field.
        """
        # if subsample is not empty
        if self.subsample_entry.get():

            # strip trailing whitespace
            subsample = self.subsample_entry.get().strip()
            # check if subsample is a valid number
            try:
                subsample_float = float(subsample)
            except ValueError:
                self.subsample_entry.config(bg="red")
                return None

            if subsample_float < 1 or subsample_float > 100:
                self.subsample_entry.config(bg="red")
                return None

            else:
                self.subsample_entry.config(bg="white")
                self.subsample = subsample_float

            return subsample_float

    def get_project(self, _event=None):
        """
        Returns the project from the input field.
        """
        # if project is not empty
        if self.project_entry.get():

            # strip trailing whitespace
            project = self.project_entry.get().strip()
            # check if project is a valid string
            if not project:
                self.project_entry.config(bg="red")
                return None
            else:
                self.project_entry.config(bg="white")
                self.project = project

            return project

    def get_species(self, _event=None):
        """
        Returns the species from the input field.
        """

        if self.species_entry.get():

            # strip trailing whitespace
            species = self.species_entry.get().strip()
            # check if species is a valid string
            if not species:
                self.species_entry.config(bg="red")
                return None
            else:
                self.species_entry.config(bg="white")
                self.species = species
            return species

    def get_analyst(self, _event=None):
        """
        Returns the analyst from the input field.
        """
        # if analyst is not empty
        if self.analyst_entry.get():

            # strip trailing whitespace
            analyst = self.analyst_entry.get().strip()
            # check if analyst is a valid string
            if not analyst:
                self.analyst_entry.config(bg="red")
                return None
            else:
                self.analyst_entry.config(bg="white")
                self.analyst = analyst

            return analyst

    def get_year(self, _event=None):
        """
        Returns the year from the input field.
        """
        # if year is not empty
        if self.year_entry.get():

            # strip trailing whitespace
            year = self.year_entry.get().strip()
            # check if year is a valid number
            if not year.isdigit() or len(year) > 4:
                self.year_entry.config(bg="red")
                return None
            else:
                self.year_entry.config(bg="white")
                self.year = year

            return year

    def get_lab_method_code(self, _event=None):
        """
        Returns the lab method code from the input field.
        """
        # if lab_method_code is not empty
        if self.lab_method_code_entry.get():

            # strip trailing whitespace
            lab_method_code = self.lab_method_code_entry.get().strip()
            # check if lab_method_code is a valid string
            if not lab_method_code:
                self.lab_method_code_entry.config(bg="red")
                return None
            else:
                self.lab_method_code_entry.config(bg="white")
                self.lab_method_code = lab_method_code

            return lab_method_code

    def write_measurements_to_csv(self):
        """
        Writes a single sample and its instances to a CSV file.

        Parameters:
        - filename (str): The name of the output CSV file.
        - sample_data (dict): Dictionary containing sample-level data.
        - instances (list of dict): List of dictionaries, each representing an instance.
        """

        self.output_path = filedialog.askdirectory(
            title="Select Directory to Save CSV")

        # Define the fieldnames for the CSV
        fieldnames = [
            "Pos_key", "Species", "Subsample", "Analyst", "Project",
            "Year", "Time of measurement", "Lab_method_code", "ID", "Length(mm)", "Width(mm)"
        ]

        filename=f"{self.pos_key}_snail_measurements.csv"

        if self.pos_key is None:
            messagebox.showerror("Error", "Pos Key is required to save measurements.")
            return
        if self.species is None:
            messagebox.showerror("Error", "Species is required to save measurements.")
            return
        if self.subsample is None:
            messagebox.showerror("Error", "Subsample is required to save measurements.")
            return
        if self.analyst is None:
            messagebox.showerror("Error", "Analyst is required to save measurements.")
            return


        # Open the CSV file for writing
        with open(os.path.join(self.output_path, filename),
                   mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Write each instance with the sample-level data

            for snail_key in self.detected_snails:
                snail = self.detected_snails[snail_key]
                row = {
                    "Pos_key": self.pos_key,
                    "Species": self.species,
                    "Subsample": self.subsample,
                    "Analyst": self.analyst,
                    "Project": self.project,
                    "Year": self.year,
                    "Time of measurement": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Lab_method_code": self.lab_method_code,
                    "ID": snail.snail_id,
                    "Length(mm)": snail.length,
                    "Width(mm)": snail.width
                }
                writer.writerow(row)

        messagebox.showinfo("info", f"output writen to {os.path.join(self.output_path, filename)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SnailGUI(root)
    root.mainloop()
