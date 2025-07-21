import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
#TODO fix import error !!!!!
from src.reference_object import ReferenceObject
from src.snail_measurer import SnailMeasurer

# Create main window
root = tk.Tk()
root.title("Snail Measurement GUI")
root.geometry("1000x700")
root.configure(bg="#d0e7f9")

# Left panel
left_frame = ttk.Frame(root, width=250)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

# Original Image section
original_image_frame = ttk.LabelFrame(left_frame, text="Image Sample", width=250, height=150)
original_image_frame.pack(fill="x", pady=10)
image_label = tk.Label(original_image_frame, text="original image")
image_label.pack(expand=True)

# Tools/Filters section
tools_frame = ttk.LabelFrame(left_frame, text="Select between various tools or filters", width=250, height=200)
tools_frame.pack(fill="x", pady=10)
#TODO add find contours button
#TODO add inspect individual snail button 
#TODO add toggle annotations (show/hide: ID, length, width)


# Right panel for processed image
right_frame = ttk.LabelFrame(root, text="Processed Image")
right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
processed_label = tk.Label(right_frame, text="Processed image")
processed_label.pack(expand=True)

# Store loaded image globally
loaded_image = None

def show_image():
    global loaded_image
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
    )
    if file_path:
        try:
            img = Image.open(file_path)
            loaded_image = cv2.imread(file_path)
            
            img = img.resize((300, 200), resample=Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk, text="")
            image_label.image = img_tk
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

def process_snails():
    global loaded_image
    if loaded_image is None:
        from tkinter import messagebox
        messagebox.showerror("Error", "No image loaded!")
        return
    
    # Create a ReferenceObject instance and calculate pixels per metric
    print(f"Loaded image shape: {loaded_image.shape}")
    print(f"dtype: {loaded_image.dtype}")
    ref_obj = ReferenceObject(reference_length_mm=10.42)
    pixels_per_metric = ref_obj.calculate_pixels_per_metric(loaded_image.copy())
    print(f"Pixels per metric: {pixels_per_metric}")

    # Run snail detection
    snail_obj = SnailMeasurer()
    snail_obj.pixels_per_metric = pixels_per_metric
    # You may want to clip the petri dish first, e.g.:
    # mask = snail_obj.clip_petri_dish(loaded_image)
    # masked_image = cv2.bitwise_and(loaded_image, loaded_image, mask=mask)
    # For now, just use the loaded image
    edged = snail_obj.prep_image(loaded_image)
    annotated_image = loaded_image.copy()
    snails = snail_obj.get_snail_contours(edged, annotated_image, pos_key="GUI")
    # Convert annotated image to display in Tkinter
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(annotated_image_rgb)
    img_pil = img_pil.resize((500, 400), resample=Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_pil)
    processed_label.config(image=img_tk, text="")
    processed_label.image = img_tk

# Add buttons
select_img_btn = ttk.Button(left_frame, text="Select Image", command=show_image)
select_img_btn.pack(pady=10)

process_btn = ttk.Button(left_frame, text="Find Snails", command=process_snails)
process_btn.pack(pady=10)

root.mainloop()
