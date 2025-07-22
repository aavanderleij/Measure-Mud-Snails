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
# Store detected snails globally
detected_snails = {}
# Store snail object globally
snail_obj = None
current_snail_idx = tk.IntVar(value=0)

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
    global loaded_image, snail_obj, detected_snails
    if loaded_image is None:
        from tkinter import messagebox
        messagebox.showerror("Error", "No image loaded!")
        return
    
    # Create a ReferenceObject instance and calculate pixels per metric
    ref_obj = ReferenceObject(reference_length_mm=10.42)
    pixels_per_metric = ref_obj.calculate_pixels_per_metric(loaded_image.copy())

    # Run snail detection
    snail_obj = SnailMeasurer()
    snail_obj.pixels_per_metric = pixels_per_metric

    edged = snail_obj.prep_image(loaded_image)
    annotated_image = loaded_image.copy()
    detected_snails = snail_obj.get_snail_contours(edged, annotated_image, pos_key="GUI")

    # Display all snails
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(annotated_image_rgb)
    img_pil = img_pil.resize((600, 400), resample=Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_pil)
    processed_label.config(image=img_tk, text="")
    processed_label.image = img_tk


def draw_single_snail(snail):
    """
    Draws a single snail's contour and bounding box on the processed image.
    
    Args:
        snail (SnailObject): The snail object to draw.
    """
    annotated_image = loaded_image.copy()
    annotated_image = snail_obj.draw_single_snail(annotated_image, snail)
    
    # Convert to PIL Image for Tkinter
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(annotated_image_rgb)
    img_pil = img_pil.resize((600, 400), resample=Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    processed_label.config(image=img_tk, text="")
    processed_label.image = img_tk

def view_single_snail():
    global detected_snails
    if not detected_snails:
        from tkinter import messagebox
        messagebox.showerror("Error", "No snails detected! Run 'Find Snails' first.")
        return
    # Show the first snail (or change to select another)
    first_snail = list(detected_snails.values())[0]
    draw_single_snail(first_snail)

def update_snail_display():
    global detected_snails
    snail_keys = list(detected_snails.keys())
    if not snail_keys:
        from tkinter import messagebox
        messagebox.showerror("Error", "No snails detected! Run 'Find Snails' first.")
        return
    idx = current_snail_idx.get()
    if idx < 0: idx = 0
    if idx >= len(snail_keys): idx = len(snail_keys) - 1
    current_snail_idx.set(idx)
    snail = detected_snails[snail_keys[idx]]
    draw_single_snail(snail)
    snail_id_entry.delete(0, tk.END)
    snail_id_entry.insert(0, str(snail_keys[idx]))

def prev_snail():
    if detected_snails:
        current_snail_idx.set(max(0, current_snail_idx.get() - 1))
        update_snail_display()

def next_snail():
    if detected_snails:
        current_snail_idx.set(min(len(detected_snails) - 1, current_snail_idx.get() + 1))
        update_snail_display()

def goto_snail():
    if detected_snails:
        snail_keys = list(detected_snails.keys())
        val = snail_id_entry.get()
        # Try to match by index first, then by snail_id
        try:
            idx = int(val)
            if 0 <= idx < len(snail_keys):
                current_snail_idx.set(idx)
                update_snail_display()
                return
        except ValueError:
            pass
        # Try to match by snail_id
        if val in snail_keys:
            idx = snail_keys.index(val)
            current_snail_idx.set(idx)
            update_snail_display()
        else:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Snail '{val}' not found.")

# --- Snail navigation interface ---
nav_frame = ttk.Frame(right_frame)
nav_frame.pack(side="bottom",pady=10)

prev_btn = ttk.Button(nav_frame, text="<-", width=3, command=prev_snail)
prev_btn.pack(side="left")

snail_id_entry = tk.Entry(nav_frame, width=5, justify="center")
snail_id_entry.pack(side="left", padx=2)

next_btn = ttk.Button(nav_frame, text="->", width=3, command=next_snail)
next_btn.pack(side="left")

goto_btn = ttk.Button(nav_frame, text="Go", width=3, command=goto_snail)
goto_btn.pack(side="left", padx=2)

# Add buttons
select_img_btn = ttk.Button(left_frame, text="Select Image", command=show_image)
select_img_btn.pack(pady=10)

process_btn = ttk.Button(left_frame, text="Find Snails", command=process_snails)
process_btn.pack(pady=10)

view_snail_btn = ttk.Button(left_frame, text="View Single Snail", command=view_single_snail)
view_snail_btn.pack(pady=10)

root.mainloop()
