import tkinter as tk

import tkinter as tk
from tkinter import ttk

# Create main window
root = tk.Tk()
root.title("Image Editor GUI")
root.geometry("800x600")
root.configure(bg="#d0e7f9")  # Light blue background

# Left panel
left_frame = ttk.Frame(root, width=250)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

# Original Image section
original_image_frame = ttk.LabelFrame(left_frame, text="Original Image", width=250, height=150)
original_image_frame.pack(fill="x", pady=10)
tk.Label(original_image_frame, text="Image (small)").pack(pady=20)

# Tools/Filters section
tools_frame = ttk.LabelFrame(left_frame, text="Select between various tools or filters", width=250, height=200)
tools_frame.pack(fill="x", pady=10)

# Right panel
right_frame = ttk.LabelFrame(root, text="Image (Large) Area to display and edit image")
right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)
tk.Label(right_frame, text="Image (Large)").pack(expand=True)

root.mainloop()
