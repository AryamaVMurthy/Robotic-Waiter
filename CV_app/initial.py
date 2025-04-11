import os
import sys
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from matplotlib.figure import Figure
import math # For distance calculation
import shutil
from sam2.sam2_image_predictor import SAM2ImagePredictor # Added HF predictor import
import platform # Import platform to check OS for scroll binding

# Set environment variable for MPS fallback for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Select device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Use optimizations for CUDA if supported
    # torch.autocast("cuda", dtype=torch.bfloat16).__enter__() # REMOVE THIS LINE
    if torch.cuda.get_device_properties(0).major >= 8: # Check for Ampere or newer for TF32
        print("Enabling TF32 optimizations for CUDA.") # Optional: Add print statement
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Support for MPS devices is preliminary. Results may vary.")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class SAM2SegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 Segmentation Tool (HF Predictor)") # Updated title
        self.root.geometry("1400x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure dataset directory - now relative to SAM root
        self.dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
        # Initialize model variables (will be loaded when an image is opened)
        # self.sam2_model = None # Removed old model variable
        self.predictor = None
        self.image = None
        self.image_path = None
        self.image_array = None
        self.image_width = 0
        self.image_height = 0
        # --- Add this flag (from app.py) ---
        self.image_set_in_predictor = False
        # ------------------------------------
        
        # Data variables
        self.points = []  # Format: [x, y]
        self.point_labels = []  # 0 for negative, 1 for positive
        self.boxes = []  # Format: [x1, y1, x2, y2]
        self.current_box = None  # Temporary box for drawing
        
        # Labels and masks
        self.labels = ["floor"]  # Default labels
        self.current_label = self.labels[0]
        self.label_map = {}  # Maps label name to integer value
        self.update_label_map()
        
        # Generated masks (per label, before selection)
        self.label_masks = {}
        self.label_mask_scores = {}
        # Selected masks (after dialog)
        self.selected_masks = {}
        
        # Calibration variables
        self.calibration_state = "idle" # "idle", "selecting_p1", "selecting_p2", "awaiting_distance"
        self.calibration_points = [None, None] # Initialize as list with None [P1, P2]
        self.calibration_distance = tk.DoubleVar(value=10.0) # Default distance
        self.cm_per_pixel = None # Calculated ratio
        self.pixels_per_cm = None # Inverse of cm_per_pixel
        
        # Grid Settings variables
        self.grid_size_cm = tk.DoubleVar(value=10.0) # Default grid size in cm
        self.grid_size_pixels = None # Calculated grid size in pixels
        self.grid_cols = None  # Number of grid columns
        self.grid_rows = None  # Number of grid rows
        
        # UI Mode
        self.input_mode = "point_positive"
        
        # Fixed point ID counter - starts at 100 to avoid conflicts with label values
        self.fixed_point_id_counter = 100
        
        # Create UI
        self.create_ui()
        
        # Initialize with a welcome message
        self.update_status("Welcome! Please open an image to start.")
    
    def update_label_map(self):
        """Update the label map based on current labels"""
        # Preserve existing data if possible (or handle migration if structure changes)
        points_data = self.label_map.get("points", {})
        fixed_points_data = self.label_map.get("fixed_points", {})
        fixed_point_ids_data = self.label_map.get("fixed_point_ids", {})
        negative_points = self.label_map.get("negative_points", [])
        calibration_data = self.label_map.get("_calibration_", None)
        grid_settings_data = self.label_map.get("_grid_settings_", None)

        # Reset the main label map part
        self.label_map = {}
        for i, label in enumerate(self.labels, start=1): # Start labels from 1 (0 is background)
            self.label_map[label] = i

        # Restore/initialize structured data
        self.label_map["points"] = points_data
        self.label_map["fixed_points"] = fixed_points_data
        self.label_map["fixed_point_ids"] = fixed_point_ids_data
        self.label_map["negative_points"] = negative_points

        # Ensure all current labels have entries
        for label in self.labels:
            if label not in self.label_map["points"]: self.label_map["points"][label] = []
            if label not in self.label_map["fixed_points"]: self.label_map["fixed_points"][label] = []
            if label not in self.label_map["fixed_point_ids"]: self.label_map["fixed_point_ids"][label] = []

        # Restore calibration/grid if they existed
        if calibration_data: self.label_map["_calibration_"] = calibration_data
        if grid_settings_data: self.label_map["_grid_settings_"] = grid_settings_data
    
    def create_ui(self):
        """Create the UI components"""
        # Main frame layout
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into left (canvas) and right (controls) panels
        self.left_panel = ttk.Frame(self.main_frame)
        # --- Right Panel Setup for Scrolling ---
        self.right_panel_container = ttk.Frame(self.main_frame, width=350) # Container for canvas+scrollbar
        self.right_panel_container.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.right_panel_container.pack_propagate(False) # Prevent container resizing

        # Create Canvas and Scrollbar
        self.right_canvas = tk.Canvas(self.right_panel_container, borderwidth=0, highlightthickness=0)
        self.right_scrollbar = ttk.Scrollbar(self.right_panel_container, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        # Create the frame *inside* the canvas that will hold the widgets
        self.right_panel = ttk.Frame(self.right_canvas) # This is now the scrollable frame

        # Place the scrollable frame onto the canvas
        self.canvas_frame_id = self.right_canvas.create_window((0, 0), window=self.right_panel, anchor="nw") # Store the window ID

        # Pack canvas and scrollbar into the container
        self.right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Update scrollregion when the size of the scrollable frame changes
        self.right_panel.bind("<Configure>", self._on_frame_configure)
        # Also update when the canvas size changes (e.g., window resize affecting container)
        self.right_canvas.bind("<Configure>", self._on_canvas_configure)

        # Bind mouse wheel scrolling
        self._bind_mouse_wheel(self.right_canvas) # Bind to canvas is usually sufficient
        # self._bind_mouse_wheel(self.right_panel) # Binding to inner frame might not be needed now

        # -----------------------------------------

        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas for image and drawing (Left Panel)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Image Canvas")
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Connect mouse events for the image canvas
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # --- Right Panel Controls (Packed into self.right_panel - the scrollable frame) ---
        # Ensure widgets use fill=tk.X where appropriate to use available width,
        # but avoid settings that force horizontal expansion beyond the container width.
        # The Treeview columns seem okay (70+80+100 = 250 < 350).

        ttk.Label(self.right_panel, text="Controls", font=("Arial", 14, "bold")).pack(pady=10, anchor='n', fill=tk.X) # Fill X

        # File operations frame
        file_frame = ttk.LabelFrame(self.right_panel, text="File Operations")
        file_frame.pack(fill=tk.X, padx=10, pady=5, anchor='n')
        ttk.Button(file_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, padx=5, pady=5)

        # Input mode frame
        mode_frame = ttk.LabelFrame(self.right_panel, text="Input Mode")
        mode_frame.pack(fill=tk.X, padx=10, pady=5, anchor='n')

        self.mode_var = tk.StringVar(value="point_positive")
        ttk.Radiobutton(mode_frame, text="Positive Points (+)", variable=self.mode_var,
                        value="point_positive", command=self.set_mode).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Negative Points (-)", variable=self.mode_var,
                        value="point_negative", command=self.set_mode).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Box", variable=self.mode_var,
                        value="box", command=self.set_mode).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Fixed Points", variable=self.mode_var,
                        value="fixed_point", command=self.set_mode).pack(anchor=tk.W, padx=5, pady=2)

        # Label selection frame
        label_frame = ttk.LabelFrame(self.right_panel, text="Label Selection")
        label_frame.pack(fill=tk.X, padx=10, pady=5, anchor='n')

        self.label_var = tk.StringVar(value=self.current_label)
        self.label_combobox = ttk.Combobox(label_frame, textvariable=self.label_var,
                                           values=self.labels, state="readonly")
        self.label_combobox.pack(fill=tk.X, padx=5, pady=5)
        self.label_combobox.bind("<<ComboboxSelected>>", self.on_label_selected)

        new_label_frame = ttk.Frame(label_frame)
        new_label_frame.pack(fill=tk.X, padx=5, pady=5)

        self.new_label_entry = ttk.Entry(new_label_frame)
        self.new_label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(new_label_frame, text="Add Label", command=self.add_new_label).pack(side=tk.RIGHT)

        # Calibration frame
        calib_frame = ttk.LabelFrame(self.right_panel, text="Calibration")
        calib_frame.pack(fill=tk.X, padx=10, pady=5, anchor='n')

        calib_select_frame = ttk.Frame(calib_frame)
        calib_select_frame.pack(fill=tk.X, padx=5, pady=2)
        self.p1_select_active = tk.BooleanVar(value=False)
        self.p2_select_active = tk.BooleanVar(value=False)
        self.p1_button = ttk.Checkbutton(calib_select_frame, text="Select P1", variable=self.p1_select_active,
                           command=self.toggle_select_p1, style='Toolbutton')
        self.p1_button.pack(side=tk.LEFT, padx=2)
        self.p2_button = ttk.Checkbutton(calib_select_frame, text="Select P2", variable=self.p2_select_active,
                           command=self.toggle_select_p2, style='Toolbutton')
        self.p2_button.pack(side=tk.LEFT, padx=2)

        calib_dist_frame = ttk.Frame(calib_frame)
        calib_dist_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(calib_dist_frame, text="Distance (cm):").pack(side=tk.LEFT, padx=(0, 5))
        self.calib_distance_entry = ttk.Entry(calib_dist_frame, textvariable=self.calibration_distance, width=10)
        self.calib_distance_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.set_calib_button = ttk.Button(calib_dist_frame, text="Set", command=self.set_calibration, state=tk.DISABLED)
        self.set_calib_button.pack(side=tk.RIGHT)

        self.calib_status_label = ttk.Label(calib_frame, text="Status: Select P1 & P2")
        self.calib_status_label.pack(fill=tk.X, padx=5, pady=5)
        self.calib_ratio_label = ttk.Label(calib_frame, text="Ratio (px/cm): N/A")
        self.calib_ratio_label.pack(fill=tk.X, padx=5, pady=5)

        # Grid Settings frame
        grid_frame = ttk.LabelFrame(self.right_panel, text="Grid Settings")
        grid_frame.pack(fill=tk.X, padx=10, pady=5, anchor='n')

        grid_size_frame = ttk.Frame(grid_frame)
        grid_size_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(grid_size_frame, text="Grid Cell Size (cm):").pack(side=tk.LEFT, padx=(0, 5))
        self.grid_size_entry = ttk.Entry(grid_size_frame, textvariable=self.grid_size_cm, width=10)
        self.grid_size_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.apply_grid_button = ttk.Button(grid_size_frame, text="Apply", command=self.apply_grid_size, state=tk.DISABLED)
        self.apply_grid_button.pack(side=tk.RIGHT)

        self.grid_pixel_label = ttk.Label(grid_frame, text="Pixel Size (px): N/A")
        self.grid_pixel_label.pack(fill=tk.X, padx=5, pady=5)

        # Point List frame
        point_list_frame = ttk.LabelFrame(self.right_panel, text="Active Points")
        point_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Treeview for points
        self.point_tree = ttk.Treeview(point_list_frame, columns=("Type", "Label/ID", "Coords"), show="headings", selectmode="extended")
        self.point_tree.heading("Type", text="Type")
        self.point_tree.heading("Label/ID", text="Label/ID")
        self.point_tree.heading("Coords", text="Coords")
        self.point_tree.column("Type", width=70, anchor='w')
        self.point_tree.column("Label/ID", width=80, anchor='w')
        self.point_tree.column("Coords", width=100, anchor='w')

        tree_scrollbar = ttk.Scrollbar(point_list_frame, orient="vertical", command=self.point_tree.yview)
        self.point_tree.configure(yscrollcommand=tree_scrollbar.set)

        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.point_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Delete button for points
        delete_button_frame = ttk.Frame(self.right_panel)
        delete_button_frame.pack(fill=tk.X, padx=10, pady=5, anchor='s')
        ttk.Button(delete_button_frame, text="Delete Selected Points", command=self.delete_selected_points).pack(fill=tk.X, padx=5, pady=5)

        # Actions frame (Generate/Save/Clear)
        actions_frame = ttk.LabelFrame(self.right_panel, text="Actions")
        actions_frame.pack(fill=tk.X, padx=10, pady=5, anchor='s')
        ttk.Button(actions_frame, text="Generate Masks", command=self.generate_masks).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(actions_frame, text="Generate & Save Grid", command=self.save_grid).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(actions_frame, text="Clear All", command=self.clear_all).pack(fill=tk.X, padx=5, pady=5)

        # Status frame
        status_frame = ttk.LabelFrame(self.right_panel, text="Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5, anchor='s')
        self.status_text = tk.Text(status_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.X, padx=5, pady=5)

        # Keyboard shortcuts
        self.root.bind('<Key>', self.on_key_press)

        # Initial UI update
        self.update_ui()

    # --- Modified Helper Methods for Scrolling ---
    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame's height."""
        # Update scrollregion based on inner frame height and canvas width
        canvas_width = self.right_canvas.winfo_width()
        bbox = self.right_canvas.bbox("all") # Get required height from bbox
        if bbox:
            scroll_height = bbox[3] # Use the height from the bounding box
            self.right_canvas.configure(scrollregion=(0, 0, canvas_width, scroll_height))
        else:
            self.right_canvas.configure(scrollregion=(0, 0, canvas_width, 0))


    def _on_canvas_configure(self, event=None):
        """Update the inner frame's width and the scrollregion when canvas resizes."""
        canvas_width = event.width
        # Set the inner frame's width to match the canvas width
        self.right_canvas.itemconfig(self.canvas_frame_id, width=canvas_width)
        # Update the scroll region width
        self._on_frame_configure() # Recalculate scroll region with new width


    def _on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling (Vertical Only)"""
        # Check if shift key is pressed (common for horizontal scroll) - skip if so
        # Note: Tkinter event state checking can be complex/platform-specific.
        # This basic check might not work everywhere.
        # if event.state & 0x1: # Check for Shift key state
        #     return

        # Determine scroll direction and amount (platform-dependent)
        if platform.system() == "Linux":
            if event.num == 4:
                self.right_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.right_canvas.yview_scroll(1, "units")
        elif platform.system() == "Windows":
            self.right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif platform.system() == "Darwin": # macOS uses delta differently
            self.right_canvas.yview_scroll(int(-1 * event.delta), "units")
        else: # Default fallback (might need adjustment)
             if hasattr(event, 'delta') and event.delta:
                 scroll_dir = -1 if event.delta > 0 else 1
                 self.right_canvas.yview_scroll(scroll_dir, "units")


    def _bind_mouse_wheel(self, widget):
        """Bind mouse wheel events to a widget for scrolling the right_canvas."""
        # Bind directly to the canvas, as events bubble up
        if platform.system() == "Linux":
            widget.bind("<Button-4>", self._on_mouse_wheel, add='+')
            widget.bind("<Button-5>", self._on_mouse_wheel, add='+')
        else: # Windows and macOS
            widget.bind("<MouseWheel>", self._on_mouse_wheel, add='+')

        # No longer need recursive binding as we bind only to the canvas
        # if isinstance(widget, (tk.Frame, ttk.Frame, ttk.LabelFrame)):
        #     for child in widget.winfo_children():
        #         self._bind_mouse_wheel(child)
    # --------------------------------------

    def update_ui(self):
        """Update the UI state based on current data, including the point list"""
        # Update combobox values
        self.label_combobox['values'] = self.labels
        
        # --- Populate Point List Treeview ---
        self.point_tree.delete(*self.point_tree.get_children()) # Clear existing items
        
        # Add Positive/Negative points (from self.points)
        for i, (point, label_flag) in enumerate(zip(self.points, self.point_labels)):
            ptype = "Positive" if label_flag == 1 else "Negative"
            # Find which label the positive point belongs to (if any)
            point_label_name = ""
            if label_flag == 1:
                 for lbl, pts in self.label_map.get("points", {}).items():
                     # Use tolerance comparison for potentially float coords? No, they should be ints.
                     if point in pts:
                         point_label_name = lbl
                         break
            label_id_text = point_label_name if label_flag == 1 else "-" # Show label for positive, '-' for negative
            coords_text = f"({point[0]}, {point[1]})"
            # Store data needed for deletion in item ID (iid)
            item_id = f"{'pos' if label_flag == 1 else 'neg'}_{i}"
            self.point_tree.insert("", tk.END, iid=item_id, values=(ptype, label_id_text, coords_text))
        
        # Add Fixed points (from self.label_map["fixed_points"])
        fixed_point_counter = 0
        for label, points in self.label_map.get("fixed_points", {}).items():
             point_ids = self.label_map.get("fixed_point_ids", {}).get(label, [])
             for i, point in enumerate(points):
                 point_id = point_ids[i] if i < len(point_ids) else "?"
                 coords_text = f"({point[0]}, {point[1]})"
                 item_id = f"fixed_{label}_{i}" # Need label and index within label's list
                 self.point_tree.insert("", tk.END, iid=item_id, values=("Fixed", f"{label} ({point_id})", coords_text))
                 fixed_point_counter += 1
        
        # Add Calibration points
        if self.calibration_points[0] is not None:
             point = self.calibration_points[0]
             coords_text = f"({point[0]}, {point[1]})"
             item_id = "calib_0"
             self.point_tree.insert("", tk.END, iid=item_id, values=("Calib", "P1", coords_text))
        if self.calibration_points[1] is not None:
             point = self.calibration_points[1]
             coords_text = f"({point[0]}, {point[1]})"
             item_id = "calib_1"
             self.point_tree.insert("", tk.END, iid=item_id, values=("Calib", "P2", coords_text))
        
        # Update calibration button states visually (redundant if checkbutton does it, but safe)
        self.p1_select_active.set(self.calibration_state == "selecting_p1")
        self.p2_select_active.set(self.calibration_state == "selecting_p2")
        
        # Update calibration 'Set' button state
        if self.calibration_points[0] is not None and self.calibration_points[1] is not None:
            self.set_calib_button.config(state=tk.NORMAL)
        else:
            self.set_calib_button.config(state=tk.DISABLED)
        
        # Update grid 'Apply' button state
        if self.pixels_per_cm is not None: # Check pixels_per_cm instead of cm_per_pixel
             self.apply_grid_button.config(state=tk.NORMAL)
        else:
             self.apply_grid_button.config(state=tk.DISABLED)
        
        # Redraw canvas
        self.redraw_canvas()
    
    def redraw_canvas(self):
        """Redraw the matplotlib canvas with current image and overlays"""
        self.ax.clear()
        
        # Draw image if available
        if self.image_array is not None:
            self.ax.imshow(self.image_array)
            
            # Draw selected masks if available
            if hasattr(self, 'selected_masks') and self.selected_masks:
                for label, mask in self.selected_masks.items():
                    if isinstance(mask, np.ndarray):
                        self.show_mask(mask, random_color=True)
                    else:
                        print(f"Warning: Invalid mask for label {label}, type: {type(mask)}")
            
            # Draw points (Positive/Negative)
            if self.points:
                points_array = np.array(self.points)
                labels_array = np.array(self.point_labels)
                self.show_points(points_array, labels_array)
            
            # Draw fixed points
            if "fixed_points" in self.label_map:
                self.show_fixed_points()
            
            # Draw boxes
            for i, box in enumerate(self.boxes):
                self.show_box(box, label=f"Box {i+1}")
            
            # Draw current box being created (if any)
            if self.current_box:
                self.show_box(self.current_box, color='blue')
            
            # Show calibration points and line if they exist
            self.show_calibration_points() # Handles None check inside
            
            # Turn off axis for cleaner display
            self.ax.set_axis_off()
            
            # Update title based on input mode (excluding calibration states)
            mode_display_names = {
                "point_positive": "Positive Points (+)",
                "point_negative": "Negative Points (-)",
                "box": "Box",
                "fixed_point": "Fixed Points"
            }
            active_mode = self.input_mode # The radio button mode
            title_text = f"Mode: {mode_display_names.get(active_mode, active_mode.title())}"
            
            # Add calibration state to title if active
            if self.calibration_state == "selecting_p1":
                 title_text += " (Selecting P1)"
            elif self.calibration_state == "selecting_p2":
                 title_text += " (Selecting P2)"
            
            title_text += f", Label: {self.current_label}"
            self.ax.set_title(title_text)
        
        self.canvas.draw()
    
    def show_mask(self, mask, random_color=False, alpha=0.6):
        """Show a mask on the current axes"""
        if mask is None or not isinstance(mask, np.ndarray):
            print(f"Error: Invalid mask type: {type(mask)}")
            return

        if random_color:
            # Generate visually distinct colors if possible
            color_val = np.random.rand(3)
            # Ensure reasonable brightness/saturation? For now, just random.
            color = np.concatenate([color_val, np.array([alpha])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, alpha]) # Default blueish

        try:
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8) # Ensure mask is uint8 for drawing
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            # Draw borders if possible
            try:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Use SIMPLE approx
                # Filter small contours? Optional.
                # Draw contours with slight transparency
                mask_image = cv2.drawContours(mask_image, contours, -1, (1.0, 1.0, 1.0, 0.7), thickness=2) # White-ish border
            except Exception as e:
                print(f"Warning: Could not draw contours: {e}")

            self.ax.imshow(mask_image)
        except Exception as e:
            print(f"Error displaying mask: {e}, mask type: {type(mask)}, mask shape: {getattr(mask, 'shape', 'unknown')}")
    
    def show_points(self, coords, labels):
        """Show points on the current axes"""
        if len(coords) == 0:
            return

        pos_points = coords[labels==1]
        neg_points = coords[labels==0]

        marker_size = 100

        if len(pos_points) > 0:
            self.ax.scatter(pos_points[:, 0], pos_points[:, 1], color='lime', marker='*', # Brighter green
                           s=marker_size, edgecolor='black', linewidth=1.0) # Black edge for visibility
        if len(neg_points) > 0:
            self.ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='X', # Use X for negative
                           s=marker_size, edgecolor='black', linewidth=1.0)
    
    def show_box(self, box, label=None, color='cyan'): # Changed default box color
        """Show a box on the current axes"""
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0

        rect = patches.Rectangle((x0, y0), width, height, linewidth=2,
                                 edgecolor=color, facecolor='none')
        self.ax.add_patch(rect)

        if label:
            self.ax.text(x0 + width/2, y0 - 5, label, color=color,
                        ha='center', va='bottom', fontsize=9, fontweight='bold', # Slightly smaller
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
    
    def show_fixed_points(self):
        """Show fixed points on the canvas"""
        if "fixed_points" not in self.label_map:
            return

        cmap = plt.cm.get_cmap('tab10', max(10, len(self.labels)))

        for i, label in enumerate(self.labels):
            if label in self.label_map.get("fixed_points", {}) and self.label_map["fixed_points"][label]:
                points = self.label_map["fixed_points"][label]
                point_ids = self.label_map.get("fixed_point_ids", {}).get(label, [])
                if not points: continue

                points_array = np.array(points)
                color = cmap(i % cmap.N)

                self.ax.scatter(points_array[:, 0], points_array[:, 1],
                               color=color, marker='o', s=60, edgecolor='black', linewidth=1.0) # Smaller, edged

                for j, point in enumerate(points):
                    point_id = point_ids[j] if j < len(point_ids) else "?"
                    self.ax.text(point[0], point[1] - 8, str(point_id), # Adjusted position
                                color=color, ha='center', va='bottom', fontsize=8, fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1'))
    
    def show_calibration_points(self):
        """Show calibration points P1 and P2 and line on the canvas"""
        points_to_draw = [p for p in self.calibration_points if p is not None]
        if not points_to_draw:
            return

        points = np.array(points_to_draw)
        self.ax.scatter(points[:, 0], points[:, 1], color='magenta', marker='D', # Diamond marker
                       s=120, edgecolor='black', linewidth=1.0)

        for i, p in enumerate(self.calibration_points):
            if p is not None:
                self.ax.text(p[0] + 6, p[1] + 6, f"P{i+1}", color='magenta', fontsize=11, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        if self.calibration_points[0] is not None and self.calibration_points[1] is not None:
            points_arr = np.array(self.calibration_points)
            self.ax.plot(points_arr[:, 0], points_arr[:, 1], color='magenta', linestyle=':', linewidth=2.5) # Dotted line
    
    def update_status(self, message):
        """Update the status text with a new message"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete('1.0', tk.END)
        self.status_text.insert(tk.END, message)
        self.status_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def open_image(self):
        """Open an image file via dialog"""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not file_path: return

        try:
            # --- Clear previous data FIRST ---
            self.clear_all()
            # ---------------------------------

            # Load the new image
            self.image = Image.open(file_path).convert("RGB") # Load as PIL Image
            self.image_path = file_path
            self.image_array = np.array(self.image) # Keep numpy array for display/calcs
            self.image_height, self.image_width = self.image_array.shape[:2]

            # Reset the flag before trying to set
            self.image_set_in_predictor = False

            # Initialize the model/predictor if not already done
            if self.predictor is None:
                self.initialize_model()

            # Set the image in the predictor if initialization was successful
            if self.predictor:
                self.set_image_in_predictor() # Use the PIL image

            # Update UI
            self.update_status(f"Loaded image: {os.path.basename(file_path)}")
            self.update_ui() # Update everything including canvas

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.update_status(f"Error loading image: {e}")
            # Ensure flag is false if image loading fails
            self.image_set_in_predictor = False
    
    def initialize_model(self):
        """Initialize the SAM2 model predictor using Hugging Face"""
        if self.predictor is not None: return # Already initialized

        try:
            self.update_status("Initializing SAM2 predictor from Hugging Face... This may take a moment.")

            # Choose the Hugging Face model name
            # Options: "facebook/sam2-hiera-tiny", "facebook/sam2-hiera-small",
            #          "facebook/sam2-hiera-base-plus", "facebook/sam2-hiera-large"
            hf_model_name = "facebook/sam2-hiera-tiny"

            self.update_status(f"Loading Predictor {hf_model_name}...")
            # Load the predictor directly
            self.predictor = SAM2ImagePredictor.from_pretrained(hf_model_name)

            # --- Move the predictor's *model* to the selected device ---
            if hasattr(self.predictor, 'model') and isinstance(self.predictor.model, torch.nn.Module):
                self.predictor.model.to(device)
                self.update_status(f"Predictor model moved to device: {device}")
            else:
                # Fallback or warning if the model attribute isn't found as expected
                self.update_status(f"Warning: Could not automatically move predictor model to device {device}. Assuming predictor handles device internally.")
                print(f"Warning: Predictor object of type {type(self.predictor)} does not have an accessible 'model' attribute for device placement.")
            # ---------------------------------------------------------


            self.update_status(f"SAM2 predictor ({hf_model_name}) initialized successfully!")

            # If an image is already loaded, set it in the predictor
            if self.image:
                self.set_image_in_predictor()

        except ImportError as e:
             messagebox.showerror("Import Error", f"Failed to import SAM2ImagePredictor: {e}\nMake sure the 'sam2' library (from Meta) is installed correctly.")
             self.update_status(f"Import Error: {e}")
             self.predictor = None # Ensure predictor is None on error
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to initialize SAM2 predictor from Hugging Face ({hf_model_name}): {str(e)}")
            self.update_status(f"Model Init Error: {e}")
            self.predictor = None # Ensure predictor is None on error
            import traceback
            traceback.print_exc()

    def set_image_in_predictor(self):
        """Sets the current PIL image in the SAM2 predictor."""
        if self.predictor and self.image:
            try:
                self.update_status("Setting image in predictor...")
                # Use the PIL Image object
                self.predictor.set_image(self.image)
                # --- Set flag to True ONLY after success ---
                self.image_set_in_predictor = True
                # -------------------------------------------
                self.update_status("Image set in predictor.")
            except Exception as e:
                # --- Reset flag on error ---
                self.image_set_in_predictor = False
                # ---------------------------
                messagebox.showerror("Predictor Error", f"Failed to set image in predictor: {e}")
                self.update_status(f"Error setting image in predictor: {e}")
                print("Detailed error during set_image:")
                import traceback
                traceback.print_exc()
        else:
             # --- Ensure flag is False if predictor/image missing ---
             self.image_set_in_predictor = False
             if not self.predictor: self.update_status("Cannot set image: Predictor not initialized.")
             if not self.image: self.update_status("Cannot set image: No image loaded.")
             # -------------------------------------------------------

    def set_mode(self):
        """Set the current input mode from radio buttons"""
        # If changing mode, ensure calibration selection is deactivated
        if self.calibration_state.startswith("selecting"):
            self.calibration_state = "idle"
            self.update_status("Calibration point selection cancelled.")
            self.check_calibration_readiness() # Update labels/buttons

        self.input_mode = self.mode_var.get()
        self.update_status(f"Mode changed to {self.input_mode}")
        self.update_ui() # Redraws canvas with updated title
    
    def on_label_selected(self, event):
        """Handle label selection from combobox"""
        self.current_label = self.label_var.get()
        self.update_status(f"Selected label: {self.current_label}")
        self.update_ui()
    
    def add_new_label(self):
        """Add a new label to the list"""
        new_label = self.new_label_entry.get().strip().lower() # Standardize to lower case

        if not new_label:
            self.update_status("Label name cannot be empty.")
            return
        if not new_label.isidentifier(): # Basic check for valid name
             self.update_status(f"Invalid label name: '{new_label}'. Use letters, numbers, underscores.")
             return
        if new_label in self.labels or new_label.startswith("_"): # Avoid conflicts with internal keys
            self.update_status(f"Label '{new_label}' already exists or is invalid.")
            return

        self.labels.append(new_label)
        self.update_label_map() # This re-initializes label map entries

        self.current_label = new_label
        self.label_var.set(new_label)
        self.new_label_entry.delete(0, tk.END)
        self.update_status(f"Added new label: {new_label}")
        self.update_ui()
    
    def on_canvas_click(self, event):
        """Handle mouse click on canvas"""
        if event.inaxes != self.ax or self.image_array is None: return
        if event.xdata is None or event.ydata is None: return # Click outside axes

        x, y = int(round(event.xdata)), int(round(event.ydata)) # Round to nearest int

        # --- Check if predictor is ready ---
        if self.predictor is None:
            self.update_status("Model not initialized. Please open an image.")
            return
        if not self.image_set_in_predictor:
            self.update_status("Image not set in predictor. Please re-open the image or check for errors.")
            # Optionally try setting it again?
            # self.set_image_in_predictor()
            # if not self.image_set_in_predictor: return
            return
        # -----------------------------------

        # --- Calibration Point Selection ---
        if self.calibration_state == "selecting_p1":
            # Check if it's the same as P2
            if self.calibration_points[1] is not None and [x,y] == self.calibration_points[1]:
                 self.update_status("Cannot select the same point for P1 and P2.")
                 return

            self.calibration_points[0] = [x, y]
            self.calibration_state = "idle" # Deactivate selection mode
            self.update_status(f"Selected P1 at ({x}, {y}).")
            self.check_calibration_readiness() # Update status labels and buttons
            self.update_ui() # Redraw canvas and update point list
            return # Handled

        elif self.calibration_state == "selecting_p2":
            # Check if it's the same as P1
            if self.calibration_points[0] is not None and [x,y] == self.calibration_points[0]:
                self.update_status("Cannot select the same point for P1 and P2.")
                return

            self.calibration_points[1] = [x, y]
            self.calibration_state = "idle" # Deactivate selection mode
            self.update_status(f"Selected P2 at ({x}, {y}).")
            self.check_calibration_readiness() # Update status labels and buttons
            self.update_ui() # Redraw canvas and update point list
            return # Handled

        # --- Regular Input Mode Handling (if not calibrating) ---
        active_mode = self.input_mode # Get mode from radio button state

        if active_mode == "point_positive":
            # Add to global list (for mask generation prompts and display)
            self.points.append([x, y])
            self.point_labels.append(1)
            # Add to the specific label's list in the map
            if self.current_label not in self.label_map.get("points", {}):
                self.label_map.setdefault("points", {})[self.current_label] = []
            self.label_map["points"][self.current_label].append([x, y])
            self.update_status(f"Added positive point at ({x}, {y}) for label '{self.current_label}'")

        elif active_mode == "point_negative":
            # Add to global list (for mask generation prompts and display)
            self.points.append([x, y])
            self.point_labels.append(0)
            # Add to the global negative list in the map
            self.label_map.setdefault("negative_points", []).append([x, y])
            self.update_status(f"Added negative point at ({x}, {y})")

        elif active_mode == "box":
            # Start drawing a box (on button press)
            self.current_box = [x, y, x, y]
            self.update_status(f"Started drawing box at ({x}, {y})")

        elif active_mode == "fixed_point":
            # Add to the specific label's list in the map
            label_fixed_points = self.label_map.setdefault("fixed_points", {}).setdefault(self.current_label, [])
            label_fixed_ids = self.label_map.setdefault("fixed_point_ids", {}).setdefault(self.current_label, [])

            # Assign a unique ID
            self.fixed_point_id_counter += 1
            current_id = self.fixed_point_id_counter

            label_fixed_points.append([x, y])
            label_fixed_ids.append(current_id)

            self.update_status(f"Added fixed point ({x}, {y}) for label '{self.current_label}' with ID {current_id}")

        # Update UI after any point/box action
        self.update_ui()
    
    def on_mouse_move(self, event):
        """Handle mouse movement on canvas, primarily for drawing boxes"""
        if self.input_mode == "box" and self.current_box and event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.current_box[2] = x
            self.current_box[3] = y
            self.redraw_canvas() # Redraw frequently while dragging box
    
    def on_canvas_release(self, event):
        """Handle mouse release on canvas, primarily for finishing boxes"""
        if self.input_mode == "box" and self.current_box and event.inaxes == self.ax:
            if event.xdata is None or event.ydata is None: # Released outside axes
                 self.current_box = None
                 self.update_status("Box drawing cancelled (released outside).")
                 self.redraw_canvas()
                 return

            x, y = int(round(event.xdata)), int(round(event.ydata))
            self.current_box[2] = x
            self.current_box[3] = y

            # Ensure box coordinates are ordered (x0, y0) top-left, (x1, y1) bottom-right
            x0, x1 = min(self.current_box[0], self.current_box[2]), max(self.current_box[0], self.current_box[2])
            y0, y1 = min(self.current_box[1], self.current_box[3]), max(self.current_box[1], self.current_box[3])

            # Minimum size check
            if x1 - x0 < 5 or y1 - y0 < 5:
                self.update_status("Box too small - please draw a larger box.")
            else:
                self.boxes.append([x0, y0, x1, y1])
                self.update_status(f"Added box at ({x0}, {y0}, {x1}, {y1})")

            # Reset current box and update UI
            self.current_box = None
            self.update_ui()
    
    def calculate_pixel_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        # Ensure points are valid lists/tuples/arrays of numbers
        if p1 is None or p2 is None: return float('inf')
        try:
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        except (TypeError, IndexError):
            return float('inf') # Invalid point format
    
    def toggle_select_p1(self):
        """Toggle the state for selecting calibration point P1."""
        if self.image_array is None:
             self.update_status("Open image first.")
             self.p1_select_active.set(False) # Ensure checkbox reflects state
             return

        if self.p1_select_active.get(): # If checkbutton is now checked (means user clicked it on)
            # Deactivate P2 if it was active
            if self.calibration_state == "selecting_p2":
                self.p2_select_active.set(False)

            self.calibration_state = "selecting_p1"
            self.update_status("Click on the image to set calibration point P1.")
            self.calib_status_label.config(text="Status: Select P1")
        else: # If checkbutton is now unchecked (user clicked it off)
             if self.calibration_state == "selecting_p1": # Only change state if it was selecting P1
                 self.calibration_state = "idle"
                 self.update_status("Cancelled P1 selection.")
                 self.check_calibration_readiness() # Update overall status

        # Ensure P2 button reflects its state if P1 was just activated
        if self.calibration_state == "selecting_p1":
             self.p2_select_active.set(False)

        self.update_ui() # Redraw canvas title
    
    def toggle_select_p2(self):
        """Toggle the state for selecting calibration point P2."""
        if self.image_array is None:
             self.update_status("Open image first.")
             self.p2_select_active.set(False) # Ensure checkbox reflects state
             return

        if self.p2_select_active.get(): # If checkbutton is now checked
            # Deactivate P1 if it was active
            if self.calibration_state == "selecting_p1":
                 self.p1_select_active.set(False)

            self.calibration_state = "selecting_p2"
            self.update_status("Click on the image to set calibration point P2.")
            self.calib_status_label.config(text="Status: Select P2")
        else: # If checkbutton is now unchecked
            if self.calibration_state == "selecting_p2": # Only change state if it was selecting P2
                 self.calibration_state = "idle"
                 self.update_status("Cancelled P2 selection.")
                 self.check_calibration_readiness() # Update overall status

        # Ensure P1 button reflects its state if P2 was just activated
        if self.calibration_state == "selecting_p2":
             self.p1_select_active.set(False)

        self.update_ui() # Redraw canvas title
    
    def check_calibration_readiness(self):
        """Check if P1 & P2 are set, update status labels and 'Set' button state."""
        p1_set = self.calibration_points[0] is not None
        p2_set = self.calibration_points[1] is not None

        status_parts = []
        if p1_set: status_parts.append("P1 Set")
        if p2_set: status_parts.append("P2 Set")

        if p1_set and p2_set:
            self.set_calib_button.config(state=tk.NORMAL)
            # Don't change calibration_state here, only on click or toggle
            if self.calibration_state != "selecting_p1" and self.calibration_state != "selecting_p2":
                 self.calib_status_label.config(text="Status: Enter distance & Set")
                 # Only focus if calibration just completed (e.g., after P2 click)
                 # self.calib_distance_entry.focus()
        else:
            self.set_calib_button.config(state=tk.DISABLED)
            if self.calibration_state != "selecting_p1" and self.calibration_state != "selecting_p2":
                if not status_parts:
                    self.calib_status_label.config(text="Status: Select P1 & P2")
                elif p1_set:
                    self.calib_status_label.config(text="Status: P1 Set. Select P2.")
                else: # p2_set must be true
                    self.calib_status_label.config(text="Status: P2 Set. Select P1.")

        # Update ratio label if calibration is lost
        if self.pixels_per_cm is None: # Check pixels_per_cm
             self.calib_ratio_label.config(text="Ratio (px/cm): N/A") # Updated label
             self.apply_grid_button.config(state=tk.DISABLED)
             self.grid_pixel_label.config(text="Pixel Size (px): N/A")

        # Ensure toggle buttons match the state (might be redundant but safe)
        self.p1_select_active.set(self.calibration_state == "selecting_p1")
        self.p2_select_active.set(self.calibration_state == "selecting_p2")
    
    def set_calibration(self):
        """Finalize calibration: calculate ratio from P1, P2 and distance input."""
        if not (self.calibration_points[0] is not None and self.calibration_points[1] is not None):
            self.update_status("Calibration Error: Both P1 and P2 must be selected first.")
            messagebox.showwarning("Calibration Error", "Please select both P1 and P2 using the buttons and clicking the image.", parent=self.root)
            return

        try:
            # Get the distance in centimeters
            distance_cm = float(self.calibration_distance.get())
            if distance_cm <= 0:
                self.update_status("Calibration Error: Distance must be positive.")
                messagebox.showerror("Error", "Distance must be a positive number.", parent=self.root)
                return

            # Get the two calibration points
            p1, p2 = self.calibration_points
            pixel_dist = self.calculate_pixel_distance(p1, p2)

            if pixel_dist < 1e-6: # Check for coincident points
                self.update_status("Calibration Error: Points P1 and P2 are coincident or too close.")
                messagebox.showerror("Error", "Calibration points P1 and P2 are coincident or too close.", parent=self.root)
                return

            # Calculate and store ratio (pixels per cm)
            self.pixels_per_cm = pixel_dist / distance_cm
            self.cm_per_pixel = 1.0 / self.pixels_per_cm # Keep cm_per_pixel as well if needed elsewhere
            self.calibration_state = "idle" # Mark as calibrated, no longer selecting

            # Update UI with calibration results
            status_msg = f"Calibration successful! Ratio: {self.pixels_per_cm:.2f} pixels/cm"
            self.update_status(status_msg)
            self.calib_status_label.config(text="Status: Calibrated")
            self.calib_ratio_label.config(text=f"Ratio (px/cm): {self.pixels_per_cm:.2f}") # Updated label

            # Enable grid size application and recalculate
            self.apply_grid_button.config(state=tk.NORMAL)
            self.apply_grid_size() # Recalculate pixel size based on current cm entry

            self.update_ui() # Redraw canvas and update lists/buttons
            self._write_calibration_json() # Write to file

        except tk.TclError:
            self.update_status("Calibration Error: Invalid distance value. Please enter a number.")
            messagebox.showerror("Error", "Please enter a valid number for the distance.", parent=self.root)
        except Exception as e:
            self.update_status(f"Calibration Error: {str(e)}")
            messagebox.showerror("Error", f"An unexpected error occurred during calibration: {str(e)}", parent=self.root)
            self.reset_calibration() # Reset on unexpected error

    def apply_grid_size(self):
        """Applies the grid size based on the entry value."""
        if self.pixels_per_cm is None:
            self.update_status("Cannot apply grid size: Calibration required.")
            return
        try:
            # Get the grid size in centimeters
            grid_size_cm = float(self.grid_size_cm.get())
            if grid_size_cm <= 0:
                raise ValueError("Grid size must be positive")

            # Calculate grid size in pixels
            self.grid_size_pixels = grid_size_cm * self.pixels_per_cm

            # Update status and UI label
            self.update_status(f"Grid size set: {grid_size_cm} cm ({self.grid_size_pixels:.1f} pixels)")
            self.grid_pixel_label.config(text=f"Pixel Size (px): {self.grid_size_pixels:.1f}")

            # Write calibration data to JSON (includes grid size)
            self._write_calibration_json()

        except (ValueError, tk.TclError) as e:
            self.update_status(f"Error: Invalid grid size value. {str(e)}")
            self.grid_pixel_label.config(text="Pixel Size (px): Invalid")
        except Exception as e:
            self.update_status(f"Error applying grid size: {e}")
            self.grid_pixel_label.config(text="Pixel Size (px): Error")

    def _write_calibration_json(self, json_path="calibration.json"):
        """Writes calibration data to calibration.json in the sam2 directory."""
        # Construct path relative to this script's directory
        script_dir = os.path.dirname(__file__)
        full_json_path = os.path.join(script_dir, json_path)
        print(f"[INFO] Writing calibration data to JSON: {full_json_path}")

        # Calculate grid dimensions if possible
        self.grid_cols = None
        self.grid_rows = None
        if self.grid_size_pixels is not None and self.image_array is not None and self.grid_size_pixels > 0:
            height, width = self.image_array.shape[:2]
            self.grid_cols = int(round(width / self.grid_size_pixels)) # Round to nearest int
            self.grid_rows = int(round(height / self.grid_size_pixels))
            print(f"[INFO] Calculated grid dimensions: {self.grid_cols} columns x {self.grid_rows} rows")
        elif self.grid_size_pixels is not None and self.grid_size_pixels <= 0:
             print("[WARN] Grid size in pixels is zero or negative, cannot calculate grid dimensions.")
        elif self.grid_size_pixels is None:
             print("[INFO] Grid size in pixels not yet calculated.")


        # Prepare calibration data
        calib_data = {
            "pixel_ratio": {
                "value": self.pixels_per_cm if self.pixels_per_cm is not None else 0.0,
                "unit": "pixels/cm"
            },
            "grid": {
                "columns": self.grid_cols, # Can be None if not calculated
                "rows": self.grid_rows,   # Can be None if not calculated
                "cell_size_cm": float(self.grid_size_cm.get()) if self.grid_size_cm.get() else None,
                "cell_size_pixels": self.grid_size_pixels # Can be None
            },
            "aruco": {
                "marker_size": 0.086  # Default ArUco marker size in meters
            }
        }

        # Write calibration data to JSON file
        try:
            with open(full_json_path, 'w') as f:
                json.dump(calib_data, f, indent=4)
            print(f"[INFO] Successfully wrote calibration data to {full_json_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write calibration JSON to {full_json_path}: {e}")
            return False
        return True

    def reset_calibration(self, keep_status=False):
        """Resets calibration variables, UI elements, and related grid settings."""
        self.calibration_state = "idle"
        self.calibration_points = [None, None]
        self.cm_per_pixel = None
        self.pixels_per_cm = None # Reset this too
        self.grid_size_pixels = None # Grid size depends on calibration
        self.grid_cols = None
        self.grid_rows = None

        if not keep_status:
             self.update_status("Calibration reset.")

        # Update UI elements related to calibration
        self.calib_status_label.config(text="Status: Select P1 & P2")
        self.calib_ratio_label.config(text="Ratio (px/cm): N/A") # Updated label
        self.set_calib_button.config(state=tk.DISABLED)
        self.apply_grid_button.config(state=tk.DISABLED)
        self.grid_pixel_label.config(text="Pixel Size (px): N/A")
        self.p1_select_active.set(False)
        self.p2_select_active.set(False)

        # Redraw canvas (will remove P1/P2 markers)
        self.redraw_canvas() # This is usually called by update_ui
    
    def delete_selected_points(self):
        """Delete points selected in the Treeview list."""
        selected_items = self.point_tree.selection()
        if not selected_items:
            self.update_status("No points selected in the list to delete.")
            return

        # Store deletions to perform after iteration
        points_to_delete = {'pos': [], 'neg': [], 'fixed': [], 'calib': []} # Store indices or keys

        for item_id in selected_items:
            parts = item_id.split('_')
            ptype = parts[0]
            try:
                if ptype == 'pos' or ptype == 'neg':
                    index = int(parts[1])
                    points_to_delete[ptype].append(index)
                elif ptype == 'fixed':
                    label = parts[1]
                    index = int(parts[2])
                    points_to_delete['fixed'].append({'label': label, 'index': index})
                elif ptype == 'calib':
                    index = int(parts[1]) # 0 for P1, 1 for P2
                    points_to_delete['calib'].append(index)
            except (IndexError, ValueError):
                print(f"Warning: Could not parse item ID for deletion: {item_id}")
                continue

        deleted_count = 0

        # --- Perform Deletions ---
        # It's crucial to delete from lists in reverse index order to avoid messing up subsequent indices

        # Delete Positive/Negative points (from self.points and self.point_labels)
        pos_indices = sorted([idx for idx in points_to_delete['pos']], reverse=True)
        neg_indices = sorted([idx for idx in points_to_delete['neg']], reverse=True)
        all_prompt_indices = sorted(list(set(pos_indices + neg_indices)), reverse=True)

        prompt_points_map = {i: p for i, p in enumerate(self.points)} # Map original index to point

        for index in all_prompt_indices:
            if index < len(self.points): # Check if index is still valid
                deleted_point = self.points.pop(index)
                deleted_label_flag = self.point_labels.pop(index)
                deleted_count += 1

                # Also remove from label_map structure
                if deleted_label_flag == 1: # Positive
                    found = False
                    for lbl, pts in self.label_map.get("points", {}).items():
                         try: pts.remove(deleted_point); found = True; break
                         except ValueError: continue
                    # if not found: print(f"Warning: Deleted positive point {deleted_point} not found in label_map['points']")
                else: # Negative
                     try: self.label_map.get("negative_points", []).remove(deleted_point)
                     except ValueError: pass # Ignore if not found
            else:
                 print(f"Warning: Index {index} out of bounds for self.points during deletion.")


        # Delete Fixed points (from self.label_map["fixed_points"] and ["fixed_point_ids"])
        # Group deletions by label, then sort indices descending for each label
        fixed_by_label = {}
        for item in points_to_delete['fixed']:
            fixed_by_label.setdefault(item['label'], []).append(item['index'])

        for label, indices in fixed_by_label.items():
             sorted_indices = sorted(indices, reverse=True)
             if label in self.label_map.get("fixed_points", {}):
                 points_list = self.label_map["fixed_points"][label]
                 ids_list = self.label_map.get("fixed_point_ids", {}).get(label, [])
                 ids_list_exists = label in self.label_map.get("fixed_point_ids", {})

                 for index in sorted_indices:
                     if index < len(points_list):
                         points_list.pop(index)
                         deleted_count += 1
                         if ids_list_exists and index < len(ids_list):
                             ids_list.pop(index)
                     else:
                          print(f"Warning: Index {index} out of bounds for fixed points list of label '{label}'.")


        # Delete Calibration points
        calib_indices = sorted(list(set(points_to_delete['calib'])), reverse=True)
        calib_deleted = False
        for index in calib_indices:
             if index == 0 or index == 1:
                 if self.calibration_points[index] is not None:
                     self.calibration_points[index] = None
                     deleted_count += 1
                     calib_deleted = True

        if calib_deleted:
             self.update_status(f"Deleted {deleted_count} point(s). Calibration points modified.")
             # Reset calibration calculation if points are removed
             self.cm_per_pixel = None
             self.pixels_per_cm = None
             self.grid_size_pixels = None
             self.check_calibration_readiness() # Update status and buttons
        else:
            self.update_status(f"Deleted {deleted_count} point(s).")

        # Refresh the UI completely
        self.update_ui()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        key = event.char
        # Standard modes
        if key == '1': self.mode_var.set("point_positive"); self.set_mode()
        elif key == '2': self.mode_var.set("point_negative"); self.set_mode()
        elif key == '3': self.mode_var.set("box"); self.set_mode()
        elif key == '4': self.mode_var.set("fixed_point"); self.set_mode()
        # Removed '5' for delete mode
        # Actions
        elif key == 'c': self.clear_all()
        elif key == 'g': self.generate_masks()
        elif key == 's': self.save_grid()
        # Calibration toggles (optional)
        elif key == 'p': self.toggle_select_p1() # 'p' for P1
        elif key == 'P': self.toggle_select_p2() # Shift+'p' for P2
    
    def clear_all(self):
        """Clear all points, boxes, masks, and reset calibration/UI."""
        self.points = []
        self.point_labels = []
        self.boxes = []
        self.current_box = None
        # self.generated_masks = [] # Not used anymore
        self.selected_masks = {}
        # self.mask_scores = [] # Not used anymore
        self.label_masks = {} # Clear generated but unselected masks
        self.label_mask_scores = {}

        # Reset label map but keep labels list
        self.label_map = {}
        self.update_label_map() # Re-initialize structure

        # Reset fixed point counter
        self.fixed_point_id_counter = 100

        # Reset calibration fully
        self.reset_calibration()

        # Clear Treeview explicitly (though update_ui will do it too)
        self.point_tree.delete(*self.point_tree.get_children())

        # --- Reset the predictor flag ---
        self.image_set_in_predictor = False
        # --------------------------------

        self.update_status("Cleared all data.")
        self.update_ui() # Update all UI elements, including canvas and point list
    
    def generate_masks(self):
        """Generate masks using SAM2 predictor based on prompts for each label"""
        if self.image is None: # Check for PIL image
            self.update_status("Please open an image first.")
            return

        # Check if predictor is initialized and image is set
        if self.predictor is None:
            self.update_status("Model not initialized. Trying to initialize...")
            self.initialize_model()
            if self.predictor is None:
                self.update_status("Model initialization failed. Cannot generate masks.")
                return # Stop if initialization failed
        if not self.image_set_in_predictor:
            self.update_status("Image not set in predictor. Trying to set...")
            self.set_image_in_predictor()
            if not self.image_set_in_predictor:
                self.update_status("Failed to set image in predictor. Cannot generate masks.")
                return # Stop if image setting failed

        # Check if we have any positive points associated with any label
        has_positive_points = False
        if "points" in self.label_map:
            for label in self.labels:
                if label in self.label_map.get("points", {}) and self.label_map["points"][label]:
                    has_positive_points = True
                    break

        # Check if we have any boxes (associated with the current label)
        has_boxes = bool(self.boxes)

        if not has_positive_points and not has_boxes:
             self.update_status("No positive points or boxes provided. Please add prompts.")
             return

        # Start timing
        start_time_total = time.time()
        self.update_status("Generating masks for each label... please wait")

        # Reset generated masks and scores
        self.label_masks = {}
        self.label_mask_scores = {}

        # Get global negative points
        negative_points = self.label_map.get("negative_points", [])

        # Prepare prompts for each label with positive points
        labels_to_process = [
            label for label in self.labels
            if label in self.label_map.get("points", {}) and self.label_map["points"][label]
        ]
        # Also process the current label if boxes exist, even if no points
        if self.boxes and self.current_label not in labels_to_process:
             # Only add if it's a valid label in our map
             if self.current_label in self.label_map:
                 labels_to_process.append(self.current_label)

        label_times = {}

        # Process each label that has positive points or boxes (if current label)
        try:
            for label in labels_to_process:
                start_time_label = time.time()
                positive_points_for_label = self.label_map.get("points", {}).get(label, [])
                # Only use boxes if this label is the currently selected one in the UI
                boxes_for_label = self.boxes if label == self.current_label else []

                # Combine all points (positive for this label, global negative)
                point_coords_list = positive_points_for_label + negative_points
                point_labels_list = [1] * len(positive_points_for_label) + [0] * len(negative_points)

                # Format for predictor: dictionary of prompts
                prompts = {}
                if point_coords_list:
                    prompts['point_coords'] = np.array(point_coords_list)
                if point_labels_list:
                    prompts['point_labels'] = np.array(point_labels_list)
                if boxes_for_label:
                    # Predictor expects boxes as [N, 4]
                    prompts['box_coords'] = np.array(boxes_for_label)

                # Skip if no prompts for this specific label combination
                if not prompts:
                    print(f"Skipping label '{label}' as it has no positive points or associated boxes.")
                    continue

                print(f"--- Processing Label: {label} ---")
                if 'point_coords' in prompts: print(f"  Points: {len(positive_points_for_label)} positive, {len(negative_points)} negative")
                if 'box_coords' in prompts: print(f"  Boxes: {len(boxes_for_label)}")

                # Predict using the predictor within inference mode and autocast
                with torch.inference_mode():
                    # Use autocast based on the globally determined device
                    if device.type == 'cuda':
                        # Use autocast with bfloat16 for CUDA
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            masks_out, scores_out, _ = self.predictor.predict(**prompts, multimask_output=True)
                    else:
                        # For CPU or MPS, run in default precision (usually float32)
                        # Autocast might offer less benefit or require different dtypes (e.g., float16 for MPS)
                        # Keeping it simple by running without autocast for non-CUDA devices.
                        masks_out, scores_out, _ = self.predictor.predict(**prompts, multimask_output=True)

                # --- Process predictor outputs ---
                # Predictor outputs numpy arrays directly
                masks = masks_out.astype(bool) # Shape: (num_masks, H, W)
                scores = scores_out # Shape: (num_masks,)

                # Ensure masks and scores have expected dimensions (remove batch dim if present)
                # This check might be unnecessary if the predictor always returns the correct shape
                if masks.ndim == 4 and masks.shape[0] == 1: masks = masks.squeeze(0)
                if scores.ndim == 2 and scores.shape[0] == 1: scores = scores.squeeze(0)

                print(f"  DEBUG: Label '{label}' - Predictor outputs (NumPy arrays):")
                print(f"  DEBUG: masks shape: {masks.shape}, scores shape: {scores.shape}")

                if masks.size == 0 or scores.size == 0 or masks.shape[0] != scores.shape[0]:
                     print(f"  Warning: Empty or mismatched masks/scores received for label '{label}'. Skipping.")
                     continue

                # Sort by score
                sorted_ind = np.argsort(scores)[::-1] # sorted_ind should now be 1D, e.g., [1 0 2]

                masks = masks[sorted_ind]
                scores = scores[sorted_ind]

                # Store for this label
                self.label_masks[label] = masks
                self.label_mask_scores[label] = scores

                # Calculate and store time for this label
                end_time_label = time.time()
                label_times[label] = end_time_label - start_time_label
                print(f"  Generated {len(masks)} masks for '{label}' in {label_times[label]:.2f}s")

            # --- Post-Generation ---
            end_time_total = time.time()
            total_time = end_time_total - start_time_total
            self.selected_masks = {} # Clear previously selected masks

            if self.label_masks:
                timing_msg = f"Generated masks for {len(self.label_masks)} labels in {total_time:.2f}s.\n"
                for lbl, t in label_times.items():
                    timing_msg += f"  - '{lbl}': {t:.2f}s ({len(self.label_masks[lbl])} masks)\n"
                print(timing_msg)
                self.update_status(timing_msg.split('\n')[0] + " Select masks in dialog.") # Short status

                self.show_multi_label_mask_selection() # Show dialog (waits for dialog)

                # Status after dialog closes is handled within the dialog's apply/cancel actions
            else:
                self.update_status(f"No masks generated (process took {total_time:.2f}s). Add positive points/boxes.")

        except Exception as e:
            self.update_status(f"Error during mask generation: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
             # Ensure UI is updated even if errors occur
             self.update_ui()
    
    def show_multi_label_mask_selection(self):
        """Show a dialog for selecting one mask per label that had masks generated."""
        if not self.label_masks:
            self.update_status("No generated masks available to select from.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Best Mask for Each Label")
        dialog.geometry("1000x800")
        dialog.transient(self.root)
        dialog.grab_set() # Make modal

        # Positioning (optional, but nice)
        # ... (positioning code omitted for brevity, same as before) ...

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_frame, text="Select ONE Mask for Each Label", font=("Arial", 14, "bold")).pack(pady=5)
        ttk.Label(main_frame, text="Labels without generated masks are not shown. Select the best mask found for each label below.", font=("Arial", 10)).pack(pady=5)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        selection_vars = {}
        event_bindings = [] # To store bindings for later unbinding

        for label in self.label_masks: # Only iterate through labels that have masks
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=f"Label: {label} ({len(self.label_masks[label])} masks)")

            # Scrollable frame setup
            canvas_frame = ttk.Frame(tab)
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            mask_display_canvas = tk.Canvas(canvas_frame, borderwidth=0, background="#ffffff") # Explicit background
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=mask_display_canvas.yview)
            scrollable_frame = ttk.Frame(mask_display_canvas) # Frame to hold the mask previews

            scrollable_frame.bind("<Configure>", lambda e, c=mask_display_canvas: c.configure(scrollregion=c.bbox("all")))
            canvas_window = mask_display_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            mask_display_canvas.configure(yscrollcommand=scrollbar.set)

            mask_display_canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Mousewheel scrolling function
            def _on_mousewheel(event, canvas=mask_display_canvas):
                # Platform-specific scrolling adjustment
                delta = 0
                if sys.platform == 'darwin': delta = event.delta # macOS
                elif event.num == 4: delta = -120 # Linux scroll up
                elif event.num == 5: delta = 120  # Linux scroll down
                else: delta = event.delta # Windows

                canvas.yview_scroll(int(-1 * (delta / 120)), "units")

            # Bind scroll wheel events (consider binding specifics per platform if needed)
            scroll_bind_tag = f"MouseWheel-{label}" # Unique tag
            mask_display_canvas.bind_class(scroll_bind_tag, '<MouseWheel>', _on_mousewheel)
            # For Linux compatibility
            mask_display_canvas.bind_class(scroll_bind_tag, '<Button-4>', _on_mousewheel)
            mask_display_canvas.bind_class(scroll_bind_tag, '<Button-5>', _on_mousewheel)
            # Add tag to instance bindings
            current_bindtags = list(mask_display_canvas.bindtags())
            if scroll_bind_tag not in current_bindtags:
                mask_display_canvas.bindtags(tuple(current_bindtags[:1]) + (scroll_bind_tag,) + tuple(current_bindtags[1:]))


            # Allow focus for scrolling without clicking inside
            focus_bind_tag = f"Focus-{label}"
            mask_display_canvas.bind_class(focus_bind_tag, '<Enter>', lambda e, c=mask_display_canvas: c.focus_set())
            # Add tag to instance bindings
            current_bindtags = list(mask_display_canvas.bindtags())
            if focus_bind_tag not in current_bindtags:
                 mask_display_canvas.bindtags(tuple(current_bindtags[:1]) + (focus_bind_tag,) + tuple(current_bindtags[1:]))


            event_bindings.append((mask_display_canvas, scroll_bind_tag, focus_bind_tag)) # Store for cleanup

            # --- Display Masks for this Label ---
            selection_vars[label] = tk.IntVar(value=-1) # -1 means no selection initially
            masks = self.label_masks[label]
            scores = self.label_mask_scores[label]
            num_masks = len(masks)
            masks_per_row = 3

            for i in range(0, num_masks, masks_per_row):
                row_frame = ttk.Frame(scrollable_frame)
                row_frame.pack(fill=tk.X, pady=10)

                for j in range(masks_per_row):
                    idx = i + j
                    if idx < num_masks:
                        mask_frame = ttk.LabelFrame(row_frame, text=f"Mask {idx+1}, Score: {scores[idx]:.3f}")
                        mask_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

                        fig = Figure(figsize=(2.5, 2.5), dpi=90) # Slightly smaller previews
                        ax = fig.add_subplot(111)
                        ax.imshow(self.image_array)
                        # Use helper that works on axes
                        self.show_mask_on_axes(masks[idx], ax, random_color=True, alpha=0.7)
                        ax.set_axis_off()
                        fig.tight_layout(pad=0) # Reduce padding

                        mask_preview_canvas = FigureCanvasTkAgg(fig, master=mask_frame)
                        mask_preview_canvas.draw()
                        mask_widget = mask_preview_canvas.get_tk_widget()
                        mask_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

                        rb = ttk.Radiobutton(mask_frame, text=f"Select Mask {idx+1}",
                                             variable=selection_vars[label], value=idx)
                        rb.pack(pady=(0,5), anchor='center')

                        # Click image/frame to select radio button
                        select_func = lambda e=None, var=selection_vars[label], val=idx: var.set(val)
                        mask_widget.bind("<Button-1>", select_func)
                        mask_frame.bind("<Button-1>", select_func) # Bind to frame too
                        # Need to unbind these later if necessary, but probably okay


        # --- Dialog Buttons ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        def cleanup_bindings():
            """Unbind events associated with the dialog canvases."""
            for canvas, scroll_tag, focus_tag in event_bindings:
                 try:
                      # Unbind by removing the class tag bindings we added
                      current_tags = list(canvas.bindtags())
                      if scroll_tag in current_tags: current_tags.remove(scroll_tag)
                      if focus_tag in current_tags: current_tags.remove(focus_tag)
                      canvas.bindtags(tuple(current_tags))
                      # Explicitly unbind instance-level bindings if class binding removal isn't enough
                      canvas.unbind('<MouseWheel>')
                      canvas.unbind('<Button-4>')
                      canvas.unbind('<Button-5>')
                      canvas.unbind('<Enter>')
                 except Exception as e:
                      print(f"Warning: Error during binding cleanup: {e}")

        def apply_selections():
            missing_selections = [lbl for lbl in self.label_masks if selection_vars[lbl].get() == -1]
            if missing_selections:
                messagebox.showwarning("Missing Selections", f"Please select a mask for all labels: {', '.join(missing_selections)}", parent=dialog)
                return

            self.selected_masks = {}
            num_selected = 0
            for label in self.label_masks:
                selected_idx = selection_vars[label].get()
                if selected_idx != -1: # Should always be true based on check above
                    try:
                        mask = self.label_masks[label][selected_idx]
                        if isinstance(mask, np.ndarray):
                            self.selected_masks[label] = mask
                            num_selected += 1
                        else: print(f"Warning: Selected mask for {label} is not a numpy array.")
                    except IndexError: print(f"Warning: Invalid index {selected_idx} for label {label}.")
                    except Exception as e: print(f"Error storing selected mask for {label}: {e}")

            cleanup_bindings()
            self.update_status(f"Applied {num_selected} selected masks for grid generation.")
            self.update_ui() # Update main UI to show selected masks on canvas
            dialog.destroy()

        def on_cancel():
            cleanup_bindings()
            self.update_status("Mask selection cancelled.")
            self.selected_masks = {} # Ensure no masks are selected if cancelled
            self.update_ui() # Update main UI
            dialog.destroy()

        ttk.Button(button_frame, text="Apply Selections", command=apply_selections).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=10, pady=10)
        dialog.protocol("WM_DELETE_WINDOW", on_cancel) # Handle closing window

        self.root.wait_window(dialog) # Wait for the dialog to close
    
    def show_mask_on_axes(self, mask, ax, random_color=False, alpha=0.6):
        """Helper to show a single mask on given matplotlib axes (used in dialog)."""
        if mask is None or not isinstance(mask, np.ndarray):
             print(f"Error in show_mask_on_axes: Invalid mask type: {type(mask)}")
             return

        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, alpha])

        try:
            h, w = mask.shape[-2:]
            mask_uint8 = mask.astype(np.uint8) # Ensure uint8
            mask_image = mask_uint8.reshape(h, w, 1) * color.reshape(1, 1, -1)

            # Optional: Draw contours for better definition
            try:
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Draw thin white border
                mask_image = cv2.drawContours(mask_image, contours, -1, (1.0, 1.0, 1.0, 0.7), thickness=1)
            except Exception: pass # Ignore contour errors here

            ax.imshow(mask_image)
        except Exception as e:
             print(f"Error in show_mask_on_axes: {e}, mask shape: {getattr(mask, 'shape', 'unknown')}")

    def _copy_to_working_dataset(self, source_dir):
        """Create a copy of the dataset in working_dataset, overwriting if exists."""
        try:
            # Create working_dataset directory in the SAM root directory
            working_dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "working_dataset")

            # Remove existing working_dataset directory if it exists
            if os.path.exists(working_dataset_dir):
                print(f"Removing existing directory: {working_dataset_dir}")
                shutil.rmtree(working_dataset_dir)

            # Recreate the working_dataset directory (ensures it's empty)
            os.makedirs(working_dataset_dir, exist_ok=True)

            # Copy the source directory contents into working_dataset
            # copytree copies content of source_dir INTO working_dataset_dir
            print(f"Copying contents of {source_dir} into {working_dataset_dir}")
            shutil.copytree(source_dir, working_dataset_dir, dirs_exist_ok=True) # Use dirs_exist_ok for robustness

            print(f"Copied dataset to {working_dataset_dir}")
        except Exception as e:
            print(f"Error copying to working dataset: {e}")
            self.update_status(f"Error copying to working dataset: {e}")

    def save_grid(self):
        """Generate grid from selected masks and save all relevant data."""
        if self.image_array is None: self.update_status("Please open an image first."); return
        if self.pixels_per_cm is None or self.grid_size_pixels is None:
             messagebox.showwarning("Calibration Needed", "Please calibrate the image and apply a grid size before saving.", parent=self.root)
             return

        if not self.selected_masks:
             # Check if masks were generated but not selected
             if hasattr(self, 'label_masks') and self.label_masks:
                  messagebox.showinfo("Select Masks", "No masks selected. Please use 'Generate Masks' and select masks in the dialog first.", parent=self.root)
             else: # No masks generated at all
                  messagebox.showinfo("Generate Masks", "No masks available. Please use 'Generate Masks' first.", parent=self.root)
             return

        start_time_total = time.time()
        self.update_status("Generating and saving grid...")

        # Generate grid first to ensure grid dimensions are calculated
        labeled_grid = self.generate_multi_label_grid()
        if labeled_grid is None or labeled_grid.size == 0:
             self.update_status("Grid generation failed. Cannot save.")
             return

        # Write calibration data to JSON with the calculated grid dimensions
        # (generate_multi_label_grid should have updated self.grid_cols/rows)
        self._write_calibration_json()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_dir = os.path.join(self.dataset_dir, timestamp)
        masks_dir = os.path.join(timestamp_dir, "masks")
        grid_dir = os.path.join(timestamp_dir, "grid")

        try:
            os.makedirs(timestamp_dir)
            os.makedirs(masks_dir)
            os.makedirs(grid_dir)
        except OSError as e:
            messagebox.showerror("Save Error", f"Failed to create directories in {self.dataset_dir}: {e}", parent=self.root)
            self.update_status(f"Directory creation error: {e}")
            return

        # --- Save Data ---
        # 1. Original Image
        original_image_path = os.path.join(timestamp_dir, "original_image.png")
        try:
            Image.fromarray(self.image_array).save(original_image_path)
        except Exception as e: print(f"Error saving original image: {e}") # Non-critical

        # 2. Label Map (including calibration and grid settings)
        # Ensure structure is clean before saving
        self.update_label_map() # Should ensure all labels have entries
        # Add current calibration/grid state to a copy for saving
        save_label_map = json.loads(json.dumps(self.label_map)) # Deep copy for basic types
        if self.pixels_per_cm is not None: # Use pixels_per_cm
            save_label_map["_calibration_"] = {
                "pixels_per_cm": self.pixels_per_cm, # Save this ratio
                "cm_per_pixel": self.cm_per_pixel, # Also save inverse if needed
                "calibration_points": self.calibration_points, # [[x,y], [x,y]] or [None, None] etc.
                "calibration_distance_cm": self.calibration_distance.get()
            }
        if self.grid_size_pixels is not None:
            save_label_map["_grid_settings_"] = {
                "grid_size_cm": self.grid_size_cm.get(),
                "grid_size_pixels": self.grid_size_pixels,
                "grid_cols": self.grid_cols, # Save calculated dimensions
                "grid_rows": self.grid_rows
            }
        label_map_path = os.path.join(timestamp_dir, "label_map.json")
        try:
            with open(label_map_path, 'w') as f:
                json.dump(save_label_map, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save label_map.json: {e}", parent=self.root)
            # Continue saving other things? Maybe return here if label map is critical.
            # return

        # 3. Selected Masks (NPY and PNG vis)
        mask_paths = {}
        mask_vis_paths = {}
        for label, mask in self.selected_masks.items():
            safe_label = label.replace(" ", "_").replace("/", "-") # Sanitize label for filename
            mask_npy_path = os.path.join(masks_dir, f"mask_{safe_label}.npy")
            mask_png_path = os.path.join(masks_dir, f"mask_{safe_label}_vis.png")
            try:
                np.save(mask_npy_path, mask)
                mask_paths[label] = mask_npy_path # Store original label as key
            except Exception as e: print(f"Error saving mask NPY for {label}: {e}")

            try: # Save visualization
                plt.figure(figsize=(self.image_width/100, self.image_height/100), dpi=100) # Match image size roughly
                plt.imshow(self.image_array)
                self.show_mask_on_axes(mask, plt.gca(), random_color=True, alpha=0.6)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(mask_png_path, bbox_inches='tight', pad_inches=0)
                plt.close() # Close figure
                mask_vis_paths[label] = mask_png_path
            except Exception as e: print(f"Error saving mask PNG vis for {label}: {e}")

        # 4. Save Generated Grid (already generated)
        grid_start_time = time.time() # Time only saving part now
        grid_npy_path = os.path.join(grid_dir, "labeled_grid.npy")
        grid_vis_path = os.path.join(grid_dir, "labeled_grid_vis.png")
        try:
            np.save(grid_npy_path, labeled_grid)
        except Exception as e:
            grid_npy_path = None # Mark as failed
            print(f"Error saving grid NPY: {e}")
        try:
            grid_vis = self.visualize_grid(labeled_grid)
            if grid_vis is not None:
                # Use BGRA for PNG alpha
                cv2.imwrite(grid_vis_path, cv2.cvtColor(grid_vis, cv2.COLOR_RGBA2BGRA))
            else:
                 grid_vis_path = None # Mark as failed
                 print("Error: Grid visualization was None.")
        except Exception as e:
            grid_vis_path = None # Mark as failed
            print(f"Error saving grid visualization: {e}")
        grid_time = time.time() - grid_start_time # Time for saving NPY and VIS

        # 5. Save Metadata JSONs (grid_info.json, summary.json)
        end_time_total = time.time()
        total_time = end_time_total - start_time_total

        # Helper function to convert numpy arrays to lists for JSON
        def numpy_to_list_converter(obj):
             if isinstance(obj, np.ndarray): return obj.tolist()
             if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
             if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
             if isinstance(obj, (np.bool_)): return bool(obj)
             if isinstance(obj, (np.void)): return None # Or handle structured arrays differently
             # Let json encoder handle other types or raise error
             # raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
             return obj # Let default encoder handle it


        grid_info = {
            "grid_path": os.path.relpath(grid_npy_path, start=timestamp_dir) if grid_npy_path else None,
            "grid_vis_path": os.path.relpath(grid_vis_path, start=timestamp_dir) if grid_vis_path else None,
            "mask_paths": {lbl: os.path.relpath(p, start=timestamp_dir) for lbl, p in mask_paths.items()},
            "mask_vis_paths": {lbl: os.path.relpath(p, start=timestamp_dir) for lbl, p in mask_vis_paths.items()},
            "label_map": save_label_map, # Contains calib/grid info
            "resolution": {"height": self.image_height, "width": self.image_width},
            "timestamp": timestamp,
            "original_image_path": os.path.relpath(original_image_path, start=timestamp_dir),
            "selected_labels": list(self.selected_masks.keys()),
            "timing": {"grid_save_time": grid_time, "total_save_time": total_time} # Updated timing key
        }
        grid_info_path = os.path.join(grid_dir, "grid_info.json")
        try:
            with open(grid_info_path, 'w') as f:
                 # Use default=numpy_to_list_converter for robust handling
                 json.dump(grid_info, f, indent=2, default=numpy_to_list_converter)
        except Exception as e: print(f"Error saving grid_info.json: {e}")


        summary_info = {
            "timestamp": timestamp,
            "original_image_path": os.path.relpath(original_image_path, start=timestamp_dir),
            "label_map_path": os.path.relpath(label_map_path, start=timestamp_dir),
            # "label_map": save_label_map, # Redundant if label_map_path is present
            "mask_paths": {lbl: os.path.relpath(p, start=timestamp_dir) for lbl, p in mask_paths.items()},
            "grid_path": os.path.relpath(grid_npy_path, start=timestamp_dir) if grid_npy_path else None,
            "grid_vis_path": os.path.relpath(grid_vis_path, start=timestamp_dir) if grid_vis_path else None,
            "selected_labels": list(self.selected_masks.keys()),
            "timing": {"grid_save_time": grid_time, "total_save_time": total_time}
        }
        summary_path = os.path.join(timestamp_dir, "summary.json")
        try:
            with open(summary_path, 'w') as f:
                 json.dump(summary_info, f, indent=2, default=numpy_to_list_converter)
        except Exception as e: print(f"Error saving summary.json: {e}")


        final_status = f"Saved grid and data to {timestamp_dir} ({total_time:.2f}s)"
        self.update_status(final_status)
        print(final_status)

        # Copy to working dataset
        self._copy_to_working_dataset(timestamp_dir)

        # Show result dialog if visualization succeeded
        if grid_vis_path and os.path.exists(grid_vis_path):
            self.show_grid_result(grid_vis_path, timestamp_dir, ", ".join(self.selected_masks.keys()))
        else:
            messagebox.showinfo("Save Complete", f"Data saved to {timestamp_dir}\n(Grid visualization failed or was not generated)", parent=self.root)

    def show_grid_result(self, grid_vis_path, save_dir, label_name):
        """Show a dialog with the grid generation result (visualization)."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Grid Generation Result")
        # dialog.geometry("800x600") # Let it auto-size based on content
        dialog.transient(self.root)

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_frame, text=f"Grid Generated for Label(s): {label_name}", font=("Arial", 14, "bold")).pack(pady=10)

        info_text = f"✓ Data saved to: {save_dir}"
        ttk.Label(main_frame, text=info_text, justify=tk.LEFT, font=("Arial", 10)).pack(pady=5, fill=tk.X)

        try:
            img = Image.open(grid_vis_path)
            # Resize proportionally to fit max width/height
            max_w, max_h = 700, 550 # Max dimensions for preview
            img.thumbnail((max_w, max_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            dialog.photo = photo # Keep reference

            img_label = ttk.Label(main_frame, image=photo)
            img_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            ttk.Label(main_frame, text=f"Preview: {os.path.basename(grid_vis_path)}").pack(pady=(0,5))
        except Exception as e:
            ttk.Label(main_frame, text=f"Error displaying preview image: {str(e)}").pack(pady=10)

        ttk.Button(main_frame, text="Close", command=dialog.destroy).pack(pady=10)
        dialog.update_idletasks() # Ensure size is calculated before centering
        # Optional: Center dialog
        # ... (centering code omitted) ...
        dialog.focus_set()

    def generate_multi_label_grid(self):
        """Generate a multi-label grid by combining selected masks, applying fixed points last."""
        if self.image_array is None: return np.zeros((1,1), dtype=np.int32) # Return empty if no image
        if self.grid_size_pixels is None or self.grid_size_pixels <= 0:
             self.update_status("Error: Grid size not calculated or invalid. Cannot generate grid.")
             messagebox.showerror("Grid Error", "Grid size is not properly set. Please calibrate and apply grid size.", parent=self.root)
             return None # Indicate failure

        height, width = self.image_array.shape[:2]

        # Calculate grid dimensions based on grid size in pixels (round to nearest int)
        self.grid_cols = int(round(width / self.grid_size_pixels))
        self.grid_rows = int(round(height / self.grid_size_pixels))

        if self.grid_cols <= 0 or self.grid_rows <= 0:
             self.update_status(f"Error: Invalid grid dimensions ({self.grid_cols}x{self.grid_rows}). Check calibration/grid size.")
             messagebox.showerror("Grid Error", f"Calculated grid dimensions ({self.grid_cols}x{self.grid_rows}) are invalid.", parent=self.root)
             return None

        print(f"Grid dimensions: {self.grid_cols} columns x {self.grid_rows} rows")

        # Create a full resolution mask grid first
        full_res_grid = np.zeros((height, width), dtype=np.int32) # Start with background 0

        print(f"Generating multi-label grid from {len(self.selected_masks)} selected masks.")

        # Sort selected masks by label value (lower values first, overwrite later)
        sorted_items = sorted(
            [(label, mask) for label, mask in self.selected_masks.items() if label in self.label_map],
            key=lambda x: self.label_map.get(x[0], float('inf')) # Use inf for labels not in map (shouldn't happen)
        )

        # Apply masks to full resolution grid
        for label, mask in sorted_items:
            label_value = self.label_map.get(label)
            if label_value is None: # Should not happen if label_map is correct
                 print(f"    Warning: Label '{label}' not found in label_map. Skipping.")
                 continue
            print(f"  Applying mask for '{label}' (Value: {label_value})")

            if not isinstance(mask, np.ndarray) or mask.shape[-2:] != (height, width):
                print(f"    Warning: Invalid mask shape for {label}. Expected ({height}, {width}), got {getattr(mask, 'shape', 'N/A')}. Skipping.")
                continue

            # Ensure mask is boolean
            if mask.ndim > 2: mask = mask.squeeze() # Remove leading dims if present
            if mask.dtype != bool: mask = mask > 0.5 # Threshold if needed (e.g., if logits were saved)

            full_res_grid[mask] = label_value # Apply label value where mask is true

        # Create the final grid with proper dimensions
        final_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        # For each grid cell, find the most frequent label
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                # Calculate pixel boundaries for this grid cell (ensure integers)
                # Use precise pixel boundaries based on rounded grid size
                y_start = int(round(i * self.grid_size_pixels))
                y_end = int(round(min((i + 1) * self.grid_size_pixels, height)))
                x_start = int(round(j * self.grid_size_pixels))
                x_end = int(round(min((j + 1) * self.grid_size_pixels, width)))

                # Handle edge cases where rounding might make start >= end
                if y_start >= y_end or x_start >= x_end: continue

                # Get the region from full resolution grid
                cell_region = full_res_grid[y_start:y_end, x_start:x_end]

                # Find the most frequent non-zero value
                if cell_region.size > 0 and np.any(cell_region > 0):
                    values, counts = np.unique(cell_region[cell_region > 0], return_counts=True)
                    final_grid[i, j] = values[np.argmax(counts)]
                else:
                    final_grid[i, j] = 0  # Background

        # Apply fixed points AFTER grid creation, ensuring they map to exactly one cell
        print("Applying fixed points...")
        fixed_points_map = self.label_map.get("fixed_points", {})
        fixed_ids_map = self.label_map.get("fixed_point_ids", {})

        for label, points in fixed_points_map.items():
            point_ids = fixed_ids_map.get(label, [])
            if len(points) != len(point_ids):
                print(f"    Warning: Mismatch fixed points ({len(points)}) / IDs ({len(point_ids)}) for '{label}'. Skipping.")
                continue

            for i, (point, point_id) in enumerate(zip(points, point_ids)):
                if not (isinstance(point, (list, tuple)) and len(point) == 2): continue
                x, y = point
                if not (isinstance(point_id, int) and point_id >= 100): continue

                # Convert pixel coordinates to grid cell coordinates (ensure integers)
                grid_x = int(x / self.grid_size_pixels)
                grid_y = int(y / self.grid_size_pixels)

                # Ensure grid coordinates are within bounds
                if 0 <= grid_y < self.grid_rows and 0 <= grid_x < self.grid_cols:
                    # Overwrite the cell value with the fixed point ID
                    final_grid[grid_y, grid_x] = point_id
                    print(f"    Applied fixed point {point_id} ({x},{y}) to grid cell ({grid_y},{grid_x})")
                else:
                    print(f"    Warning: Fixed point {point_id} ({x},{y}) for '{label}' maps to grid cell ({grid_y},{grid_x}) which is out of bounds ({self.grid_rows}x{self.grid_cols}).")

        unique_values = np.unique(final_grid)
        print(f"Finished grid generation. Unique values: {unique_values}")
        return final_grid

    def visualize_grid(self, grid):
        """Create a visualization of the grid overlayed on the original image."""
        if self.image_array is None: return None # Cannot visualize without image
        if grid is None or grid.size == 0: return None # Cannot visualize empty grid

        img_height, img_width = self.image_array.shape[:2]
        grid_height, grid_width = grid.shape

        if grid_height <= 0 or grid_width <= 0: return None # Invalid grid dimensions

        # Calculate size of each grid cell in pixels (can be float)
        cell_height_px = img_height / grid_height
        cell_width_px = img_width / grid_width

        # Create full-size visualization array based on grid values
        vis_grid = np.zeros((img_height, img_width), dtype=grid.dtype)

        # Fill each pixel based on which grid cell it falls into
        for i in range(grid_height):
            for j in range(grid_width):
                y_start = int(round(i * cell_height_px))
                y_end = int(round((i + 1) * cell_height_px)) if i < grid_height - 1 else img_height
                x_start = int(round(j * cell_width_px))
                x_end = int(round((j + 1) * cell_width_px)) if j < grid_width - 1 else img_width

                # Handle potential rounding issues making start >= end
                if y_start >= y_end or x_start >= x_end: continue

                vis_grid[y_start:y_end, x_start:x_end] = grid[i, j]

        unique_values = np.unique(vis_grid)
        num_unique = len(unique_values)
        if num_unique <= 1 and 0 in unique_values: # Only background
             # Return original image with full alpha if it was RGB
             if self.image_array.ndim == 3 and self.image_array.shape[2] == 3:
                 return np.concatenate([self.image_array, np.full((img_height, img_width, 1), 255, dtype=np.uint8)], axis=2)
             # Otherwise return original (might already have alpha or be grayscale)
             return self.image_array


        # Generate colors using a suitable colormap (e.g., tab20 for distinct colors)
        try: cmap = plt.get_cmap('tab20', num_unique)
        except ValueError: cmap = plt.get_cmap('viridis', num_unique) # Fallback

        colors = cmap(np.linspace(0, 1, num_unique))
        # Ensure background (value 0) is transparent
        zero_idx = np.where(unique_values == 0)[0]
        if len(zero_idx) > 0: colors[zero_idx[0]] = [0, 0, 0, 0] # Set alpha to 0 for background

        # Map grid values to RGBA colors
        grid_rgba = np.zeros((img_height, img_width, 4), dtype=np.float32)
        value_to_color_idx = {val: i for i, val in enumerate(unique_values)}

        for value in unique_values:
            if value == 0: continue # Skip background
            color_idx = value_to_color_idx.get(value)
            if color_idx is not None:
                # Ensure color_idx is within bounds of generated colors
                grid_rgba[vis_grid == value] = colors[color_idx % len(colors)]

        # Prepare original image (float, 4-channel)
        original_float = self.image_array.astype(np.float32) / 255.0
        if original_float.ndim == 3 and original_float.shape[2] == 3:
             # Add alpha channel if original was RGB
             original_rgba = np.concatenate([original_float, np.ones((img_height, img_width, 1), dtype=np.float32)], axis=2)
        elif original_float.ndim == 2:
             # Handle grayscale: convert to RGB then add alpha
             original_rgb = cv2.cvtColor(original_float, cv2.COLOR_GRAY2RGB)
             original_rgba = np.concatenate([original_rgb, np.ones((img_height, img_width, 1), dtype=np.float32)], axis=2)
        else:
             # Assume it's already RGBA or some other format we pass through
             original_rgba = original_float


        # Alpha blending: Overlay * alpha + Background * (1 - alpha_overlay)
        alpha_overlay = grid_rgba[..., 3:4] # Keep dimension for broadcasting
        # Ensure original_rgba has 3 color channels for blending
        bg_rgb = original_rgba[..., :3] if original_rgba.shape[2] >= 3 else cv2.cvtColor(original_rgba, cv2.COLOR_GRAY2RGB)

        grid_vis_float = grid_rgba[..., :3] * alpha_overlay + bg_rgb * (1.0 - alpha_overlay)

        # Combine final alpha (use overlay alpha where > 0, else background alpha)
        bg_alpha = original_rgba[..., 3:4] if original_rgba.shape[2] == 4 else np.ones((img_height, img_width, 1), dtype=np.float32)
        final_alpha = np.where(alpha_overlay > 0, alpha_overlay, bg_alpha)
        grid_vis_float_rgba = np.concatenate([grid_vis_float, final_alpha], axis=2)

        # Convert back to uint8
        grid_vis_uint8 = (np.clip(grid_vis_float_rgba, 0, 1) * 255).astype(np.uint8)
        return grid_vis_uint8

    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit?", parent=self.root):
            self.root.destroy()

    def _load_camera_params(self, yaml_path="camera.yaml"):
        # Construct path relative to this script's directory
        script_dir = os.path.dirname(__file__)
        full_yaml_path = os.path.join(script_dir, yaml_path)

        print(f"[INFO] Loading camera parameters from: {full_yaml_path}")
        if not os.path.exists(full_yaml_path):
            print(f"[ERROR] Camera calibration file not found: {full_yaml_path}")
            # Handle error appropriately - maybe return None or default values
            return None
        # This function seems incomplete in the original file,
        # but it's not directly related to the SAM2 change.
        # Leaving it as is for now.


# --- Main Application Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    # Add style for checkbuttons to look like regular buttons (optional)
    style = ttk.Style()
    # Check if theme exists before configuring (avoid errors on some systems)
    try:
        style.theme_use('clam') # Or 'alt', 'default', 'classic'
    except tk.TclError:
        print("Warning: 'clam' theme not found, using default.")

    style.configure('Toolbutton', relief='raised', padding=6)
    # Configure selected state appearance
    style.map('Toolbutton',
              relief=[('selected', 'sunken'), ('!selected', 'raised')],
              background=[('selected', 'lightgray')]) # Example selected background

    app = SAM2SegmentationTool(root)
    root.mainloop()