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
from transformers import SamProcessor, SamModel

# Set environment variable for MPS fallback for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Select device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Use optimizations for CUDA
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
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
        self.root.title("SAM2 Segmentation Tool")
        self.root.geometry("1400x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure dataset directory
        self.dataset_dir = "dataset"
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
        # Initialize model variables (will be loaded when an image is opened)
        self.model = None
        self.processor = None
        self.image = None
        self.image_path = None
        self.image_array = None
        self.image_width = 0
        self.image_height = 0
        
        # Data variables
        self.points = []  # Format: [x, y]
        self.point_labels = []  # 0 for negative, 1 for positive
        self.boxes = []  # Format: [x1, y1, x2, y2]
        self.current_box = None  # Temporary box for drawing
        
        # Labels and masks
        self.labels = ["floor"]  # Default labels - removed door and wall as requested
        self.current_label = self.labels[0]
        self.label_map = {}  # Maps label name to integer value
        self.update_label_map()
        
        self.generated_masks = []
        self.selected_masks = {}
        self.mask_scores = []
        
        # UI Mode
        self.input_mode = "point_positive"  # "point_positive", "point_negative", "box", "fixed_point"
        
        # Fixed point ID counter - starts at 100 to avoid conflicts with label values
        self.fixed_point_id_counter = 100
        
        # Create UI
        self.create_ui()
        
        # Initialize with a welcome message
        self.update_status("Welcome! Please open an image to start.")
    
    def update_label_map(self):
        """Update the label map based on current labels"""
        # Preserve existing points data
        points_data = {}
        negative_points = []
        fixed_points_data = {}
        
        if "points" in self.label_map:
            points_data = self.label_map["points"]
        if "negative_points" in self.label_map:
            negative_points = self.label_map["negative_points"]
        if "fixed_points" in self.label_map:
            fixed_points_data = self.label_map["fixed_points"]
        
        # Update label values
        for i, label in enumerate(self.labels, start=1):  # Start from 1, 0 is background
            self.label_map[label] = i
        
        # Restore points data
        self.label_map["points"] = points_data
        for label in self.labels:
            if label not in self.label_map["points"]:
                self.label_map["points"][label] = []
        
        # Restore fixed points data
        self.label_map["fixed_points"] = fixed_points_data
        for label in self.labels:
            if label not in self.label_map["fixed_points"]:
                self.label_map["fixed_points"][label] = []
        
        # Restore negative points
        self.label_map["negative_points"] = negative_points
    
    def create_ui(self):
        """Create the UI components"""
        # Main frame layout
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into left (canvas) and right (controls) panels
        self.left_panel = ttk.Frame(self.main_frame)
        self.right_panel = ttk.Frame(self.main_frame, width=300)
        
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.right_panel.pack_propagate(False)
        
        # Canvas for image and drawing
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Image Canvas")
        self.ax.set_axis_off()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Right panel controls
        ttk.Label(self.right_panel, text="Controls", font=("Arial", 14, "bold")).pack(pady=10)
        
        # File operations frame
        file_frame = ttk.LabelFrame(self.right_panel, text="File Operations")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(file_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, padx=5, pady=5)
        
        # Input mode frame
        mode_frame = ttk.LabelFrame(self.right_panel, text="Input Mode")
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
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
        label_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.label_var = tk.StringVar(value=self.current_label)
        self.label_combobox = ttk.Combobox(label_frame, textvariable=self.label_var, 
                                           values=self.labels, state="readonly")
        self.label_combobox.pack(fill=tk.X, padx=5, pady=5)
        self.label_combobox.bind("<<ComboboxSelected>>", self.on_label_selected)
        
        # Add new label frame
        new_label_frame = ttk.Frame(label_frame)
        new_label_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.new_label_entry = ttk.Entry(new_label_frame)
        self.new_label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(new_label_frame, text="Add Label", command=self.add_new_label).pack(side=tk.RIGHT)
        
        # Actions frame
        actions_frame = ttk.LabelFrame(self.right_panel, text="Actions")
        actions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(actions_frame, text="Generate Masks", command=self.generate_masks).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(actions_frame, text="Generate & Save Grid", command=self.save_grid).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(actions_frame, text="Clear All", command=self.clear_all).pack(fill=tk.X, padx=5, pady=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.right_panel, text="Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Current selections frame
        selections_frame = ttk.LabelFrame(self.right_panel, text="Current Selections")
        selections_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.selections_text = tk.Text(selections_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.selections_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Keyboard shortcuts
        self.root.bind('<Key>', self.on_key_press)
        
        # Initial UI update
        self.update_ui()
    
    def update_ui(self):
        """Update the UI state based on current data"""
        # Update combobox values
        self.label_combobox['values'] = self.labels
        
        # Update selections text
        self.selections_text.config(state=tk.NORMAL)
        self.selections_text.delete('1.0', tk.END)
        
        # Add current label
        self.selections_text.insert(tk.END, f"Current Label: {self.current_label}\n\n")
        
        # Add points info
        if self.points:
            self.selections_text.insert(tk.END, f"Points ({len(self.points)}):\n")
            for i, (point, label) in enumerate(zip(self.points, self.point_labels)):
                point_type = "Positive" if label == 1 else "Negative"
                self.selections_text.insert(tk.END, f"  {i+1}. {point_type} at ({point[0]}, {point[1]})\n")
            self.selections_text.insert(tk.END, "\n")
        
        # Add fixed points info
        if "fixed_points" in self.label_map:
            total_fixed_points = 0
            for label in self.labels:
                if label in self.label_map["fixed_points"]:
                    points = self.label_map["fixed_points"][label]
                    if points:
                        total_fixed_points += len(points)
            
            if total_fixed_points > 0:
                self.selections_text.insert(tk.END, f"Fixed Points ({total_fixed_points}):\n")
                for label in self.labels:
                    if label in self.label_map["fixed_points"]:
                        points = self.label_map["fixed_points"][label]
                        if points:
                            self.selections_text.insert(tk.END, f"  Label '{label}' ({len(points)} points)\n")
                self.selections_text.insert(tk.END, "\n")
        
        # Add boxes info
        if self.boxes:
            self.selections_text.insert(tk.END, f"Boxes ({len(self.boxes)}):\n")
            for i, box in enumerate(self.boxes):
                self.selections_text.insert(tk.END, f"  {i+1}. Box at ({box[0]}, {box[1]}, {box[2]}, {box[3]})\n")
        
        self.selections_text.config(state=tk.DISABLED)
        
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
                    # Make sure mask is actually a numpy array
                    if isinstance(mask, np.ndarray):
                        self.show_mask(mask, random_color=True)
                    else:
                        print(f"Warning: Invalid mask for label {label}, type: {type(mask)}")
            
            # Draw points
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
            
            # Turn off axis for cleaner display
            self.ax.set_axis_off()
            
            # Update title
            mode_str = {
                "point_positive": "Positive Points (+)",
                "point_negative": "Negative Points (-)",
                "box": "Box",
                "fixed_point": "Fixed Points"
            }[self.input_mode]
            self.ax.set_title(f"Mode: {mode_str}, Label: {self.current_label}")
        
        self.canvas.draw()
    
    def show_mask(self, mask, random_color=False, alpha=0.6):
        """Show a mask on the current axes"""
        if mask is None or not isinstance(mask, np.ndarray):
            print(f"Error: Invalid mask type: {type(mask)}")
            return
            
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, alpha])
        
        try:
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8)
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            
            # Draw borders if possible
            try:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # Smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
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
        
        marker_size = 100  # Smaller than the matplotlib example since we're using a Tkinter canvas
        
        if len(pos_points) > 0:
            self.ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                           s=marker_size, edgecolor='white', linewidth=1.25)
        if len(neg_points) > 0:
            self.ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                           s=marker_size, edgecolor='white', linewidth=1.25)
    
    def show_box(self, box, label=None, color='green'):
        """Show a box on the current axes"""
        x0, y0, x1, y1 = box
        width = x1 - x0
        height = y1 - y0
        
        rect = patches.Rectangle((x0, y0), width, height, linewidth=2, 
                                 edgecolor=color, facecolor='none')
        self.ax.add_patch(rect)
        
        # Add label if provided
        if label:
            self.ax.text(x0 + width/2, y0 - 5, label, color=color, 
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
    def show_fixed_points(self):
        """Show fixed points on the canvas"""
        if "fixed_points" not in self.label_map:
            return
            
        # Using different colors for each label's fixed points
        # Get a color map with enough colors for all labels
        cmap = plt.cm.get_cmap('tab10', max(10, len(self.labels)))
        
        for i, label in enumerate(self.labels):
            if label in self.label_map["fixed_points"] and self.label_map["fixed_points"][label]:
                points = self.label_map["fixed_points"][label]
                if not points:
                    continue
                    
                points_array = np.array(points)
                
                # Calculate point IDs from fixed_point_ids for this label or assign default
                point_ids = []
                if "fixed_point_ids" in self.label_map and label in self.label_map["fixed_point_ids"]:
                    point_ids = self.label_map["fixed_point_ids"][label]
                
                # Generate color for this label's fixed points
                color = cmap(i % cmap.N)
                
                # Draw points
                self.ax.scatter(points_array[:, 0], points_array[:, 1], 
                               color=color, marker='o', s=80, edgecolor='white', linewidth=1.5)
                
                # Add ID label for each point
                for j, point in enumerate(points):
                    point_id = point_ids[j] if j < len(point_ids) else j + 100
                    self.ax.text(point[0], point[1] - 10, str(point_id), 
                                color=color, ha='center', va='bottom', fontsize=9, fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
    
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
        
        if not file_path:
            return
        
        try:
            # Load the image
            self.image = Image.open(file_path).convert("RGB")
            self.image_path = file_path
            self.image_array = np.array(self.image)
            self.image_height, self.image_width = self.image_array.shape[:2]
            
            # Clear previous data
            self.clear_all()
            
            # Initialize SAM2 model if not already initialized
            self.initialize_model()
            
            # Update UI
            self.update_status(f"Loaded image: {os.path.basename(file_path)}")
            self.redraw_canvas()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def initialize_model(self):
        """Initialize the SAM model and processor using Hugging Face Transformers"""
        if self.model is not None and self.processor is not None:
            return  # Already initialized

        # Reset model/processor state
        self.model = None
        self.processor = None

        try:
            self.update_status("Initializing SAM model from Hugging Face... This may take a moment.")

            # Load directly using Hugging Face model name
            hf_model_name = "facebook/sam-vit-large" # Using standard SAM ViT Large
            # Note: facebook/sam2-hiera-large might not have direct support in transformers yet
            # Let's use a known compatible model like facebook/sam-vit-large or facebook/sam-vit-huge

            self.update_status(f"Loading Processor {hf_model_name}...")
            # --- Use SamProcessor ---
            self.processor = SamProcessor.from_pretrained(hf_model_name)

            self.update_status(f"Loading Model {hf_model_name}...")
            # --- Use SamModel ---
            self.model = SamModel.from_pretrained(hf_model_name)
            self.model.to(device) # Move model to the selected device
            self.model.eval() # Set model to evaluation mode

            self.update_status(f"SAM model ({hf_model_name}) initialized successfully!")

        except ImportError:
             # This error should now only trigger if 'transformers' is missing
             messagebox.showerror("Error", "Failed to import SamProcessor or SamModel. Is the 'transformers' package installed correctly?")
             import traceback
             traceback.print_exc()
        except Exception as e:
            # Catch potential model loading errors (e.g., network, model name invalid)
            messagebox.showerror("Error", f"Failed to initialize SAM model from Hugging Face ({hf_model_name}): {str(e)}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
    
    def set_mode(self):
        """Set the current input mode"""
        self.input_mode = self.mode_var.get()
        self.update_status(f"Mode changed to {self.input_mode}")
        self.redraw_canvas()
    
    def on_label_selected(self, event):
        """Handle label selection from combobox"""
        self.current_label = self.label_var.get()
        self.update_status(f"Selected label: {self.current_label}")
        # Don't reset points here, just update UI
        self.update_ui()
    
    def add_new_label(self):
        """Add a new label to the list"""
        new_label = self.new_label_entry.get().strip()
        
        if not new_label:
            self.update_status("Label name cannot be empty.")
            return
            
        if new_label in self.labels:
            self.update_status(f"Label '{new_label}' already exists.")
            return
            
        # Add the new label
        self.labels.append(new_label)
        self.update_label_map()
        
        # Update UI
        self.current_label = new_label
        self.label_var.set(new_label)
        self.new_label_entry.delete(0, tk.END)
        self.update_status(f"Added new label: {new_label}")
        self.update_ui()
    
    def on_canvas_click(self, event):
        """Handle mouse clicks on the canvas"""
        if event.inaxes != self.ax or self.image_array is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.input_mode == "point_positive":
            # Add to current label's points
            if self.current_label not in self.label_map["points"]:
                 self.label_map["points"][self.current_label] = []
            self.label_map["points"][self.current_label].append([x, y])
            # Also add to the global list for display (compatibility)
            self.points.append([x, y])
            self.point_labels.append(1)
            self.update_status(f"Added positive point for '{self.current_label}' at ({x}, {y})")
        elif self.input_mode == "point_negative":
            # Add to global negative points
            if "negative_points" not in self.label_map:
                self.label_map["negative_points"] = []
            self.label_map["negative_points"].append([x, y])
            # Also add to the global list for display (compatibility)
            self.points.append([x, y])
            self.point_labels.append(0)
            self.update_status(f"Added negative point at ({x}, {y})")
        elif self.input_mode == "box":
            self.current_box = [x, y, x, y]  # Start drawing box
        elif self.input_mode == "fixed_point":
            # Add fixed point for the current label
            if self.current_label not in self.label_map["fixed_points"]:
                self.label_map["fixed_points"][self.current_label] = []
            if "fixed_point_ids" not in self.label_map:
                self.label_map["fixed_point_ids"] = {}
            if self.current_label not in self.label_map["fixed_point_ids"]:
                 self.label_map["fixed_point_ids"][self.current_label] = []

            point_id = self.fixed_point_id_counter
            self.fixed_point_id_counter += 1

            self.label_map["fixed_points"][self.current_label].append([x, y])
            self.label_map["fixed_point_ids"][self.current_label].append(point_id)
            self.update_status(f"Added fixed point ID {point_id} for '{self.current_label}' at ({x}, {y})")

        self.update_ui()
    
    def on_mouse_move(self, event):
        """Handle mouse movement on the canvas (for drawing box)"""
        if event.inaxes != self.ax or self.image_array is None:
            return

        if self.input_mode == "box" and self.current_box:
            x, y = int(event.xdata), int(event.ydata)
            self.current_box[2], self.current_box[3] = x, y
            self.redraw_canvas() # Redraw to show the box being dragged
    
    def on_canvas_release(self, event):
        """Handle mouse button release on the canvas"""
        if event.inaxes != self.ax or self.image_array is None:
            return

        if self.input_mode == "box" and self.current_box:
            x, y = int(event.xdata), int(event.ydata)
            # Ensure x1 < x2 and y1 < y2
            x1 = min(self.current_box[0], x)
            y1 = min(self.current_box[1], y)
            x2 = max(self.current_box[0], x)
            y2 = max(self.current_box[1], y)

            # Add box only if it has non-zero width and height
            if x1 < x2 and y1 < y2:
                self.boxes.append([x1, y1, x2, y2])
                self.update_status(f"Added box for '{self.current_label}' at ({x1}, {y1}, {x2}, {y2})")
            else:
                self.update_status("Box creation cancelled (zero size).")

            self.current_box = None  # Reset current box
            self.update_ui()
    
    def on_key_press(self, event):
        """Handle key presses for shortcuts"""
        if event.char == '1':
            self.mode_var.set("point_positive")
            self.set_mode()
        elif event.char == '2':
            self.mode_var.set("point_negative")
            self.set_mode()
        elif event.char == '3':
            self.mode_var.set("box")
            self.set_mode()
        elif event.char == '4':
            self.mode_var.set("fixed_point")
            self.set_mode()
        elif event.char == 'g':
            self.generate_masks()
        elif event.char == 's':
            self.save_grid()
        elif event.char == 'c':
            self.clear_all()
    
    def clear_all(self):
        """Clear all points, boxes, masks, and selections"""
        self.points = []
        self.point_labels = []
        self.boxes = []
        self.current_box = None
        self.label_masks = {}
        self.label_mask_scores = {}
        self.selected_masks = {}
        self.fixed_point_id_counter = 100 # Reset counter

        # Clear label map data but keep labels
        self.label_map["points"] = {label: [] for label in self.labels}
        self.label_map["negative_points"] = []
        self.label_map["fixed_points"] = {label: [] for label in self.labels}
        self.label_map["fixed_point_ids"] = {label: [] for label in self.labels}


        self.update_status("Cleared all inputs and generated masks.")
        self.update_ui()
    
    def generate_masks(self):
        """Generate masks using SAM model and processor based on prompts for each label"""
        if self.image_array is None:
            self.update_status("Please open an image first.")
            return

        # Check if model and processor are initialized
        if self.model is None or self.processor is None:
            self.update_status("Model not initialized. Trying to initialize...")
            self.initialize_model()
            if self.model is None or self.processor is None:
                self.update_status("Model initialization failed. Cannot generate masks.")
                return # Stop if initialization failed

        # Check if we have any positive points associated with any label
        has_positive_points = False
        if "points" in self.label_map:
            for label in self.labels:
                if label in self.label_map["points"] and self.label_map["points"][label]:
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
            labels_to_process.append(self.current_label)

        label_times = {}

        # Process each label that has positive points or boxes (if current label)
        for label in labels_to_process:
            start_time_label = time.time()
            positive_points_for_label = self.label_map.get("points", {}).get(label, [])
            boxes_for_label = self.boxes if label == self.current_label else [] # Only use boxes for the currently selected label

            # Combine all points (positive for this label, global negative)
            point_coords = positive_points_for_label + negative_points
            point_labels_list = [1] * len(positive_points_for_label) + [0] * len(negative_points)

            # Format for processor: list of lists for points/boxes per image
            input_points = [[point for point in point_coords]] if point_coords else None # e.g., [[[x1, y1], [x2, y2]]]
            input_labels = [[lbl for lbl in point_labels_list]] if point_coords else None # e.g., [[[1, 0]]]
            input_boxes = [[box for box in boxes_for_label]] if boxes_for_label else None # e.g., [[[x1,y1,x2,y2]]]

            # Skip if no prompts for this specific label combination
            if input_points is None and input_boxes is None:
                print(f"Skipping label '{label}' as it has no positive points or associated boxes.")
                continue

            print(f"Processing label '{label}'...")
            if input_points: print(f"  Points: {len(positive_points_for_label)} positive, {len(negative_points)} negative")
            if input_boxes: print(f"  Boxes: {len(boxes_for_label)}")

            try:
                # Preprocess using the processor
                # --- Use self.processor ---
                inputs = self.processor(
                    images=self.image, # Processor often prefers PIL image
                    input_points=input_points,
                    input_labels=input_labels,
                    input_boxes=input_boxes,
                    return_tensors="pt"
                ).to(device)

                # Predict using the model
                with torch.inference_mode():
                    # Use autocast based on the globally determined device
                    if device.type == 'cuda':
                        with torch.autocast(device.type, dtype=torch.bfloat16):
                             # --- Use self.model ---
                             outputs = self.model(**inputs, multimask_output=True)
                    else: # CPU or other devices might not use autocast
                        # --- Use self.model ---
                        outputs = self.model(**inputs, multimask_output=True)

                # --- START DEBUG PRINTS ---
                print(f"DEBUG: Label '{label}' - Raw model outputs:")
                print(f"DEBUG: outputs.pred_masks shape: {outputs.pred_masks.shape}") # Expect (1, num_masks, H, W)
                print(f"DEBUG: outputs.iou_scores shape: {outputs.iou_scores.shape}") # Expect (1, num_masks)
                # --- END DEBUG PRINTS ---

                # Post-process masks
                # --- Use self.processor ---
                original_image_size_hw = self.image_array.shape[:2]
                masks_tensor = self.processor.post_process_masks(
                    outputs.pred_masks.cpu(), # Get predicted masks from model output
                    inputs["original_sizes"].cpu(), # Original size tensor
                    inputs["reshaped_input_sizes"].cpu() # Reshaped size tensor
                )[0] # post_process_masks returns a list per image in batch, take the first

                # --- FIX: Squeeze the extra dimension (axis 0) from masks_tensor ---
                if masks_tensor.shape[0] == 1:
                    masks_tensor = masks_tensor.squeeze(0) # Shape becomes (num_masks, H, W)

                print(f"DEBUG: Label '{label}' - After post_process_masks and squeeze:") # Updated Debug Print
                print(f"DEBUG: masks_tensor shape: {masks_tensor.shape}") # Expect (num_masks, H, W)

                # Convert masks to numpy boolean
                masks = masks_tensor.numpy().astype(bool) # Shape should now be (num_masks, H, W)

                # Get scores
                # --- FIX: Squeeze the extra dimension (axis 0) from scores ---
                scores_tensor = outputs.iou_scores.cpu()[0] # Shape is (1, num_masks)
                if scores_tensor.shape[0] == 1:
                    scores_tensor = scores_tensor.squeeze(0) # Shape becomes (num_masks,)
                scores = scores_tensor.numpy() # Shape should now be (num_masks,)

                # Sort by score
                sorted_ind = np.argsort(scores)[::-1] # sorted_ind should now be 1D, e.g., [1 0 2]

                print(f"DEBUG: Label '{label}' - Before indexing error (after fixes):") # Updated Debug Print
                print(f"DEBUG: masks shape: {masks.shape}") # Expect (num_masks, H, W)
                print(f"DEBUG: scores shape: {scores.shape}") # Expect (num_masks,)
                print(f"DEBUG: sorted_ind: {sorted_ind}") # Expect 1D array

                # --- This line should now work correctly ---
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                # -----------------------------------------

                # Store for this label
                self.label_masks[label] = masks
                self.label_mask_scores[label] = scores

                # Calculate and store time for this label
                end_time_label = time.time()
                label_times[label] = end_time_label - start_time_label
                print(f"Generated {len(masks)} masks for label '{label}' in {label_times[label]:.2f} seconds")

            except Exception as label_e:
                print(f"Error processing label '{label}': {label_e}")
                import traceback
                traceback.print_exc()
                # Optionally continue to next label or stop

        # Clear any previous selected masks
        self.selected_masks = {}

        # Calculate total time and display timing information
        end_time_total = time.time()
        total_time = end_time_total - start_time_total

        if self.label_masks:
            # Update the display to show we have masks
            self.update_ui()

            # Show the multi-label mask selection dialog
            self.show_multi_label_mask_selection()

            # Create timing message
            timing_msg = f"Generated masks for {len(self.label_masks)} labels in {total_time:.2f} seconds.\n"
            for label, t in label_times.items():
                timing_msg += f"  - '{label}': {t:.2f}s for {len(self.label_masks[label])} masks\n"

            self.update_status(timing_msg)
            print(timing_msg)
        else:
            self.update_status(f"No masks were generated (process took {total_time:.2f} seconds)")
    
    def show_multi_label_mask_selection(self):
        """Show a dialog to select the best mask for each label."""
        if not self.label_masks:
            self.update_status("No masks generated to select from.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Masks for Each Label")
        dialog.geometry("1000x700")
        dialog.transient(self.root) # Keep dialog on top of main window
        dialog.grab_set() # Make dialog modal

        # Notebook for tabs
        notebook = ttk.Notebook(dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Store selected mask index for each label
        selected_indices = {}
        radio_vars = {}

        # Determine a consistent colormap for labels across tabs
        cmap = plt.cm.get_cmap('tab10', max(10, len(self.labels))) # Older, more compatible way
        # Ensure we have enough colors if more than 10 labels
        label_colors = [cmap(i % cmap.N) for i in range(len(self.labels))]
        label_color_map = {label: color for label, color in zip(self.labels, label_colors)}

        # Create a tab for each label that has generated masks
        for label, masks in self.label_masks.items():
            if not masks.any(): # Skip if no masks for this label
                continue

            tab_frame = ttk.Frame(notebook)
            notebook.add(tab_frame, text=f"{label} ({len(masks)} masks)")

            # Scrollable frame for masks
            canvas = tk.Canvas(tab_frame)
            scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Radio button variable for this label
            radio_vars[label] = tk.IntVar(value=-1) # Default to no selection

            # Display each mask with a radio button
            num_masks = len(masks)
            cols = 3 # Number of columns for mask previews
            rows = (num_masks + cols - 1) // cols

            # Get the color for this specific label
            label_color = label_color_map.get(label, cmap(0)) # Default color if label somehow missing

            for i, mask in enumerate(masks):
                row, col = divmod(i, cols)

                mask_frame = ttk.Frame(scrollable_frame, borderwidth=1, relief="solid")
                mask_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

                # Display the mask overlay
                try:
                    # Create overlay: original image + colored mask
                    mask_overlay = np.zeros_like(self.image_array, dtype=np.uint8)
                    mask_bool = mask.astype(bool) # Ensure boolean
                    # Apply label color where mask is true
                    mask_overlay[mask_bool] = (np.array(label_color[:3]) * 255).astype(np.uint8)

                    # Blend original image and mask overlay
                    alpha = 0.6
                    display_img_array = cv2.addWeighted(self.image_array, 1 - alpha, mask_overlay, alpha, 0)
                    display_img = Image.fromarray(display_img_array)

                    # Resize for display if too large
                    max_disp_size = (200, 200)
                    display_img.thumbnail(max_disp_size, Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(display_img)

                    img_label = ttk.Label(mask_frame, image=img_tk)
                    img_label.image = img_tk # Keep reference
                    img_label.pack(fill=tk.BOTH, expand=True)
                except Exception as display_e:
                    ttk.Label(mask_frame, text=f"Error displaying mask: {display_e}").pack(pady=10)

                # Radio button to select this mask
                rb = ttk.Radiobutton(mask_frame, text=f"Select Mask {i+1}", variable=radio_vars[label], value=i)
                rb.pack(pady=2)

        # Frame for Apply/Cancel buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        def apply_selections():
            self.selected_masks = {}
            all_selected = True
            for label in self.label_masks.keys():
                 if label in radio_vars:
                     selected_index = radio_vars[label].get()
                     if selected_index != -1:
                         self.selected_masks[label] = self.label_masks[label][selected_index]
                         print(f"Selected mask {selected_index} for label '{label}'")
                     else:
                         messagebox.showwarning("Selection Incomplete", f"Please select a mask for label '{label}'.", parent=dialog)
                         all_selected = False
                         break # Stop checking once one is missing
                 else:
                     # This case shouldn't happen if tabs were created correctly
                     print(f"Warning: No radio variable found for label '{label}'")


            if all_selected:
                dialog.destroy()
                self.update_status(f"Applied selected masks for {len(self.selected_masks)} labels.")
                self.update_ui() # Redraw main canvas with selected masks

        ttk.Button(button_frame, text="Apply Selections", command=apply_selections).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)

        # Show the dialog
        dialog.wait_window()
    
    def save_grid(self):
        """Generate and save the multi-label grid and associated data"""
        if not self.selected_masks:
            self.update_status("No masks selected. Please generate and select masks first.")
            return

        # Check if masks are selected for all labels that had generated masks
        missing_selection = False
        for label in self.label_masks.keys():
            if label not in self.selected_masks:
                missing_selection = True
                break
        if missing_selection:
             self.update_status("Not all generated labels have a selected mask. Please complete selection.")
             # Optionally, allow saving partial grid? For now, require all.
             # return

        try:
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.dataset_dir, timestamp)
            os.makedirs(save_dir, exist_ok=True)

            # Create subdirectories
            masks_dir = os.path.join(save_dir, "masks")
            grid_dir = os.path.join(save_dir, "grid")
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(grid_dir, exist_ok=True)

            # 1. Save original image
            original_image_path = os.path.join(save_dir, "original_image.png")
            self.image.save(original_image_path)
            print(f"Saved original image to {original_image_path}")

            # 2. Save label map (contains points, fixed points, label values)
            label_map_path = os.path.join(save_dir, "label_map.json")
            # Convert numpy arrays in label_map to lists for JSON serialization
            serializable_label_map = {}
            for key, value in self.label_map.items():
                if isinstance(value, dict):
                    serializable_label_map[key] = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in value.items()
                    }
                elif isinstance(value, np.ndarray):
                     serializable_label_map[key] = value.tolist()
                elif isinstance(value, list):
                     # Handle lists potentially containing numpy arrays (e.g., negative_points)
                     serializable_label_map[key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
                else:
                    serializable_label_map[key] = value

            with open(label_map_path, 'w') as f:
                json.dump(serializable_label_map, f, indent=4)
            print(f"Saved label map to {label_map_path}")

            # 3. Save selected masks (NPY and PNG)
            for label, mask in self.selected_masks.items():
                mask_npy_path = os.path.join(masks_dir, f"mask_{label}.npy")
                mask_png_path = os.path.join(masks_dir, f"mask_{label}.png")

                # Save raw mask as numpy array
                np.save(mask_npy_path, mask)

                # Save mask visualization (binary black/white)
                mask_vis = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_vis).save(mask_png_path)
                print(f"Saved mask for '{label}' to {mask_npy_path} and {mask_png_path}")

            # 4. Generate and save the multi-label grid
            labeled_grid = self.generate_multi_label_grid()
            grid_npy_path = os.path.join(grid_dir, "labeled_grid.npy")
            grid_png_path = os.path.join(grid_dir, "labeled_grid_vis.png")

            np.save(grid_npy_path, labeled_grid)
            print(f"Saved labeled grid (numpy) to {grid_npy_path}")

            # 5. Save grid visualization
            grid_vis_image = self.visualize_grid(labeled_grid)
            Image.fromarray(grid_vis_image).save(grid_png_path)
            print(f"Saved labeled grid visualization to {grid_png_path}")

            # 6. Save metadata (grid info, summary)
            grid_info = {
                "shape": labeled_grid.shape,
                "dtype": str(labeled_grid.dtype),
                "unique_values": np.unique(labeled_grid).tolist(),
                "label_map_reference": "label_map.json"
            }
            grid_info_path = os.path.join(grid_dir, "grid_info.json")
            with open(grid_info_path, 'w') as f:
                json.dump(grid_info, f, indent=4)
            print(f"Saved grid info to {grid_info_path}")

            summary = {
                "timestamp": timestamp,
                "original_image": os.path.basename(self.image_path),
                "saved_directory": save_dir,
                "labels_in_grid": list(self.selected_masks.keys()),
                "fixed_points_present": "fixed_points" in self.label_map and any(self.label_map["fixed_points"].values())
            }
            summary_path = os.path.join(save_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"Saved summary to {summary_path}")


            self.update_status(f"Successfully saved grid and data to:\n{save_dir}")

            # Show result dialog
            self.show_grid_result(grid_vis_image, save_dir)

        except Exception as e:
            self.update_status(f"Error saving grid: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Save Error", f"Failed to save grid data: {str(e)}")
    
    def show_grid_result(self, grid_vis_image, save_dir):
        """Show a dialog displaying the final grid visualization and save path."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Grid Generation Result")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(main_frame, text=f"Grid saved successfully to:\n{save_dir}", wraplength=550).pack(pady=10)

        # Display the grid visualization
        try:
            img_pil = Image.fromarray(grid_vis_image)
            # Resize for display if too large
            max_disp_size = (500, 400)
            img_pil.thumbnail(max_disp_size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)

            img_label = ttk.Label(main_frame, image=img_tk)
            img_label.image = img_tk # Keep reference
            img_label.pack(pady=10)
        except Exception as e:
            ttk.Label(main_frame, text=f"Error displaying image: {str(e)}").pack(pady=10)

        # Button to close dialog
        ttk.Button(main_frame, text="Close", command=dialog.destroy).pack(pady=10)

        # Show the dialog
        dialog.focus_set()
    
    def generate_multi_label_grid(self):
        """Generate a multi-label grid by combining selected masks from different labels"""
        height, width = self.image_array.shape[:2]

        # Create the labeled grid (same size as original image)
        labeled_grid = np.zeros((height, width), dtype=np.int32)

        print(f"Generating multi-label grid from {len(self.selected_masks)} selected masks")

        # Process each selected mask in order of label value (lower values first)
        for label, mask in sorted(self.selected_masks.items(),
                                 key=lambda x: self.label_map.get(x[0], 0)):
            # Get the value for this label
            label_value = self.label_map.get(label, 0)

            if label_value == 0:
                print(f"Skipping label '{label}' as it has value 0")
                continue

            print(f"Processing label '{label}' with value {label_value}")

            # Convert mask to boolean if needed
            if mask.ndim > 2:
                mask = np.squeeze(mask)  # Remove singleton dimensions

            if mask.dtype != bool:
                mask = mask > 0

            # Apply this mask to the grid with its label value
            # Overwrite any previous values (higher label values take precedence)
            labeled_grid[mask] = label_value

            print(f"Applied mask for label '{label}' with value {label_value}")

        # Add fixed points to the grid with their unique IDs
        if "fixed_points" in self.label_map and "fixed_point_ids" in self.label_map:
            for label in self.labels:
                if label in self.label_map["fixed_points"] and label in self.label_map["fixed_point_ids"]:
                    fixed_points = self.label_map["fixed_points"][label]
                    fixed_point_ids = self.label_map["fixed_point_ids"][label]

                    # Apply each fixed point to the grid
                    for i, (point, point_id) in enumerate(zip(fixed_points, fixed_point_ids)):
                        x, y = point

                        # Ensure the point is within image bounds
                        if 0 <= x < width and 0 <= y < height:
                            # Create a small region around the point (3x3 pixel square)
                            for dx in range(-1, 2):
                                for dy in range(-1, 2):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < width and 0 <= ny < height:
                                        # Only overwrite background or lower label values?
                                        # For now, overwrite unconditionally like labels
                                        labeled_grid[ny, nx] = point_id

                            print(f"Added fixed point at ({x}, {y}) with ID {point_id} to grid")

        # Print unique values in the final grid
        unique_values = np.unique(labeled_grid)
        print(f"Unique values in multi-label grid: {unique_values}")

        return labeled_grid
    
    def visualize_grid(self, grid):
        """Create a visualization of the grid by overlaying on the original image"""
        height, width = self.image_array.shape[:2]

        # Get number of unique values in grid
        unique_values = np.unique(grid)
        num_labels = len(unique_values)

        # Create a colormap for the labeled grid
        # Use a more diverse colormap like 'tab20' or 'viridis' if many labels/points
        colors = plt.cm.viridis(np.linspace(0, 1, max(num_labels, 10))) # Ensure enough colors
        colors[0] = [0, 0, 0, 0]  # Make background (label 0) transparent

        # Create an RGBA image from the labeled grid
        grid_rgba = np.zeros((height, width, 4), dtype=np.float32)

        # Map unique grid values to colors
        value_to_color_idx = {value: i for i, value in enumerate(unique_values)}

        for r in range(height):
            for c in range(width):
                 grid_value = grid[r, c]
                 if grid_value != 0: # Skip background
                     color_idx = value_to_color_idx.get(grid_value, 0) # Default to background if somehow missing
                     grid_rgba[r, c] = colors[color_idx % len(colors)] # Use modulo for safety


        # Combine with the original image for visualization
        alpha = 0.6  # Transparency of the overlay
        original_rgba = np.concatenate([self.image_array,
                                       np.ones((height, width, 1), dtype=np.uint8) * 255],
                                       axis=2) / 255.0

        grid_vis = original_rgba * (1 - alpha) + grid_rgba * alpha
        grid_vis = np.clip(grid_vis, 0, 1) # Ensure values are within [0, 1]
        grid_vis = (grid_vis * 255).astype(np.uint8)

        return grid_vis
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.root.destroy()

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = SAM2SegmentationTool(root)
    root.mainloop()