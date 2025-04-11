import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import traceback

class NPYVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("NPY File Visualizer")
        self.root.geometry("1000x700")
        
        # Data variables
        self.npy_data = None
        self.npy_path = None
        self.label_map = None
        self.original_image = None
        
        # Create UI
        self.create_ui()
        
        # Welcome message
        self.update_status("Welcome! Please load a .npy file to visualize.")
    
    def create_ui(self):
        """Create the UI components"""
        # Main frame layout
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into left (visualization) and right (controls) panels
        self.left_panel = ttk.Frame(self.main_frame)
        self.right_panel = ttk.Frame(self.main_frame, width=250)
        
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.right_panel.pack_propagate(False)
        
        # Canvas for visualization
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("NPY Data Visualization")
        self.ax.set_axis_off()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Right panel controls
        ttk.Label(self.right_panel, text="Controls", font=("Arial", 14, "bold")).pack(pady=10)
        
        # File operations frame
        file_frame = ttk.LabelFrame(self.right_panel, text="File Operations")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(file_frame, text="Load NPY File", command=self.load_npy).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(file_frame, text="Load Label Map (JSON)", command=self.load_label_map).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(file_frame, text="Load Original Image", command=self.load_image).pack(fill=tk.X, padx=5, pady=5)
        
        # Visualization options frame
        viz_frame = ttk.LabelFrame(self.right_panel, text="Visualization Options")
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Colormap selection
        ttk.Label(viz_frame, text="Colormap:").pack(anchor=tk.W, padx=5, pady=2)
        self.colormap_var = tk.StringVar(value="viridis")
        cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                 'tab10', 'tab20', 'Set1', 'Set2', 'Set3', 
                 'Pastel1', 'Pastel2', 'Paired', 'coolwarm']
        self.colormap_combo = ttk.Combobox(viz_frame, textvariable=self.colormap_var, values=cmaps)
        self.colormap_combo.pack(fill=tk.X, padx=5, pady=2)
        self.colormap_combo.bind("<<ComboboxSelected>>", self.update_visualization)
        
        # Alpha (transparency) slider
        ttk.Label(viz_frame, text="Transparency:").pack(anchor=tk.W, padx=5, pady=2)
        self.alpha_var = tk.DoubleVar(value=0.7)
        alpha_slider = ttk.Scale(viz_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                 variable=self.alpha_var, command=self.update_transparency)
        alpha_slider.pack(fill=tk.X, padx=5, pady=2)
        
        # Display options
        self.show_values_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(viz_frame, text="Show values on grid", variable=self.show_values_var, 
                      command=self.update_visualization).pack(anchor=tk.W, padx=5, pady=5)
        
        self.show_original_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_frame, text="Show original image", variable=self.show_original_var, 
                      command=self.update_visualization).pack(anchor=tk.W, padx=5, pady=5)
        
        # Information frame
        info_frame = ttk.LabelFrame(self.right_panel, text="File Information")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.right_panel, text="Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=3, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Export buttons
        export_frame = ttk.Frame(self.right_panel)
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(export_frame, text="Export as PNG", command=self.export_visualization).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export with Legend", command=self.export_with_legend).pack(fill=tk.X, pady=2)
    
    def load_npy(self):
        """Load a NumPy .npy file"""
        file_path = filedialog.askopenfilename(
            title="Select NPY File",
            filetypes=[("NumPy Files", "*.npy")]
        )
        
        if not file_path:
            return
        
        try:
            # Load the NumPy file
            self.npy_data = np.load(file_path)
            self.npy_path = file_path
            
            # Update information
            self.update_info()
            
            # Update visualization
            self.update_visualization()
            
            self.update_status(f"Loaded NPY file: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load NPY file: {str(e)}")
    
    def load_label_map(self):
        """Load a label map JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select Label Map JSON",
            filetypes=[("JSON Files", "*.json")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                self.label_map = json.load(f)
            
            self.update_info()
            self.update_visualization()
            
            self.update_status(f"Loaded label map: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load label map: {str(e)}")
    
    def load_image(self):
        """Load the original image for overlay"""
        file_path = filedialog.askopenfilename(
            title="Select Original Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif")]
        )
        
        if not file_path:
            return
        
        try:
            self.original_image = plt.imread(file_path)
            self.update_visualization()
            self.update_status(f"Loaded original image: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def update_transparency(self, event=None):
        """Update visualization when transparency is changed"""
        self.update_visualization()
    
    def update_info(self):
        """Update the information text with details about the loaded data"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        
        if self.npy_data is not None:
            # Add NPY file information
            self.info_text.insert(tk.END, f"NPY File: {os.path.basename(self.npy_path)}\n")
            self.info_text.insert(tk.END, f"Shape: {self.npy_data.shape}\n")
            self.info_text.insert(tk.END, f"Data Type: {self.npy_data.dtype}\n")
            
            # Add min/max values
            min_val = np.min(self.npy_data)
            max_val = np.max(self.npy_data)
            self.info_text.insert(tk.END, f"Min Value: {min_val}\n")
            self.info_text.insert(tk.END, f"Max Value: {max_val}\n")
            
            # Add unique values information
            unique_vals = np.unique(self.npy_data)
            self.info_text.insert(tk.END, f"Unique Values: {len(unique_vals)}\n")
            
            # If we have a label map, use it to show label names
            if self.label_map:
                self.info_text.insert(tk.END, "\nLabels:\n")
                # Create a reverse mapping from value to name
                value_to_name = {v: k for k, v in self.label_map.items() 
                               if isinstance(v, (int, float)) and k != "points"}
                
                for val in unique_vals:
                    if val == 0:
                        self.info_text.insert(tk.END, f"  {val}: Background\n")
                    elif val in value_to_name:
                        self.info_text.insert(tk.END, f"  {val}: {value_to_name[val]}\n")
                    else:
                        self.info_text.insert(tk.END, f"  {val}: Unknown\n")
        
        self.info_text.config(state=tk.DISABLED)
    
    def update_visualization(self, event=None):
        """Update the visualization based on current settings"""
        if self.npy_data is None:
            return
        
        # Clear the plot
        self.ax.clear()
        
        # Show original image if available and option is selected
        if self.original_image is not None and self.show_original_var.get():
            self.ax.imshow(self.original_image)
        
        # Get the colormap
        cmap = self.colormap_var.get()
        alpha = self.alpha_var.get()
        
        # Special handling for labeled data
        if np.issubdtype(self.npy_data.dtype, np.integer) or len(np.unique(self.npy_data)) <= 30:
            # For integer data or data with few unique values, treat as labels
            unique_vals = np.unique(self.npy_data)
            
            # Create a mask for each label value and show with different colors
            if self.original_image is None or not self.show_original_var.get():
                # If no original image, show a black background first
                h, w = self.npy_data.shape[:2]
                self.ax.imshow(np.zeros((h, w, 3)), cmap='gray')
            
            # Create legend handles
            legend_handles = []
            legend_labels = []
            
            # Generate a deterministic color list for unique values
            # This ensures consistent colors for values across visualizations
            num_unique = len(unique_vals)
            
            # Use a fixed set of distinct colors for values above 0
            # Create a specialized colormap with high contrast between adjacent values
            distinct_colors = [
                # First color is white/transparent for background (0)
                [1, 1, 1, 0],
                # Distinct colors for values >= 1
                [0.8, 0.1, 0.1, alpha],  # Red
                [0.1, 0.6, 0.1, alpha],  # Green
                [0.1, 0.2, 0.8, alpha],  # Blue
                [0.9, 0.6, 0.1, alpha],  # Orange
                [0.7, 0.1, 0.7, alpha],  # Purple
                [0.1, 0.7, 0.7, alpha],  # Teal
                [0.8, 0.7, 0.1, alpha],  # Gold
                [0.6, 0.3, 0.2, alpha],  # Brown
                [0.9, 0.2, 0.5, alpha],  # Pink
                [0.5, 0.5, 0.5, alpha],  # Gray
            ]
            
            # If we have more unique values than predefined colors, generate more colors
            if num_unique > len(distinct_colors):
                # Use the colormap to generate additional colors
                additional_colors = plt.get_cmap(cmap)(
                    np.linspace(0, 1, num_unique - len(distinct_colors) + 1)
                )
                # Add alpha channel
                additional_colors = np.column_stack([
                    additional_colors[:, :3], 
                    np.ones(num_unique - len(distinct_colors) + 1) * alpha
                ])
                # Combine predefined colors with additional colors
                color_map = np.vstack([distinct_colors, additional_colors])
            else:
                # Use just the predefined colors
                color_map = np.array(distinct_colors[:num_unique])
            
            # Plot each value with its own color
            for i, val in enumerate(unique_vals):
                # Create mask for this value
                mask = self.npy_data == val
                
                # Skip empty masks
                if not np.any(mask):
                    continue
                
                # Get color for this value
                if val == 0:  # Background
                    color = color_map[0]
                else:
                    # Map to appropriate color, ensuring values > 0 get distinct colors
                    color_idx = min(int(val), len(color_map) - 1)
                    color = color_map[color_idx]
                
                # Apply transparency
                color[-1] = alpha
                
                # Create RGBA mask
                mask_rgba = np.zeros((*mask.shape, 4))
                mask_rgba[mask] = color
                
                # Show mask
                self.ax.imshow(mask_rgba)
                
                # Get label name for this value
                label_name = str(val)
                if self.label_map:
                    # Find the label name for this value
                    value_to_name = {v: k for k, v in self.label_map.items() 
                                   if isinstance(v, (int, float)) and k != "points"}
                    if val in value_to_name:
                        label_name = value_to_name[val]
                
                # Add to legend (skip background if it would make the legend too cluttered)
                if val != 0 or len(unique_vals) <= 5:
                    patch = plt.Rectangle((0, 0), 1, 1, color=color)
                    legend_handles.append(patch)
                    if val == 0:
                        legend_labels.append(f"Background (0)")
                    else:
                        legend_labels.append(f"{label_name} ({val})")
                
                # Add label text in the center of each region if enabled
                if self.show_values_var.get() and np.any(mask) and val != 0:
                    # Find the center of this label region
                    y_indices, x_indices = np.where(mask)
                    center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
                    
                    # Add text with contrasting color for visibility
                    text_color = 'white' if np.mean(color[:3]) < 0.6 else 'black'
                    self.ax.text(center_x, center_y, label_name, color=text_color, fontsize=12, 
                               ha='center', va='center', fontweight='bold',
                               bbox=dict(facecolor=color[:3], alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Add legend if we have more than one non-zero value
            if legend_handles:
                # Determine legend position: outside the plot if we have many values, inside if just a few
                if len(legend_handles) > 5:
                    # Place legend outside the plot
                    legend = self.ax.legend(legend_handles, legend_labels, 
                                       loc='center left', bbox_to_anchor=(1, 0.5),
                                       framealpha=0.8, title="Labels")
                else:
                    # Place legend inside the plot, in the upper right
                    legend = self.ax.legend(legend_handles, legend_labels, 
                                       loc='upper right', framealpha=0.7, title="Labels")
                
                # Make legend title bold
                legend.get_title().set_fontweight('bold')
        else:
            # For floating point data, treat as a heatmap
            masked_data = np.ma.masked_equal(self.npy_data, 0)  # Mask zeros if any
            im = self.ax.imshow(masked_data, cmap=cmap, alpha=alpha)
            cbar = plt.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            cbar.set_label('Value')
        
        # Add grid values summary to title
        unique_vals = np.unique(self.npy_data)
        unique_text = f"Values: {', '.join([str(v) for v in unique_vals])}"
        title = f"NPY Data Visualization - {os.path.basename(self.npy_path)} - {unique_text}"
        self.ax.set_title(title, fontsize=10)
        self.ax.set_axis_off()
        
        # Redraw the canvas
        self.canvas.draw()
    
    def export_visualization(self):
        """Export the current visualization as a PNG file"""
        if self.npy_data is None:
            messagebox.showinfo("Export", "Nothing to export. Please load a file first.")
            return
        
        # Ask for the save path
        default_name = os.path.splitext(os.path.basename(self.npy_path))[0] + "_visualization.png"
        save_path = filedialog.asksaveasfilename(
            title="Save Visualization",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            self.fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            self.update_status(f"Saved visualization to {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save visualization: {str(e)}")
    
    def export_with_legend(self):
        """Export visualization with a comprehensive legend"""
        if self.npy_data is None:
            messagebox.showinfo("Export", "Nothing to export. Please load a file first.")
            return
        
        # Ask for the save path
        default_name = os.path.splitext(os.path.basename(self.npy_path))[0] + "_with_legend.png"
        save_path = filedialog.asksaveasfilename(
            title="Save Visualization with Legend",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # Create a new figure with extra space for the legend
            fig, (ax_img, ax_legend) = plt.subplots(1, 2, figsize=(12, 8), 
                                                   gridspec_kw={'width_ratios': [3, 1]})
            
            # Plot image in the left subplot
            if self.original_image is not None and self.show_original_var.get():
                ax_img.imshow(self.original_image)
            
            # Get the colormap
            cmap = self.colormap_var.get()
            alpha = self.alpha_var.get()
            
            # Special handling for labeled data
            if np.issubdtype(self.npy_data.dtype, np.integer) or len(np.unique(self.npy_data)) <= 30:
                # For integer data or data with few unique values, treat as labels
                unique_vals = np.unique(self.npy_data)
                
                # Create a mask for each label value and show with different colors
                if self.original_image is None or not self.show_original_var.get():
                    # If no original image, show a black background first
                    h, w = self.npy_data.shape[:2]
                    ax_img.imshow(np.zeros((h, w, 3)), cmap='gray')
                
                # Generate a deterministic color list for unique values
                # This ensures consistent colors for values across visualizations
                num_unique = len(unique_vals)
                
                # Use a fixed set of distinct colors for values above 0
                distinct_colors = [
                    # First color is white/transparent for background (0)
                    [1, 1, 1, 0],
                    # Distinct colors for values >= 1
                    [0.8, 0.1, 0.1, alpha],  # Red
                    [0.1, 0.6, 0.1, alpha],  # Green
                    [0.1, 0.2, 0.8, alpha],  # Blue
                    [0.9, 0.6, 0.1, alpha],  # Orange
                    [0.7, 0.1, 0.7, alpha],  # Purple
                    [0.1, 0.7, 0.7, alpha],  # Teal
                    [0.8, 0.7, 0.1, alpha],  # Gold
                    [0.6, 0.3, 0.2, alpha],  # Brown
                    [0.9, 0.2, 0.5, alpha],  # Pink
                    [0.5, 0.5, 0.5, alpha],  # Gray
                ]
                
                # If we have more unique values than predefined colors, generate more colors
                if num_unique > len(distinct_colors):
                    # Use the colormap to generate additional colors
                    additional_colors = plt.get_cmap(cmap)(
                        np.linspace(0, 1, num_unique - len(distinct_colors) + 1)
                    )
                    # Add alpha channel
                    additional_colors = np.column_stack([
                        additional_colors[:, :3], 
                        np.ones(num_unique - len(distinct_colors) + 1) * alpha
                    ])
                    # Combine predefined colors with additional colors
                    color_map = np.vstack([distinct_colors, additional_colors])
                else:
                    # Use just the predefined colors
                    color_map = np.array(distinct_colors[:num_unique])
                
                # Plot each value with its own color
                for i, val in enumerate(unique_vals):
                    # Create mask for this value
                    mask = self.npy_data == val
                    
                    # Skip empty masks
                    if not np.any(mask):
                        continue
                    
                    # Get color for this value
                    if val == 0:  # Background
                        color = color_map[0]
                    else:
                        # Map to appropriate color, ensuring values > 0 get distinct colors
                        color_idx = min(int(val), len(color_map) - 1)
                        color = color_map[color_idx]
                    
                    # Apply transparency
                    color[-1] = alpha
                    
                    # Create RGBA mask
                    mask_rgba = np.zeros((*mask.shape, 4))
                    mask_rgba[mask] = color
                    
                    # Show mask
                    ax_img.imshow(mask_rgba)
                    
                    # Add label text in the center of each region if enabled
                    if self.show_values_var.get() and np.any(mask) and val != 0:
                        # Find the center of this label region
                        y_indices, x_indices = np.where(mask)
                        center_y, center_x = int(np.mean(y_indices)), int(np.mean(x_indices))
                        
                        # Get the label name if available
                        label_name = str(val)
                        if self.label_map:
                            # Find the label name for this value
                            value_to_name = {v: k for k, v in self.label_map.items() 
                                           if isinstance(v, (int, float)) and k != "points"}
                            if val in value_to_name:
                                label_name = value_to_name[val]
                        
                        # Add text with contrasting color for visibility
                        text_color = 'white' if np.mean(color[:3]) < 0.6 else 'black'
                        ax_img.text(center_x, center_y, label_name, color=text_color, fontsize=12, 
                                  ha='center', va='center', fontweight='bold',
                                  bbox=dict(facecolor=color[:3], alpha=0.7, boxstyle='round,pad=0.3'))
                
                # Create detailed legend in the right subplot
                ax_legend.axis('off')
                ax_legend.set_title("Legend - Label Information", fontweight='bold')
                
                # Add each label as a colored box with text
                y_pos = 0.9
                y_step = 0.9 / (len(unique_vals) + 1)
                
                # Function to get label name
                def get_label_name(val):
                    if self.label_map:
                        value_to_name = {v: k for k, v in self.label_map.items() 
                                       if isinstance(v, (int, float)) and k != "points"}
                        if val in value_to_name:
                            return value_to_name[val]
                    return str(val)
                
                for i, val in enumerate(unique_vals):
                    # Skip empty classes
                    if not np.any(self.npy_data == val):
                        continue
                        
                    if val == 0:
                        label_text = "Background (Value: 0)"
                        color = (1, 1, 1)  # White for background
                    else:
                        # Get the label name if available
                        label_name = get_label_name(val)
                        label_text = f"{label_name} (Value: {val})"
                        
                        # Get the same color used in the visualization
                        if val == 0:  # Background
                            color = color_map[0][:3]
                        else:
                            # Map to appropriate color, ensuring values > 0 get distinct colors
                            color_idx = min(int(val), len(color_map) - 1)
                            color = color_map[color_idx][:3]
                    
                    # Find how many pixels have this value
                    pixel_count = np.sum(self.npy_data == val)
                    percentage = (pixel_count / self.npy_data.size) * 100
                    label_text += f"\nPixels: {pixel_count:,} ({percentage:.1f}%)"
                    
                    # Add colored rectangle
                    rect = plt.Rectangle((0.1, y_pos-y_step), 0.1, y_step*0.8, 
                                        facecolor=color, alpha=1.0 if val == 0 else alpha,
                                        edgecolor='black', linewidth=1)
                    ax_legend.add_patch(rect)
                    
                    # Add label text
                    ax_legend.text(0.25, y_pos-y_step/2, label_text, 
                                 va='center', ha='left', fontsize=10)
                    
                    y_pos -= y_step
                
                # Add file information
                ax_legend.text(0.1, 0.05, f"File: {os.path.basename(self.npy_path)}\n"
                             f"Shape: {self.npy_data.shape}\n"
                             f"Unique Values: {len(unique_vals)}\n"
                             f"Values: {', '.join([str(v) for v in unique_vals])}\n"
                             f"Colormap: {cmap}",
                             va='bottom', ha='left', fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
            else:
                # For floating point data, treat as a heatmap
                masked_data = np.ma.masked_equal(self.npy_data, 0)  # Mask zeros if any
                im = ax_img.imshow(masked_data, cmap=cmap, alpha=alpha)
                plt.colorbar(im, ax=ax_img)
                
                # Clear legend panel
                ax_legend.axis('off')
                ax_legend.text(0.5, 0.5, "Continuous Data\n(See colorbar)", 
                             ha='center', va='center', fontsize=12,
                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
            
            # Add grid values summary to title
            unique_vals = np.unique(self.npy_data)
            unique_text = f"Values: {', '.join([str(v) for v in unique_vals])}"
            title = f"NPY Data Visualization - {os.path.basename(self.npy_path)} - {unique_text}"
            ax_img.set_title(title, fontsize=10)
            ax_img.set_axis_off()
            
            # Save the figure
            plt.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.update_status(f"Saved visualization with legend to {os.path.basename(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save visualization: {str(e)}")
            traceback.print_exc()
    
    def update_status(self, message):
        """Update the status text with a new message"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete('1.0', tk.END)
        self.status_text.insert(tk.END, message)
        self.status_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = NPYVisualizer(root)
    root.mainloop() 