import os
import sys
import numpy as np
import torch
import cv2
import json
import time
import math
from PIL import Image

# --- Remove sys.path manipulation if sam2 is installed via pip ---
# script_dir = os.path.dirname(__file__)
# sam_base_dir = os.path.abspath(os.path.join(script_dir, '..')) # Example if needed
# sys.path.insert(0, sam_base_dir)

try:
    # from sam2.build_sam import build_sam2 # Removed old import
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2 modules: {e}")
    print("Please ensure the 'sam2' library (from Meta) is installed correctly (e.g., pip install git+https://github.com/facebookresearch/segment-anything-2.git).")
    sys.exit(1)

# --- Device Selection (copied from initial.py) ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Use optimizations for CUDA if available
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Support for MPS devices is preliminary. Results may vary.")
else:
    device = torch.device("cpu")
print(f"SAMCallbackService using device: {device}")

class SAMCallbackService:
    def __init__(self,
                 hf_model_name="facebook/sam2-hiera-tiny", # Added HF model name arg
                 working_dir="../working_dataset", # Default relative to CV_app/
                 calibration_json_path="calibration.json", # Default relative to CV_app/
                 original_grid_name="labeled_grid.npy",
                 labeled_grid_name="labeled_grid.npy", # Keep for loading initial prompts
                 floor_label="floor"):
        """
        Initializes the SAM2 predictor from Hugging Face and loads necessary data.

        Args:
            hf_model_name (str): Hugging Face model identifier (e.g., "facebook/sam2-hiera-tiny").
            working_dir (str): Path to the working_dataset directory (relative to script dir).
            calibration_json_path (str): Path to the calibration.json file (relative to script dir).
            original_grid_name (str): Filename of the original grid numpy file (within working_dir/grid).
            labeled_grid_name (str): Filename of the labeled grid numpy file (within working_dir/grid).
            floor_label (str): The label string used for the floor in label_map.json.
        """
        self.initialized = False
        self.predictor = None
        # self.sam2_model = None # Removed old model variable
        self.hf_model_name = hf_model_name
        self.pixels_per_cm = None
        self.grid_cols = None
        self.grid_rows = None
        self.grid_size_cm = None
        self.initial_floor_points = None
        self.original_grid = None
        self.floor_label_value = None
        self.floor_label = floor_label

        # Get the directory of this script (e.g., CV_app/)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # Path to working dataset (e.g., ../working_dataset relative to CV_app/)
        self.working_dir = os.path.abspath(os.path.join(self.script_dir, working_dir))
        self.device = device

        # Construct absolute paths
        abs_original_grid_path = os.path.join(self.working_dir, "grid", original_grid_name)
        abs_labeled_grid_path = os.path.join(self.working_dir, "grid", labeled_grid_name) # Used for initial prompts
        abs_label_map_path = os.path.join(self.working_dir, "label_map.json")
        abs_calib_json_path = os.path.join(self.script_dir, calibration_json_path)

        print("Initializing SAMCallbackService...")
        print(f"Script directory: {self.script_dir}")
        print(f"Working directory: {self.working_dir}")
        print(f"Calibration JSON path: {abs_calib_json_path}")
        print(f"Original grid path: {abs_original_grid_path}")
        print(f"Labeled grid path (for prompts): {abs_labeled_grid_path}")
        print(f"Label map path: {abs_label_map_path}")

        try:
            # 1. Load Label Map (needed for floor label value)
            print(f"Loading label map from {abs_label_map_path}...")
            self._load_label_map(abs_label_map_path)
            if self.floor_label_value is None:
                raise ValueError(f"Could not find label value for '{self.floor_label}' in label_map.json.")
            print(f"Loaded floor label value: {self.floor_label_value}")

            # 2. Load Original Grid (used as base for updates)
            print(f"Loading original grid from {abs_original_grid_path}...")
            self._load_original_grid(abs_original_grid_path)
            if self.original_grid is None:
                # Try loading the labeled grid path as fallback if original name differs
                print(f"[WARN] Original grid '{original_grid_name}' not found or failed. Trying '{labeled_grid_name}'...")
                self._load_original_grid(abs_labeled_grid_path)
                if self.original_grid is None:
                    raise FileNotFoundError("Base grid file (original or labeled) not found or failed to load.")
            print(f"Base grid loaded. Shape: {self.original_grid.shape}")

            # 3. Load Calibration Data
            print(f"Loading calibration data from {abs_calib_json_path}...")
            self.pixels_per_cm, self.grid_cols, self.grid_rows, self.grid_size_cm = self._load_calibration_json(abs_calib_json_path)
            if self.pixels_per_cm is None:
                print("[WARN] Calibration data not loaded. Grid-related operations might fail.")
            else:
                print(f"[INFO] Calibration loaded: {self.pixels_per_cm:.2f} px/cm, Grid: {self.grid_cols}x{self.grid_rows}, Size: {self.grid_size_cm} cm")
                # Verify grid dimensions match loaded grid if possible
                if self.original_grid is not None and (self.original_grid.shape[0] != self.grid_rows or self.original_grid.shape[1] != self.grid_cols):
                    print(f"[WARN] Calibration grid dimensions ({self.grid_rows}x{self.grid_cols}) mismatch loaded grid ({self.original_grid.shape}). Using loaded grid dimensions.")
                    self.grid_rows, self.grid_cols = self.original_grid.shape

            # 4. Load Initial Prompts (Floor Points from label_map.json)
            print(f"Loading initial prompts from label map...")
            self._load_initial_prompts_from_label_map(abs_label_map_path) # Changed source
            if self.initial_floor_points is None or len(self.initial_floor_points) == 0:
                # Allow initialization without initial points, but warn
                print(f"[WARN] Could not find positive points for label '{self.floor_label}' in label_map.json. Initial prediction might be less accurate.")
                self.initial_floor_points = np.empty((0, 2), dtype=np.int32) # Use empty array
            else:
                print(f"Loaded {len(self.initial_floor_points)} initial positive points for '{self.floor_label}'.")

            # 5. Load SAM2 Predictor from Hugging Face
            print(f"Loading SAM2 predictor '{self.hf_model_name}'...")
            self.predictor = SAM2ImagePredictor.from_pretrained(self.hf_model_name)
            # Predictor should automatically use the selected `device` if configured correctly during install/pytorch setup
            # self.predictor.model.to(self.device) # Explicit move if needed

            print("SAM2 predictor loaded successfully.")

            self.initialized = True
            print("SAMCallbackService initialized successfully.")

        except FileNotFoundError as e:
            print(f"[ERROR] Initialization failed: Required file not found - {e}")
        except ValueError as e:
            print(f"[ERROR] Initialization failed: Invalid data - {e}")
        except ImportError as e:
            # Error already printed during import attempt
             print(f"[ERROR] Initialization failed due to Import Error.")
        except Exception as e:
            print(f"[ERROR] Unexpected error during initialization: {e}")
            import traceback
            traceback.print_exc()

    def _load_label_map(self, path):
        """Loads label map and extracts floor label value."""
        try:
            with open(path, 'r') as f:
                label_map_data = json.load(f)
            # Find the integer value for the floor label
            self.floor_label_value = label_map_data.get(self.floor_label)
            # Store the loaded map if needed for other things later
            self.label_map = label_map_data
        except FileNotFoundError:
            print(f"[ERROR] Label map file not found at {path}")
            self.label_map = None
            self.floor_label_value = None
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to decode JSON from {path}")
            self.label_map = None
            self.floor_label_value = None
        except Exception as e:
            print(f"[ERROR] Failed to load label map from {path}: {e}")
            self.label_map = None
            self.floor_label_value = None

    def _load_original_grid(self, path):
        """Loads the base grid numpy file."""
        try:
            if os.path.exists(path):
                self.original_grid = np.load(path)
            else:
                 print(f"[ERROR] Grid file not found at {path}")
                 self.original_grid = None
        except Exception as e:
            print(f"[ERROR] Failed to load grid from {path}: {e}")
            self.original_grid = None

    def _load_calibration_json(self, path):
        """Loads calibration data from the specific calibration.json structure."""
        pixels_per_cm = None
        grid_cols = None
        grid_rows = None
        grid_size_cm = None
        try:
            with open(path, 'r') as f:
                calib_data = json.load(f)

            # --- Access values based on the provided JSON structure ---
            if 'pixel_ratio' in calib_data and isinstance(calib_data['pixel_ratio'], dict):
                pixels_per_cm = calib_data['pixel_ratio'].get('value')

            if 'grid' in calib_data and isinstance(calib_data['grid'], dict):
                grid_cols = calib_data['grid'].get('columns')
                grid_rows = calib_data['grid'].get('rows')
                # Use 'cell_size_cm' key
                grid_size_cm = calib_data['grid'].get('cell_size_cm')
            # ---------------------------------------------------------

            # Perform validation and provide specific warnings
            valid = True
            missing_or_invalid = []
            if not isinstance(pixels_per_cm, (int, float)) or pixels_per_cm <= 0:
                missing_or_invalid.append("pixel_ratio.value")
                valid = False
            if not isinstance(grid_cols, int) or grid_cols <= 0:
                missing_or_invalid.append("grid.columns")
                valid = False
            if not isinstance(grid_rows, int) or grid_rows <= 0:
                missing_or_invalid.append("grid.rows")
                valid = False
            # Check grid.cell_size_cm
            if not isinstance(grid_size_cm, (int, float)) or grid_size_cm <= 0:
                missing_or_invalid.append("grid.cell_size_cm")
                valid = False

            if not valid:
                 print(f"[WARN] Invalid or missing values in {path}: {', '.join(missing_or_invalid)}")
                 # Return None for all if any are invalid
                 return None, None, None, None

            # If all valid, return the values
            return pixels_per_cm, grid_cols, grid_rows, grid_size_cm

        except FileNotFoundError:
            print(f"[WARN] Calibration file not found at {path}")
            return None, None, None, None
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to decode JSON from {path}")
            return None, None, None, None
        except KeyError as e:
             print(f"[ERROR] Missing expected key in calibration data {path}: {e}")
             return None, None, None, None
        except Exception as e:
            print(f"[ERROR] Failed to load calibration data from {path}: {e}")
            return None, None, None, None

    def _load_initial_prompts_from_label_map(self, label_map_path):
        """Loads initial positive floor points from the label_map.json file."""
        try:
            if self.label_map is None: # Ensure label map was loaded
                 print("[ERROR] Cannot load initial prompts, label_map not loaded.")
                 self.initial_floor_points = None
                 return

            points_data = self.label_map.get("points", {})
            floor_points_list = points_data.get(self.floor_label, [])

            if floor_points_list:
                self.initial_floor_points = np.array(floor_points_list, dtype=np.int32)
            else:
                self.initial_floor_points = np.empty((0, 2), dtype=np.int32) # Use empty array if none found

        except Exception as e:
            print(f"[ERROR] Failed to load initial prompts from label map {label_map_path}: {e}")
            self.initial_floor_points = None


    def update_grid_with_obstacle(self, frame, robot_grid_coords, obstacle_distance_cm, obstacle_direction_vector):
        """
        Updates the grid based on a detected obstacle using SAM2.

        Args:
            frame (np.ndarray): The current camera frame (BGR format).
            robot_grid_coords (tuple): The robot's current (col, row) in the grid.
            obstacle_distance_cm (float): Distance to the obstacle in cm.
            obstacle_direction_vector (tuple): (x, y) vector indicating obstacle direction relative to robot.

        Returns:
            np.ndarray or None: The updated grid, or None if an error occurs.
        """
        if not self.initialized:
            print("[ERROR] SAMCallbackService not initialized.")
            return None
        if self.predictor is None:
            print("[ERROR] SAM predictor not available.")
            return None
        if self.pixels_per_cm is None or self.grid_cols is None or self.grid_rows is None:
            print("[ERROR] Calibration data missing. Cannot perform update.")
            return None
        if self.original_grid is None:
             print("[ERROR] Original grid not loaded. Cannot perform update.")
             return None

        start_time = time.time()
        print("\n--- Starting Grid Update ---")
        print(f"Robot Grid Coords: {robot_grid_coords}")
        print(f"Obstacle Distance (cm): {obstacle_distance_cm}")
        print(f"Obstacle Direction Vector: {obstacle_direction_vector}")

        # Create a copy of the original grid to modify
        new_grid = self.original_grid.copy()
        img_h, img_w = frame.shape[:2]

        try:
            # --- 1. Convert Robot Grid Coords to Pixel Coords ---
            # Calculate pixel center of the robot's grid cell
            pixel_per_grid_x = img_w / self.grid_cols
            pixel_per_grid_y = img_h / self.grid_rows
            robot_px_x = (robot_grid_coords[0] + 0.5) * pixel_per_grid_x
            robot_px_y = (robot_grid_coords[1] + 0.5) * pixel_per_grid_y
            print(f"  Robot Pixel Coords (approx center): ({robot_px_x:.1f}, {robot_px_y:.1f})")

            # --- 2. Calculate Obstacle Pixel Coords ---
            # Obstacle distance in pixels
            obstacle_distance_px = obstacle_distance_cm * self.pixels_per_cm

            # Obstacle offset from robot in pixels
            norm = math.sqrt(obstacle_direction_vector[0]**2 + obstacle_direction_vector[1]**2)
            if norm < 1e-6:
                 print("[WARN] Obstacle direction vector has zero length. Using robot position.")
                 obstacle_offset_px_x = 0
                 obstacle_offset_px_y = 0
            else:
                 # Normalize direction vector before scaling by distance
                 norm_dx = obstacle_direction_vector[0] / norm
                 norm_dy = obstacle_direction_vector[1] / norm
                 obstacle_offset_px_x = norm_dx * obstacle_distance_px
                 obstacle_offset_px_y = norm_dy * obstacle_distance_px

            # Absolute obstacle pixel coordinates
            obstacle_px_x = robot_px_x + obstacle_offset_px_x
            obstacle_px_y = robot_px_y + obstacle_offset_px_y

            # Clip coordinates to be within image bounds
            obstacle_px_x = int(np.clip(obstacle_px_x, 0, img_w - 1))
            obstacle_px_y = int(np.clip(obstacle_px_y, 0, img_h - 1))
            obstacle_point = np.array([[obstacle_px_x, obstacle_px_y]], dtype=np.int32)

            print(f"  Calculated Obstacle Pixel Coords: ({obstacle_px_x}, {obstacle_px_y})")

            # --- 3. Prepare SAM Prompts ---
            # Combine initial positive floor points + new negative obstacle point
            if self.initial_floor_points is None: # Should be an empty array if none loaded
                 print("[ERROR] Initial floor points array is None.")
                 return None

            point_coords = np.vstack((self.initial_floor_points, obstacle_point))
            # Labels: 1 for positive floor points, 0 for negative obstacle point
            point_labels = np.concatenate((np.ones(len(self.initial_floor_points), dtype=np.int32), np.array([0], dtype=np.int32)))

            # --- 4. Run SAM Prediction ---
            print(f"  Running SAM prediction with {len(point_coords)} points ({len(point_labels)-1} pos, 1 neg)...")
            predict_start_time = time.time()

            # Convert frame BGR to RGB for SAM predictor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Predictor expects PIL Image or NumPy array (RGB)
            self.predictor.set_image(frame_rgb)

            # Use inference_mode and autocast similar to initial.py
            with torch.inference_mode():
                if self.device.type == 'cuda':
                    with torch.autocast(self.device.type, dtype=torch.bfloat16):
                        masks, scores, logits = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True, # Get multiple masks
                        )
                else: # CPU or MPS
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True, # Get multiple masks
                    )

            predict_time = time.time() - predict_start_time
            print(f"  SAM prediction finished in {predict_time:.3f}s.")

            # masks shape: (N, H, W) where N is the number of masks generated
            if masks is None or masks.ndim != 3 or masks.shape[0] == 0:
                 print("[ERROR] SAM did not return valid masks.")
                 return None

            # --- Select the best mask (e.g., highest score, or largest valid area) ---
            # Using highest score among the typically 3 masks returned
            best_mask_idx = np.argmax(scores)
            selected_mask = masks[best_mask_idx]
            selected_score = scores[best_mask_idx]
            print(f"  Selected mask index {best_mask_idx} with score {selected_score:.3f}")

            # --- 5. Convert SAM mask to grid coordinates ---
            print("  Converting SAM mask to grid coordinates...")
            # Resize mask to match grid dimensions using INTER_NEAREST
            resized_mask = cv2.resize(selected_mask.astype(np.uint8), (self.grid_cols, self.grid_rows), interpolation=cv2.INTER_NEAREST)
            resized_mask_bool = resized_mask.astype(bool)

            # Find cells that were originally floor but are NOT floor in the new mask
            # These are considered obstacle cells based on the negative prompt
            if self.floor_label_value is None:
                 print("[ERROR] Floor label value not set. Cannot determine obstacle cells.")
                 return None

            obstacle_cells = (new_grid == self.floor_label_value) & (~resized_mask_bool)

            # Set these obstacle cells to 0 (background/non-navigable)
            new_grid[obstacle_cells] = 0
            print(f"  Grid update finished. Marked {np.sum(obstacle_cells)} cells as obstacle.")

            total_time = time.time() - start_time
            print(f"Total processing time for update: {total_time:.3f}s")
            return new_grid

        except Exception as e:
            print(f"[ERROR] Unexpected error during grid update: {e}")
            import traceback
            traceback.print_exc()
            return None

# --- Example Usage ---
if __name__ == "__main__":
    print("Starting SAM Callback Service Example...")

    # --- Configuration ---
    # Paths are relative to this script (CV_app/sam_callback.py)
    WORKING_DIR_REL = "../working_dataset" # Relative path to working_dataset from CV_app/
    CALIBRATION_JSON_REL = "calibration.json" # Relative path to calibration.json from CV_app/
    FLOOR_LABEL = "floor" # Make sure this matches the label used in initial.py

    # --- Instantiate the service ---
    # This loads the predictor and initial data
    sam_service = SAMCallbackService(
        hf_model_name="facebook/sam2-hiera-tiny", # Or other desired HF model
        working_dir=WORKING_DIR_REL,
        calibration_json_path=CALIBRATION_JSON_REL,
        floor_label=FLOOR_LABEL
    )

    if sam_service.initialized:
        print("\n--- Service Initialized. Ready for updates. ---")

        # --- Dummy Data for Testing ---
        # Create a dummy frame (e.g., black image) matching expected dimensions if needed
        # Example: Use dimensions from calibration if available
        if sam_service.pixels_per_cm and sam_service.grid_cols and sam_service.grid_rows and sam_service.grid_size_cm:
             dummy_h = int(sam_service.grid_rows * sam_service.grid_size_cm * sam_service.pixels_per_cm)
             dummy_w = int(sam_service.grid_cols * sam_service.grid_size_cm * sam_service.pixels_per_cm)
             print(f"Creating dummy frame of size: {dummy_w}x{dummy_h}")
             dummy_frame = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        else:
             print("[WARN] Cannot determine frame size from calibration. Using default 640x480.")
             dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Default size

        # Example robot position and obstacle detection
        robot_pos = (sam_service.grid_cols // 2, sam_service.grid_rows - 1) # Middle bottom row
        obstacle_dist = 50.0 # cm
        obstacle_dir = (0, -1) # Directly in front (negative Y direction in grid/image)

        # --- Perform Update ---
        updated_grid = sam_service.update_grid_with_obstacle(
            frame=dummy_frame,
            robot_grid_coords=robot_pos,
            obstacle_distance_cm=obstacle_dist,
            obstacle_direction_vector=obstacle_dir
        )

        if updated_grid is not None:
            print("\n--- Grid Update Successful ---")
            # You could visualize or save the updated_grid here
            # Example: Save to a test file
            output_path = os.path.join(sam_service.script_dir, "test_updated_grid.npy")
            np.save(output_path, updated_grid)
            print(f"Saved test updated grid to {output_path}")
            unique, counts = np.unique(updated_grid, return_counts=True)
            print(f"Updated grid unique values & counts: {dict(zip(unique, counts))}")
        else:
            print("\n--- Grid Update Failed ---")

    else:
        print("\n--- Service Initialization Failed ---")
