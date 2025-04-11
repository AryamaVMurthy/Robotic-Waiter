import cv2
import numpy as np
import time
import os
import sys
import yaml
import math
import json

class ArucoCallbackService:
    """
    A service to detect ArUco markers (position and angle) in individual frames.

    Initializes the ArUco detector and loads calibration data once.
    Provides a method to process a frame and return marker information.
    """
    def __init__(self,
                 working_dir="working_dataset",
                 camera_yaml_path="camera.yaml",
                 calibration_json_path="calibration.json",
                 aruco_dict_type=cv2.aruco.DICT_4X4_50,
                 marker_size_meters=0.05, # Default, can be overridden by JSON
                 reference_aruco_svg="aruco.svg"):
        """Initialize the ArucoCallbackService.

        Args:
            working_dir (str): Path to the working dataset directory.
            camera_yaml_path (str): Path to the camera calibration YAML file (relative to script dir).
            calibration_json_path (str): Path to the calibration JSON file (relative to script dir).
            aruco_dict_type (int): The type of ArUco dictionary to use.
            marker_size_meters (float): The size of the ArUco marker in meters. This is a default
                                     and will be overridden if found in calibration.json.
            reference_aruco_svg (str): Path to the reference ArUco SVG file (relative to script dir).
        """
        self.script_dir = os.path.dirname(__file__)
        self.working_dir = os.path.join(self.script_dir, working_dir) # Assume working_dir is relative to script
        self.reference_aruco_svg = os.path.join(self.script_dir, reference_aruco_svg)

        # Load calibration and camera parameters using paths relative to script dir
        self.camera_matrix, self.dist_coeffs = self._load_camera_calibration(os.path.join(self.script_dir, camera_yaml_path))
        self.grid_cols, self.grid_rows, self.marker_size_meters = self._load_calibration_json(os.path.join(self.script_dir, calibration_json_path))

        # If marker size wasn't loaded from JSON, use the default
        if self.marker_size_meters is None:
            self.marker_size_meters = marker_size_meters
            print(f"[WARN] Using default marker size: {self.marker_size_meters} meters")

        self.initialized = False
        self.detector = None
        self.aruco_dict_type = aruco_dict_type

        print("Initializing ArucoCallbackService...")
        try:
            # 3. Setup ArUco Detector
            print(f"Setting up ArUco detector (Dict: {self.aruco_dict_type})...")
            aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
            parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            print("ArUco detector initialized.")

            self.initialized = True
            print("ArucoCallbackService initialized successfully.")

        except FileNotFoundError as e:
            print(f"[ERROR] Initialization failed: {e}")
        except ValueError as e:
             print(f"[ERROR] Initialization failed: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error during initialization: {e}")
            import traceback
            traceback.print_exc()

    def _load_camera_calibration(self, yaml_path):
        """Loads camera calibration data from a YAML file."""
        print(f"[INFO] Loading camera calibration from: {yaml_path}")
        try:
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

            with open(yaml_path, 'r') as f:
                calib_data = yaml.safe_load(f)

            camera_matrix = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
            dist_coeffs = np.array([
                calib_data['distortion_coefficients']['k1'],
                calib_data['distortion_coefficients']['k2'],
                calib_data['distortion_coefficients']['p1'],
                calib_data['distortion_coefficients']['p2'],
                calib_data['distortion_coefficients']['k3']
            ])
            return camera_matrix, dist_coeffs
        except Exception as e:
            print(f"[ERROR] Error loading camera calibration file {yaml_path}: {e}")
            return None, None

    def _load_calibration_json(self, json_path):
        """Loads calibration values from the calibration.json file."""
        try:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Calibration file not found: {json_path}")
            
            with open(json_path, 'r') as f:
                calib_data = json.load(f)

            # Load grid dimensions
            grid_cols = None
            grid_rows = None
            if 'grid' in calib_data:
                if 'columns' in calib_data['grid']:
                    grid_cols = calib_data['grid']['columns']
                if 'rows' in calib_data['grid']:
                    grid_rows = calib_data['grid']['rows']
                print(f"[INFO] Loaded grid dimensions: {grid_cols}x{grid_rows}")

            # Load ArUco marker size if present
            marker_size = None
            if 'aruco' in calib_data and 'marker_size' in calib_data['aruco']:
                marker_size = calib_data['aruco']['marker_size']
                print(f"[INFO] Loaded ArUco marker size: {marker_size}m")

            return grid_cols, grid_rows, marker_size

        except Exception as e:
            print(f"Error loading or parsing calibration file {json_path}: {e}")
            return None, None, None

    def _pixel_to_grid(self, px, py, img_width, img_height):
        """Converts pixel coordinates to grid cell indices [col, row]."""
        if self.grid_cols is None or self.grid_rows is None or self.grid_cols <= 0 or self.grid_rows <= 0:
            return None # Cannot calculate if dimensions are invalid

        pixel_per_grid_x = img_width / self.grid_cols
        pixel_per_grid_y = img_height / self.grid_rows

        if pixel_per_grid_x <= 0 or pixel_per_grid_y <= 0:
             return None

        grid_col = int(px / pixel_per_grid_x)
        grid_row = int(py / pixel_per_grid_y)

        grid_col = max(0, min(grid_col, self.grid_cols - 1))
        grid_row = max(0, min(grid_row, self.grid_rows - 1))

        return [grid_col, grid_row]

    def detect_markers_in_frame(self, frame):
        """
        Detects ArUco markers in a single frame and returns their info.

        Args:
            frame (np.ndarray): The input image frame (BGR format).

        Returns:
            list: A list of dictionaries. Each dictionary contains:
                  {'id': int, 'grid_coords': [col, row] or None, 'angle': float}
                  Returns an empty list if no markers are detected or an error occurs.
        """
        if not self.initialized:
            print("[ERROR] ArucoCallbackService not initialized.")
            return []
        if frame is None or not isinstance(frame, np.ndarray):
             print("[ERROR] Invalid frame input.")
             return []
        if self.detector is None:
             print("[ERROR] ArUco detector not available.")
             return []
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("[ERROR] Camera calibration not available.")
            return []


        detected_markers_info = []
        try:
            img_height, img_width = frame.shape[:2]

            # Detect markers
            corners, ids, rejected = self.detector.detectMarkers(frame)

            if ids is not None:
                # Estimate pose for detected markers
                rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size_meters, self.camera_matrix, self.dist_coeffs)

                for i, marker_id in enumerate(ids):
                    marker_id_val = int(marker_id[0])
                    corner_set = corners[i][0]

                    # --- Calculate Grid Coordinates ---
                    center_x = int(np.mean(corner_set[:, 0]))
                    center_y = int(np.mean(corner_set[:, 1]))
                    grid_coords = self._pixel_to_grid(center_x, center_y, img_width, img_height)

                    # --- Calculate Angle ---
                    rvec = rvecs[i][0]
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    marker_x_axis_cam = rotation_matrix[:, 0]
                    dx, dy, dz = marker_x_axis_cam
                    angle_rad = math.atan2(-dy, dx) # Image Y down, angle w.r.t image +X
                    angle_deg = math.degrees(angle_rad) # Range (-180, 180]

                    detected_markers_info.append({
                        'id': marker_id_val,
                        'grid_coords': grid_coords, # Will be None if grid dimensions aren't loaded
                        'angle': angle_deg
                    })

        except Exception as e:
            print(f"[ERROR] Error during marker detection: {e}")
            import traceback
            traceback.print_exc()
            return [] # Return empty list on error

        return detected_markers_info

# --- Example Usage ---
if __name__ == "__main__":
    print("Starting Aruco Callback Service Example...")

    # --- Instantiate the service ---
    # This loads calibration data and initializes the detector
    aruco_service = ArucoCallbackService(
        working_dir="../working_dataset", # Example: Relative path up and into working_dataset
        camera_yaml_path="camera.yaml",
        calibration_json_path="calibration.json",
        reference_aruco_svg="aruco.svg"
    )

    if aruco_service.initialized:
        print("\n--- Service Initialized. Ready for frame processing. ---")

        # --- Example Frame Processing ---
        # Load a test image (replace with your frame source)
        test_image_path = "test1.jpeg" # Make sure this exists in sam2/ or provide full path
        if os.path.exists(os.path.join(aruco_service.working_dir, test_image_path)):
            dummy_frame = cv2.imread(os.path.join(aruco_service.working_dir, test_image_path))

            if dummy_frame is not None:
                print(f"\nProcessing test image: {test_image_path}")
                start_time = time.time()
                marker_results = aruco_service.detect_markers_in_frame(dummy_frame)
                proc_time = time.time() - start_time
                print(f"Processing took {proc_time:.4f} seconds.")

                if marker_results:
                    print(f"Detected {len(marker_results)} markers:")
                    for marker in marker_results:
                        grid_str = f"[{marker['grid_coords'][0]},{marker['grid_coords'][1]}]" if marker['grid_coords'] else "N/A"
                        print(f"  ID: {marker['id']}, Grid: {grid_str}, Angle: {marker['angle']:.1f}°")
                else:
                    print("No markers detected in the test image.")
            else:
                print(f"[ERROR] Failed to load test image: {test_image_path}")
        else:
            print(f"[WARN] Test image not found: {test_image_path}. Skipping processing example.")

    else:
        print("\n--- Service failed to initialize. Exiting example. ---")
