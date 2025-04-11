import cv2
import numpy as np
import time
import os
import sys
import yaml
import argparse
import math
import json
from aruco_callback import ArucoCallbackService

# --- Helper Functions ---

def load_calibration(calibration_json_path):
    """
    Load all calibration parameters from calibration.json using ArucoCallbackService.
    
    Args:
        calibration_json_path (str): Path to calibration.json file
        
    Returns:
        tuple: (camera_matrix, dist_coeffs, grid_cols, grid_rows, marker_size) or (None, None, None, None, None) if loading fails
    """
    try:
        # Initialize ArucoCallbackService to load calibration
        aruco_service = ArucoCallbackService(
            calibration_json_path=calibration_json_path,
            marker_size_meters=0.05  # Default value, will be overridden by JSON if available
        )
        
        if not aruco_service.initialized:
            print("[ERROR] Failed to initialize ArucoCallbackService for calibration loading")
            return None, None, None, None, None
            
        # Get all calibration parameters from the service
        camera_matrix = aruco_service.camera_matrix
        dist_coeffs = aruco_service.dist_coeffs
        grid_cols = aruco_service.grid_cols
        grid_rows = aruco_service.grid_rows
        marker_size = aruco_service.marker_size_meters
        
        if camera_matrix is None or dist_coeffs is None:
            print("[ERROR] Camera calibration parameters not loaded")
            return None, None, None, None, None
            
        if grid_cols is None or grid_rows is None:
            print("[ERROR] Grid dimensions not loaded")
            return None, None, None, None, None
            
        print(f"[INFO] Loaded calibration: Grid {grid_cols}x{grid_rows}, Marker size: {marker_size}m")
        return camera_matrix, dist_coeffs, grid_cols, grid_rows, marker_size
        
    except Exception as e:
        print(f"[ERROR] Failed to load calibration: {e}")
        return None, None, None, None, None

def pixel_to_grid(px, py, img_width, img_height, grid_cols, grid_rows):
    """Converts pixel coordinates to grid cell indices [col, row]."""
    if grid_cols is None or grid_rows is None or grid_cols <= 0 or grid_rows <= 0:
        # print("[WARN] Invalid grid dimensions for pixel_to_grid conversion.")
        return None # Return None if dimensions are invalid

    pixel_per_grid_x = img_width / grid_cols
    pixel_per_grid_y = img_height / grid_rows

    # Ensure division by zero doesn't occur if grid size somehow invalid
    if pixel_per_grid_x <= 0 or pixel_per_grid_y <= 0:
         return None

    grid_col = int(px / pixel_per_grid_x)
    grid_row = int(py / pixel_per_grid_y)

    # Clamp to valid grid indices (0 to N-1)
    grid_col = max(0, min(grid_col, grid_cols - 1))
    grid_row = max(0, min(grid_row, grid_rows - 1))

    return [grid_col, grid_row] # Return as list [col, row]

def load_reference_angle(svg_path):
    """Loads the reference angle from the ArUco marker SVG file.
    
    Args:
        svg_path (str): Path to the ArUco marker SVG file.
        
    Returns:
        float or None: The reference angle in degrees, or None if loading fails.
    """
    try:
        if not os.path.exists(svg_path):
            print(f"[ERROR] Reference SVG file not found: {svg_path}")
            return None
            
        # The reference angle is typically 0 degrees for the standard ArUco marker
        # This is because the marker's orientation is defined with the top-left corner
        # as the reference point, which corresponds to 0 degrees
        return 0.0
        
    except Exception as e:
        print(f"[ERROR] Failed to load reference angle from {svg_path}: {e}")
        return None

def detect_aruco_markers(frame, camera_matrix, dist_coeffs, marker_size=0.05):
    """
    Detect ArUco markers in a frame and return their poses.
    
    Args:
        frame (np.ndarray): Input frame (BGR format)
        camera_matrix (np.ndarray): Camera matrix
        dist_coeffs (np.ndarray): Distortion coefficients
        marker_size (float): Size of ArUco markers in meters
        
    Returns:
        dict: Dictionary mapping marker IDs to their poses (rvec, tvec)
    """
    if camera_matrix is None or dist_coeffs is None:
        print("[ERROR] Camera calibration parameters not provided")
        return None
        
    # Initialize ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None:
        return None
        
    # Estimate poses
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_size, camera_matrix, dist_coeffs
    )
    
    # Create dictionary of detected markers
    markers = {}
    for i, marker_id in enumerate(ids.flatten()):
        markers[marker_id] = (rvecs[i], tvecs[i])
        
    return markers

def visualize_markers(frame, markers, camera_matrix, dist_coeffs, marker_size=0.05):
    """
    Visualize detected ArUco markers in the frame.
    
    Args:
        frame (np.ndarray): Input frame (BGR format)
        markers (dict): Dictionary of detected markers
        camera_matrix (np.ndarray): Camera matrix
        dist_coeffs (np.ndarray): Distortion coefficients
        marker_size (float): Size of ArUco markers in meters
        
    Returns:
        np.ndarray: Frame with visualization
    """
    if markers is None:
        return frame
        
    # Draw axes for each marker
    for marker_id, (rvec, tvec) in markers.items():
        cv2.aruco.drawDetectedMarkers(frame, np.array([rvec]), np.array([tvec]))
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size/2)
        
    return frame

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Detect ArUco markers, position, and angle from IP Webcam.')
    parser.add_argument('--ip', type=str, default='192.168.1.100', help='IP address of the IP Webcam server.')
    parser.add_argument('--port', type=str, default='8080', help='Port of the IP Webcam server.')
    parser.add_argument('--calib', type=str, default='calibration.json', help='Path to calibration JSON file.')
    args = parser.parse_args()

    # Define paths relative to the script's directory
    script_dir = os.path.dirname(__file__)
    calib_json_path = os.path.join(script_dir, args.calib)
    reference_svg_path = os.path.join(script_dir, "aruco.svg")

    # Load all calibration data
    camera_matrix, dist_coeffs, grid_cols, grid_rows, marker_size = load_calibration(calib_json_path)
    
    if camera_matrix is None or dist_coeffs is None:
        print("[ERROR] Failed to load camera calibration")
        return
    
    if grid_cols is None or grid_rows is None:
        print("[ERROR] Failed to load grid dimensions")
        return
    
    if marker_size is None:
        print("[WARN] Using default marker size of 0.05 meters")
        marker_size = 0.05

    # Load reference SVG
    ref_angle = load_reference_angle(reference_svg_path)
    if ref_angle is None:
        print("[WARN] Using default reference angle of 0 degrees")
        ref_angle = 0.0

    # 2. Setup ArUco Detection
    aruco_dict_type = cv2.aruco.DICT_4X4_50
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    print(f"[INFO] Using ArUco dictionary: DICT_4X4_50")

    # 3. Setup Video Stream
    video_url = f"http://{args.ip}:{args.port}/video"
    print(f"[INFO] Attempting to connect to IP Webcam at: {video_url}")
    cap = cv2.VideoCapture(video_url)
    # Optional: Improve stream stability
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream from {video_url}")
        print("        Check IP address, port, network connection, and if IP Webcam app is running.")
        sys.exit(1)
    print("[INFO] Video stream opened successfully. Press 'q' to quit.")

    last_print_time = time.time()
    img_height, img_width = None, None # Initialize image dimensions

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARN] Could not read frame from stream. Retrying...")
            time.sleep(0.5) # Wait before retrying
            # Optional: Try to reconnect
            cap.release()
            cap = cv2.VideoCapture(video_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                 print("[ERROR] Failed to reconnect to video stream. Exiting.")
                 break
            continue

        # Store frame dimensions once
        if img_height is None or img_width is None:
            img_height, img_width = frame.shape[:2]
            print(f"[INFO] Frame dimensions: {img_width}x{img_height}")

        current_time = time.time()

        # Process frame only every second
        if current_time - last_print_time >= 1.0:
            last_print_time = current_time

            # Detect markers
            corners, ids, rejected = detector.detectMarkers(frame)

            print("\n" + "-" * 50)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"Time: {timestamp}")

            if ids is not None:
                print(f"Detected {len(ids)} markers.")
                # Estimate pose for detected markers
                rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, camera_matrix, dist_coeffs)

                for i, marker_id in enumerate(ids):
                    marker_id_val = marker_id[0]
                    corner_set = corners[i][0] # Shape (4, 2)

                    # --- Calculate Grid Coordinates ---
                    center_x = int(np.mean(corner_set[:, 0]))
                    center_y = int(np.mean(corner_set[:, 1]))
                    grid_coords = None
                    if grid_cols is not None and grid_rows is not None: # Check if dimensions were loaded
                         grid_coords = pixel_to_grid(center_x, center_y, img_width, img_height, grid_cols, grid_rows)

                    # --- Calculate Angle ---
                    rvec = rvecs[i][0]
                    rotation_matrix, _ = cv2.Rodrigues(rvec) # Convert rvec to 3x3 rotation matrix

                    # The columns of the rotation matrix are the marker axes in camera coords.
                    # Marker X-axis vector in camera coordinates (points right on reference marker)
                    marker_x_axis_cam = rotation_matrix[:, 0]
                    dx, dy, dz = marker_x_axis_cam

                    # Calculate angle relative to image +X axis (right)
                    # Use atan2(-dy, dx) because image Y points down.
                    # This makes 0 degrees point right, 90 up, -90 down, 180 left.
                    angle_rad = math.atan2(-dy, dx)
                    angle_deg = math.degrees(angle_rad) # Result is in (-180, 180]

                    # Remove normalization to keep angle in range [-180, 180)
                    # angle_deg_normalized = (angle_deg + 360) % 360

                    # --- Print Results ---
                    grid_str = f"[{grid_coords[0]},{grid_coords[1]}]" if grid_coords else "N/A"
                    print(f"  Marker {marker_id_val}: Grid {grid_str}, Angle: {angle_deg:.1f}°")

            else:
                print("No markers detected.")
            print("-" * 50)

        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit key pressed. Exiting.")
            break

    # --- Cleanup ---
    cap.release()
    print("[INFO] Video stream released.")

if __name__ == '__main__':
    main()
