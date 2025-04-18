"""
Camera Feed and ArUco Detection Debugger (Map Indices Input)

Connects to the specified IP Webcam, runs ArUco detection assuming
'grid_coords' from the service are (gx, gy) corresponding to map (col, row).
Calculates pixel and real-world coordinates from the map indices.
Displays:
- Live video feed with detected markers highlighted.
- The calculated pixel location of the marker center clearly marked.
- Console output detailing coordinates (map, pixel, real-world) and angles
  (raw Aruco, standard navigation) for the target robot marker.
- Overlay text on the video feed showing the same information.
- Direction line visualized using OpenCV convention (0=Right, CW Positive), corrected.
- Press 's' to save the current frame to the 'initial_images' folder.
- Press 'q' to quit.
"""

import cv2
import numpy as np
import os
import sys
import json
import math
import time
import traceback
from datetime import datetime # Needed for timestamped filenames

# --- Add parent directory to sys.path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- End Path Modification ---

# --- Custom Imports ---
try:
    from CV_app.aruco_callback import ArucoCallbackService
except ImportError as e: print(f"Import Error: {e}"); exit()
try:
    from simulation import normalize_angle
except ImportError:
    print("Error: Could not import normalize_angle from simulation.py.")
    def normalize_angle(angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle <= -math.pi: angle += 2.0 * math.pi
        return angle

# --- Configuration (Match your server scripts) ---
IP_WEBCAM_URL = "http://192.168.25.173:8080" # Make sure this matches your setup
ROBOT_ARUCO_ID = 0 # The ID of the marker on your robot

# Paths for calibration/camera info
CALIBRATION_FILE = os.path.join(parent_dir, 'CV_app/calibration.json')
CAMERA_YAML_FILE = os.path.join(parent_dir, 'CV_app/camera.yaml')
WORKING_DIR = os.path.join(parent_dir, "working_dataset")

# Output Folder for Snapshots
SNAPSHOT_FOLDER = os.path.join(script_dir, "initial_images")

# Coordinate System & Scaling (Load from calibration)
PX_PER_CM = 0.0
CM_PER_METER = 100.0
GRID_CELL_SIZE_PX = 0.0
GRID_COLS = 0 # Not strictly needed here, but good for context
GRID_ROWS = 0
calibration_valid = False
try:
    with open(CALIBRATION_FILE, 'r') as f:
        calibration_data = json.load(f)
    PX_PER_CM = float(calibration_data['pixel_ratio']['value'])
    GRID_CELL_SIZE_PX = float(calibration_data['grid']['cell_size_pixels'])
    GRID_COLS = int(calibration_data['grid']['columns'])
    GRID_ROWS = int(calibration_data['grid']['rows'])
    print(f"Loaded PX_PER_CM = {PX_PER_CM}")
    print(f"Loaded GRID_CELL_SIZE_PX = {GRID_CELL_SIZE_PX}")
    print(f"Loaded Grid Dimensions: {GRID_COLS} cols x {GRID_ROWS} rows")
    if PX_PER_CM > 0 and GRID_CELL_SIZE_PX > 0:
        calibration_valid = True
    else:
        print("Warning: Invalid scaling values (<= 0) loaded from calibration.")
except Exception as e:
    print(f"Error loading calibration data from {CALIBRATION_FILE}: {e}")
    print("Coordinate conversions may fail.")

# --- Coordinate Transformation Helpers ---
def map_indices_to_pixel(row, col):
    """Converts map indices (row, col) to pixel coordinates (px, py)."""
    if not calibration_valid: return None, None
    px = col * GRID_CELL_SIZE_PX
    py = row * GRID_CELL_SIZE_PX
    return px, py

def pixel_to_real(px, py):
    """Converts pixel coordinates (px, py) to real-world meters (rx, ry)."""
    if not calibration_valid: return None, None
    x_cm = px / PX_PER_CM
    y_cm = py / PX_PER_CM
    return x_cm / CM_PER_METER, y_cm / CM_PER_METER

# --- Main Debug Function ---
def run_debug_feed():
    # Create Snapshot Folder
    snapshot_folder_path = os.path.abspath(SNAPSHOT_FOLDER) # Use absolute path
    if not os.path.exists(snapshot_folder_path):
        try:
            os.makedirs(snapshot_folder_path)
            print(f"Created snapshot folder: {snapshot_folder_path}")
        except OSError as e:
            print(f"FATAL: Error creating snapshot folder {snapshot_folder_path}: {e}")
            return
    else:
         print(f"Snapshot folder exists: {snapshot_folder_path}") # Confirm folder exists

    print("Initializing ArUco Service...")
    aruco_service = ArucoCallbackService(
        working_dir=WORKING_DIR,
        camera_yaml_path=CAMERA_YAML_FILE,
        calibration_json_path=CALIBRATION_FILE
    )
    if not aruco_service.initialized: print("FATAL: Failed to initialize ArucoCallbackService."); return

    print(f"Attempting to connect to IP Webcam stream: {IP_WEBCAM_URL}")
    cap = None
    stream_url = IP_WEBCAM_URL if IP_WEBCAM_URL.endswith('/video') else f"{IP_WEBCAM_URL}/video"
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"WARN: Could not open IP Webcam stream at {stream_url}. Trying base URL...")
        cap = cv2.VideoCapture(IP_WEBCAM_URL)
        if not cap.isOpened(): print(f"FATAL: Could not open IP Webcam stream at base URL {IP_WEBCAM_URL} either."); return
    print("Successfully connected to IP Webcam stream.")

    frame_count = 0
    start_time = time.time()

    # Text Overlay Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0) # Green
    line_type = 2
    text_start_x = 10
    text_start_y = 60 # Start below FPS
    line_height = 25

    # Marker parameters
    marker_center_color = (0, 0, 255) # Red
    marker_center_radius = 5
    marker_label_color = (255, 255, 0) # Cyan
    marker_label_offset_x = 10
    marker_label_offset_y = -10
    direction_line_color = (255, 0, 0) # Blue
    direction_line_thickness = 2
    direction_line_length = 30

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to capture frame or frame is empty. Retrying connection...");
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(stream_url);
                if not cap.isOpened(): cap = cv2.VideoCapture(IP_WEBCAM_URL)
                if not cap.isOpened(): print("Error: Failed to reconnect."); time.sleep(5.0)
                continue

            frame_count += 1
            display_frame = frame.copy()

            # Detect Markers
            markers = aruco_service.detect_markers_in_frame(frame)

            found_robot = False
            robot_info_text = []

            if markers:
                # Draw all detected markers outlines
                all_ids = [m.get('id') for m in markers if m.get('id') is not None]
                all_corners = [m.get('corners') for m in markers if m.get('corners') is not None]
                if all_ids and len(all_ids) == len(all_corners):
                     formatted_corners = [np.array(c, dtype=np.float32).reshape(-1,1,2) if len(c)==4 else None for c in all_corners]
                     valid_corners = [c for c in formatted_corners if c is not None and c.shape == (4,1,2)]
                     valid_ids = [all_ids[i] for i, c in enumerate(formatted_corners) if c is not None and c.shape == (4,1,2)]
                     if valid_corners:
                         cv2.aruco.drawDetectedMarkers(display_frame, valid_corners, np.array(valid_ids, dtype=np.int32))

                # Process and Print Target Robot Info
                for marker in markers:
                    marker_id = marker.get('id')
                    if marker_id == ROBOT_ARUCO_ID:
                        found_robot = True
                        aruco_grid_coords = marker.get('grid_coords') # Assumed (gx, gy)
                        angle_deg_aruco = marker.get('angle')         # Assumed (0=Down, CW+)

                        print("-" * 30)
                        print(f"Robot Marker Found (ID: {marker_id})")
                        robot_info_text.append(f"Robot ID: {marker_id} Found")

                        row, col, px, py, world_x_m, world_y_m = None, None, None, None, None, None
                        world_theta_rad_norm = None # Nav angle
                        center_px, center_py = None, None # Pixel coords for drawing

                        if aruco_grid_coords:
                            gx, gy = aruco_grid_coords
                            # --- >>> ASSUMPTION: gx=col, gy=row <<< ---
                            row, col = gy, gx
                            print(f"  Map Indices(row, col): ({row:>6.1f}, {col:>6.1f})")
                            robot_info_text.append(f"Map R: {row:>6.1f}, C: {col:>6.1f}")
                            # --- >>> END ASSUMPTION <<< ---

                            # Calculate Pixel Coords from Map Indices
                            px, py = map_indices_to_pixel(row, col)
                            if px is not None:
                                print(f"  Pixel Coords (px, py): ({px:>4.0f}, {py:>4.0f})")
                                robot_info_text.append(f"Px: {px:>4.0f}, Py: {py:>4.0f}")
                                center_px = int(round(px)) # Use for drawing
                                center_py = int(round(py))
                                cv2.circle(display_frame, (center_px, center_py), marker_center_radius, marker_center_color, -1)
                                cv2.putText(display_frame, "Center (Px,Py)",
                                            (center_px + marker_label_offset_x, center_py + marker_label_offset_y),
                                            font, font_scale * 0.8, marker_label_color, line_type)

                                # Calculate Real World Coords from Pixel Coords
                                world_x_m, world_y_m = pixel_to_real(px, py)
                                if world_x_m is not None:
                                    print(f"  Real Coords  (m):    ({world_x_m:>6.3f}, {world_y_m:>6.3f})")
                                    robot_info_text.append(f"Rx: {world_x_m:>6.3f}m, Ry: {world_y_m:>6.3f}m")
                                else:
                                    print("  Real Coords  (m):    N/A (Calculation failed)")
                                    robot_info_text.append("Rx: N/A, Ry: N/A")
                            else:
                                print("  Pixel Coords (px, py): N/A (Calculation failed)")
                                print("  Real Coords  (m):    N/A")
                                robot_info_text.append("Px: N/A, Py: N/A")
                                robot_info_text.append("Rx: N/A, Ry: N/A")
                        else:
                            print("  Map Indices(row, col): Not Detected")
                            print("  Pixel Coords (px, py): N/A")
                            print("  Real Coords  (m):    N/A")
                            robot_info_text.append("Coords: Not Detected")

                        if angle_deg_aruco is not None:
                            print(f"  Aruco Angle(0=D, CW): {angle_deg_aruco:>5.1f} deg")
                            robot_info_text.append(f"Ang(Aruco): {angle_deg_aruco:>5.1f} deg")

                            # Calculate Standard Navigation Angle (0=Right, CCW+)
                            world_theta_rad = math.radians(270.0 - angle_deg_aruco) # Corrected formula
                            world_theta_rad_norm = normalize_angle(world_theta_rad)
                            print(f"  Nav Angle  (0=R, CCW): {math.degrees(world_theta_rad_norm):>5.1f} deg ({world_theta_rad_norm:.3f} rad)")
                            robot_info_text.append(f"Ang(Nav): {math.degrees(world_theta_rad_norm):>5.1f} deg")

                            # Calculate Angle for Visualization (0=Right, CW Positive)
                            opencv_angle_rad = math.radians(90.0 - angle_deg_aruco)
                            # Add 180 degrees (pi radians) to flip direction
                            opencv_angle_rad_corrected = opencv_angle_rad + math.pi
                            # Normalize the corrected angle
                            opencv_angle_rad_norm = normalize_angle(opencv_angle_rad_corrected)

                            # Draw direction line using the CORRECTED OpenCV angle convention
                            if center_px is not None:
                                end_x = int(center_px + direction_line_length * math.cos(opencv_angle_rad_norm))
                                end_y = int(center_py + direction_line_length * math.sin(opencv_angle_rad_norm)) # Use '+' for sin
                                cv2.line(display_frame, (center_px, center_py), (end_x, end_y),
                                         direction_line_color, direction_line_thickness)
                        else:
                            print("  Aruco Angle(0=D, CW): Not Detected")
                            print("  Nav Angle  (0=R, CCW): N/A")
                            robot_info_text.append("Angles: Not Detected")
                        print("-" * 30)
                        break # Stop after finding the robot marker

            if not found_robot:
                 if frame_count % 60 == 0:
                    print(f"Robot Marker (ID: {ROBOT_ARUCO_ID}) not detected in frame.")
                 robot_info_text.append(f"Robot ID: {ROBOT_ARUCO_ID} NOT Found")

            # Display FPS
            end_time = time.time(); elapsed = end_time - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (text_start_x, 30),
                            font, font_scale, font_color, line_type)

            # Display Robot Info Text Overlay
            current_y = text_start_y
            for line in robot_info_text:
                cv2.putText(display_frame, line, (text_start_x, current_y),
                            font, font_scale, font_color, line_type)
                current_y += line_height

            # Show Frame
            cv2.imshow("IP Webcam Feed - ArUco Debug (Map Indices Input)", display_frame)

            # Handle Key Presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit requested.")
                break
            elif key == ord('s'):
                if frame is not None and frame.size > 0: # Check if frame is valid
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"snapshot_{timestamp}.jpg"
                    save_path = os.path.join(snapshot_folder_path, filename)
                    try:
                        print(f"Attempting to save snapshot to: {save_path}") # Log attempt
                        success = cv2.imwrite(save_path, frame) # Save original frame
                        if success:
                            print(f"Successfully saved snapshot to: {save_path}")
                        else:
                            print(f"Error: cv2.imwrite failed to save snapshot to {save_path}. Check permissions and path validity.")
                    except Exception as e:
                        print(f"Exception while saving snapshot to {save_path}: {e}")
                        traceback.print_exc() # Print full traceback for exceptions
                else:
                    print("Skipping save: Frame is empty or invalid.")

        except Exception as e:
            print(f"Error in main loop: {e}")
            traceback.print_exc()
            time.sleep(1) # Avoid rapid error loops

    # Cleanup
    print("Releasing resources...")
    if cap: cap.release()
    cv2.destroyAllWindows()
    print("Debug script finished.")

if __name__ == "__main__":
    run_debug_feed() 