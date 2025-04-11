#!/usr/bin/env python3
"""
Path Following Server (Explicit Waypoint Control - Aligned with CTE Server Structure)

Workflow:
1. Load grid map, process (1=floor=safe).
2. Start camera thread (assumes Aruco service returns grid coords gx, gy -> col, row).
3. Display grid, wait for first ArUco detection (defines start MAP indices row, col).
4. Prompt user to click grid for goal PIXEL point, convert to MAP goal (row, col).
5. Run A* planning using MAP indices (row, col).
6. Convert planned MAP path to REAL path for control loop.
7. Start CoAP server/client and control loop (Explicit Waypoint State Machine) for navigation.
8. Visualization uses MAP coordinates for path/target/robot-map-pos.
"""

import asyncio
import socket
import math
import numpy as np
import cv2 # Needed for camera capture
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import aiocoap
import aiocoap.resource as resource
import json
import time
from threading import Thread, Lock
from collections import deque
import random
import os
import sys
import traceback

# --- Add parent directory to sys.path ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- End Path Modification ---

# --- Custom Imports (Same as CTE Server) ---
try:
    from CV_app.aruco_callback import ArucoCallbackService
except ImportError as e: print(f"Import Error: {e}"); exit()
try:
    # Import normalize_angle and ROBOT_RADIUS (used by visualizer)
    from simulation import normalize_angle, ROBOT_RADIUS
except ImportError: print("Error: Could not import from simulation.py."); exit()
try:
    # Use updated planner and selection functions
    from astar_planner import run_astar_on_grid, visualize_safe_grid, select_point
except ImportError: print("Error: Could not import from astar_planner.py."); exit()

# --- Load Calibration Data (Same as CTE Server) ---
CALIBRATION_FILE = os.path.join(parent_dir, 'CV_app/calibration.json')
calibration_data = {}
calibration_valid = False
PX_PER_CM = 0.0
GRID_CELL_SIZE_PX = 0.0
try:
    with open(CALIBRATION_FILE, 'r') as f:
        calibration_data = json.load(f)
    PX_PER_CM = float(calibration_data['pixel_ratio']['value'])
    GRID_CELL_SIZE_PX = float(calibration_data['grid']['cell_size_pixels'])
    print(f"Using PX_PER_CM = {PX_PER_CM} from calibration file.")
    print(f"Using GRID_CELL_SIZE_PX = {GRID_CELL_SIZE_PX} from calibration file.")
    if PX_PER_CM > 0 and GRID_CELL_SIZE_PX > 0:
        calibration_valid = True
    else:
        print("Warning: Invalid scaling values (<= 0) loaded from calibration.")
except Exception as e: print(f"Error loading calibration data: {e}. Please check {CALIBRATION_FILE}."); exit()

CM_PER_METER = 100.0

# --- Configuration (Mirrors CTE Server, including tuned values) ---
# Network
SERVER_IP = '10.1.39.64' # Your PC's IP
SERVER_PORT = 5683
ESP32_IP = '192.168.137.40' # Set ESP32 IP directly
CONTROL_LOOP_DELAY = 0.05
IP_WEBCAM_URL = "http://192.168.137.33:8080"

# Robot Identification
ROBOT_ARUCO_ID = 0

# --- Motor Speed Configuration (PWM 0-255) ---
MOTOR_FORWARD_SPEED = 200 # Default forward speed (Synced with CTE)
MOTOR_TURN_SPEED = 140    # Default turning speed (Synced with CTE)

# --- Explicit Waypoint Controller Parameters (meters, radians) ---
# Use values tuned for delay
ANGLE_THRESHOLD = 0.80  # Radians (~28.6 degrees) - Reverted from 0.80 based on human edit
WAYPOINT_RADIUS = 0.040 # Meters (4.0 cm) - Keep increased radius for delay tolerance

# A* Planning Parameters (Same as CTE Server)
OCCUPANCY_GRID_PATH = os.path.abspath(os.path.join(parent_dir, 'working_dataset/grid/labeled_grid.npy'))
ASTAR_SAFETY_DISTANCE_PX = 15
ASTAR_SAFETY_DISTANCE_CELLS = 0
if calibration_valid:
    ASTAR_SAFETY_DISTANCE_CELLS = int(round(ASTAR_SAFETY_DISTANCE_PX / GRID_CELL_SIZE_PX))
    print(f"Calculated A* Safety Distance in Cells: {ASTAR_SAFETY_DISTANCE_CELLS}")
else:
    print("Warning: Cannot calculate safety distance in cells due to invalid calibration.")

# --- Global State (Same as CTE Server) ---
robot_pose_real = (0.0, 0.0, 0.0) # (rx_m, ry_m, theta_rad_nav)
robot_pose_pixels = None          # (px, py) - Relative to camera frame now
robot_pose_map_indices = None     # (row, col) - Calculated from pixels
robot_pose_lock = Lock()
latest_camera_frame = None        # Store the latest frame for selection snapshot
latest_frame_lock = Lock()
esp32_status = "DISCONNECTED"
esp32_last_update = 0
command_seq = 0
last_valid_pose_time = 0.0
planned_path_map_indices = []     # Store path as [(row, col), ...] - USED BY VISUALIZER
current_path_real = None          # Store real path for control
current_waypoint_index = -1
controller_state = "IDLE"         # IDLE, ALIGNING_TO_WAYPOINT, MOVING_TO_WAYPOINT
current_command_sent = "NONE"
command_start_time = 0.0          # Used by explicit waypoint logic
smoothed_robot_theta_rad = None
EMA_ALPHA = 0.4                   # Angle smoothing factor
visualizer = None
# initial_start_pose_real = None    # Not needed by explicit waypoint controller

# --- Coordinate Transformation Helpers (Same as CTE Server) ---
def map_indices_to_pixel(row, col):
    """Converts map indices (row, col) to pixel coordinates (px, py)."""
    if not calibration_valid: return None, None
    px = col * GRID_CELL_SIZE_PX
    py = row * GRID_CELL_SIZE_PX
    return px, py

def pixel_to_map_indices(px, py):
    """Converts pixel coordinates (px, py) to map indices (row, col)."""
    if not calibration_valid: return None, None
    col = px / GRID_CELL_SIZE_PX
    row = py / GRID_CELL_SIZE_PX
    return row, col

def pixel_to_real(px, py):
    """Converts pixel coordinates (px, py) to real-world meters (rx, ry)."""
    if not calibration_valid: return None, None
    x_cm = px / PX_PER_CM
    y_cm = py / PX_PER_CM
    return x_cm / CM_PER_METER, y_cm / CM_PER_METER

def map_indices_to_real(row, col):
    """Converts map indices (row, col) to real-world meters (rx, ry)."""
    px, py = map_indices_to_pixel(row, col)
    if px is None: return None, None
    return pixel_to_real(px, py)

def real_to_pixel(rx_m, ry_m):
    """Converts real-world meters (rx_m, ry_m) back to pixel coordinates (px, py)."""
    if not calibration_valid: return None, None
    rx_cm = rx_m * CM_PER_METER; ry_cm = ry_m * CM_PER_METER
    px = rx_cm * PX_PER_CM; py = ry_cm * PX_PER_CM
    return int(round(px)), int(round(py))

# --- Camera Pose Function (Same as CTE Server) ---
def get_current_pose_from_camera():
    """
    Reads the latest pose, returns tuple:
    (robot_pose_real, robot_pose_pixels, robot_pose_map_indices) or (None, None, None) if stale.
    """
    with robot_pose_lock:
        current_real = robot_pose_real
        current_pixels = robot_pose_pixels
        current_map_indices = robot_pose_map_indices
        last_update = last_valid_pose_time
    pose_age = time.time() - last_update
    if pose_age > 1.0: # Check if pose is too old (adjust threshold if needed)
        if time.time() % 5 < CONTROL_LOOP_DELAY * 2: # Print warning occasionally
             print(f"Warning: Robot pose estimate is stale ({pose_age:.1f}s old).")
        return None, None, None
    return current_real, current_pixels, current_map_indices

# --- Camera Processing Thread (Same as CTE Server) ---
def camera_thread_function(grid_width, grid_height):
    """
    Continuously captures frames, uses ArucoCallbackService, assumes 'grid_coords'
    are (gx, gy) corresponding to (col, row), calculates other coordinate systems,
    and updates the global robot pose state. Uses corrected angle conversion.
    """
    global robot_pose_real, robot_pose_pixels, robot_pose_map_indices, robot_pose_lock, last_valid_pose_time
    global latest_camera_frame, latest_frame_lock
    print("Initializing Aruco Service and Camera/Stream...")

    aruco_service = ArucoCallbackService(
        working_dir=os.path.join(parent_dir, "working_dataset"),
        camera_yaml_path=os.path.join(parent_dir, "CV_app/camera.yaml"),
        calibration_json_path=CALIBRATION_FILE
    )
    if not aruco_service.initialized: print("FATAL: Failed to initialize ArucoCallbackService."); return

    cap = None
    if IP_WEBCAM_URL:
        print(f"Attempting to connect to IP Webcam stream: {IP_WEBCAM_URL}")
        stream_url = IP_WEBCAM_URL if IP_WEBCAM_URL.endswith('/video') else f"{IP_WEBCAM_URL}/video"
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"FATAL: Could not open IP Webcam stream at {stream_url}. Trying base URL...")
            cap = cv2.VideoCapture(IP_WEBCAM_URL)
            if not cap.isOpened(): print(f"FATAL: Could not open IP Webcam stream at base URL {IP_WEBCAM_URL} either."); return
        print("Successfully connected to IP Webcam stream.")
    else: print("FATAL: No IP_WEBCAM_URL set."); return

    # print_interval = 2.0; last_print_time = 0 # Keep commented out
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame. Retrying connection..."); cap.release(); time.sleep(1.0)
                if IP_WEBCAM_URL:
                     stream_url = IP_WEBCAM_URL if IP_WEBCAM_URL.endswith('/video') else f"{IP_WEBCAM_URL}/video"
                     cap = cv2.VideoCapture(stream_url)
                     if not cap.isOpened(): cap = cv2.VideoCapture(IP_WEBCAM_URL)
                if not cap.isOpened(): print("Error: Failed to reconnect."); time.sleep(5.0)
                continue

            with latest_frame_lock:
                latest_camera_frame = frame # Store the raw frame

            markers = aruco_service.detect_markers_in_frame(frame)
            found_robot = False
            current_real = None
            current_pixels = None
            current_map_indices = None # Store as (row, col)

            if markers:
                for marker in markers:
                    if marker.get('id') == ROBOT_ARUCO_ID:
                        aruco_grid_coords = marker.get('grid_coords') # Assumed (gx, gy)
                        angle_deg_aruco = marker.get('angle')         # Assumed (0=Down, CW+)

                        if aruco_grid_coords and angle_deg_aruco is not None:
                            gx, gy = aruco_grid_coords
                            row, col = gy, gx # ASSUMPTION: gx=col, gy=row
                            current_map_indices = (row, col)

                            px, py = map_indices_to_pixel(row, col)
                            if px is not None:
                                current_pixels = (px, py)
                                rx, ry = pixel_to_real(px, py)
                                if rx is not None:
                                    # CORRECTED Angle Calculation for Nav Frame (0=Right, CCW+)
                                    real_theta_rad = math.radians(angle_deg_aruco + 270.0)
                                    real_theta_rad_norm = normalize_angle(real_theta_rad)
                                    current_real = (rx, ry, real_theta_rad_norm)
                                    found_robot = True
                        break # Found robot, exit inner loop

            with robot_pose_lock:
                if found_robot and current_real:
                    robot_pose_real = current_real
                    robot_pose_pixels = current_pixels
                    robot_pose_map_indices = current_map_indices
                    last_valid_pose_time = time.time()

        except Exception as e:
            print(f"Error in camera thread: {e}")
            traceback.print_exc()
            time.sleep(1)

# --- CoAP Communication (Same as CTE Server) ---
async def send_coap_command(client_context, target_ip, command_data, seq_num):
    if not target_ip: print("Error: ESP32 IP unknown."); return False
    command_data["seq"] = seq_num # Ensure sequence number is in payload
    payload_bytes = json.dumps(command_data).encode('utf-8')
    request = aiocoap.Message(code=aiocoap.PUT, payload=payload_bytes, uri=f'coap://{target_ip}/command')
    print(f"--- CoAP TX (Command) --->\n  To: {target_ip}\n  Payload: {command_data}\n--------------------------")
    try:
        response = await asyncio.wait_for(client_context.request(request).response, timeout=1.0) # 1s timeout
        print(f"<--- CoAP RX (Ack) --- {response.code}")
        return True # Indicate success
    except asyncio.TimeoutError: print("!!! CoAP TX Error: Timeout waiting for ACK !!!"); return False
    except Exception as e: print(f"!!! CoAP TX Error: {e} !!!"); return False

class StatusResource(resource.Resource):
    async def render_put(self, request):
        global esp32_status, esp32_last_update
        payload = request.payload.decode('utf-8')
        print(f"<--- CoAP RX (Status) --- {payload}")
        try:
            status_data = json.loads(payload)
            esp32_status = status_data.get("status", "UNKNOWN")
            esp32_last_update = time.time()
        except json.JSONDecodeError: print("!!! CoAP RX Error: Invalid JSON status received !!!")
        except Exception as e: print(f"!!! CoAP RX Error: {e} !!!")
        return aiocoap.Message(code=aiocoap.CHANGED)

# --- Visualization Class (Same as CTE Server - plots MAP coords) ---
class ServerVisualizer:
    def __init__(self, grid_map, robot_radius_px):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.map_image = self.ax.imshow(grid_map, cmap='gray', origin='upper', vmin=0, vmax=1)
        self.robot_marker, = self.ax.plot([], [], 'bo', markersize=6, label='Robot (Pixel Pos)')
        self.robot_map_marker, = self.ax.plot([], [], 'rx', markersize=8, mew=2, label='Robot (Map Pos)')
        self.robot_dir = patches.Arrow(0, 0, 0, 0, width=robot_radius_px/2, fc='yellow', label='Direction')
        self.ax.add_patch(self.robot_dir); self.robot_dir.set_visible(False)
        self.path_line, = self.ax.plot([], [], 'r-', linewidth=1, label='Planned Path (Map)')
        self.target_point_marker, = self.ax.plot([], [], 'gx', markersize=10, mew=2, label='Target Waypoint (Map)')
        self.robot_trace_line, = self.ax.plot([], [], 'c-', linewidth=0.5, alpha=0.7, label='Trace (Pixels)')
        self.robot_trace_x = deque(maxlen=200); self.robot_trace_y = deque(maxlen=200)
        self.ax.set_title("Robot Navigation")
        self.ax.legend(fontsize='small', loc='upper right')
        grid_height, grid_width = grid_map.shape
        self.ax.set_xlim(0, grid_width)
        self.ax.set_ylim(grid_height, 0)
        self.ax.set_aspect('equal', adjustable='box')
        plt.show(block=False)

    def update(self, robot_pose_real, robot_pose_pixels, robot_pose_map_indices, path_map_indices, target_waypoint_idx_in_map_path, state_text):
        self.ax.set_title(f"Robot Navigation - State: {state_text}")
        # Robot Pixel Pose and Direction Arrow
        if robot_pose_pixels and robot_pose_real:
            robot_px, robot_py = robot_pose_pixels
            robot_theta_nav = robot_pose_real[2]
            self.robot_marker.set_data([robot_px], [robot_py]); self.robot_marker.set_visible(True)
            arrow_length_px = 20; arrow_width = 6
            dx_px = arrow_length_px * math.cos(robot_theta_nav)
            dy_px = -arrow_length_px * math.sin(robot_theta_nav)
            arrow_start_x, arrow_start_y = robot_px - dx_px*0.1, robot_py - dy_px*0.1
            if self.robot_dir in self.ax.patches: self.robot_dir.remove()
            self.robot_dir = patches.Arrow(arrow_start_x, arrow_start_y, dx_px*0.9, dy_px*0.9, width=arrow_width, fc='yellow', zorder=10)
            self.ax.add_patch(self.robot_dir); self.robot_dir.set_visible(True)
            self.robot_trace_x.append(robot_px); self.robot_trace_y.append(robot_py)
            self.robot_trace_line.set_data(list(self.robot_trace_x), list(self.robot_trace_y))
        else:
            self.robot_marker.set_visible(False)
            self.robot_dir.set_visible(False)
        # Robot Map Pose Marker
        if robot_pose_map_indices:
            map_row, map_col = robot_pose_map_indices
            self.robot_map_marker.set_data([map_col], [map_row]); self.robot_map_marker.set_visible(True)
        else:
            self.robot_map_marker.set_visible(False)
        # Path (Map Indices)
        if path_map_indices:
            path_cols, path_rows = zip(*[(col, row) for row, col in path_map_indices])
            self.path_line.set_data(path_cols, path_rows); self.path_line.set_visible(True)
        else:
            self.path_line.set_data([], []); self.path_line.set_visible(False)
        # Target Waypoint (Map Indices)
        if path_map_indices and 0 <= target_waypoint_idx_in_map_path < len(path_map_indices):
            target_row, target_col = path_map_indices[target_waypoint_idx_in_map_path]
            self.target_point_marker.set_data([target_col], [target_row]); self.target_point_marker.set_visible(True)
        else:
            self.target_point_marker.set_visible(False)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

# --- Math Helpers (Needed by Explicit Waypoint Controller) ---
def distance(pose1_xy, pose2_xy):
    """Calculate Euclidean distance between two points (x,y)."""
    return math.sqrt((pose1_xy[0] - pose2_xy[0])**2 + (pose1_xy[1] - pose2_xy[1])**2)

def calculate_heading(current_pose_xy, target_xy):
    """Calculate the world angle (radians, 0=Right, CCW+) from current (x,y) to target (x,y)."""
    dx = target_xy[0] - current_pose_xy[0]
    dy = target_xy[1] - current_pose_xy[1]
    return math.atan2(dy, dx)

# --- UI Functions (Same as CTE Server) ---
def display_grid_for_selection(grid_to_display, title, start_map_indices=None):
    """Displays the grid using Matplotlib and waits for user input."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_to_display, cmap='gray', origin='upper')
    ax.set_title(title)
    if start_map_indices:
        start_row, start_col = start_map_indices
        ax.plot(start_col, start_row, 'go', markersize=8, label='Detected Start')
        ax.legend()
    fig.canvas.draw_idle()
    plt.show(block=False)
    return fig, ax

def select_point_on_image_snapshot(image_to_display, title):
    """
    Allows the user to graphically select a point on a displayed camera snapshot.
    Returns the corresponding PIXEL coordinates (px, py) relative to the image.
    """
    print(f"\nPlease select the '{title}' on the camera snapshot...")
    if image_to_display is None or image_to_display.size == 0:
        print("Error: Invalid image provided for selection.")
        return None
    fig, ax = plt.subplots()
    if len(image_to_display.shape) == 3 and image_to_display.shape[2] == 3:
        ax.imshow(cv2.cvtColor(image_to_display, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image_to_display, cmap='gray')
    ax.set_title(title + " (Click on the image)")
    point_px = None
    while point_px is None:
        pts = fig.ginput(1, timeout=-1)
        if pts:
            point_px = pts[0]
            print(f"Selected Pixel Coords (on image): ({int(point_px[0])}, {int(point_px[1])})")
        else:
            print("No point selected (window closed?). Exiting selection.")
            plt.close(fig)
            return None
    plt.close(fig)
    return int(point_px[0]), int(point_px[1])

# --- >>> CONTROL LOOP (Explicit Waypoint Logic) <<< ---
async def control_loop(client_context):
    """Main state machine loop for explicit waypoint control."""
    global controller_state, command_start_time, command_seq, current_command_sent
    global current_path_real, current_waypoint_index, visualizer, ESP32_IP
    global planned_path_map_indices # Use MAP path for visualization
    global smoothed_robot_theta_rad, EMA_ALPHA

    print("Explicit Waypoint Control loop starting...")
    print("Control loop waiting for valid initial pose...")
    while True:
         # Get real pose for initialization
         current_pose_real_init, _, _ = get_current_pose_from_camera()
         if current_pose_real_init is not None:
             smoothed_robot_theta_rad = current_pose_real_init[2] # Initialize angle smoother
             print("Control loop received initial pose. Starting navigation logic.")
             break
         await asyncio.sleep(0.2)

    last_sent_command_type = "NONE" # Track last command type sent

    while True:
        start_time_loop = time.time()
        # Get all current pose representations
        current_pose_raw, current_pose_px, current_pose_map = get_current_pose_from_camera()

        # Apply Angle Smoothing to get the pose used for control
        current_pose = None # This is the smoothed REAL pose (rx, ry, theta_nav)
        if current_pose_raw is not None:
            raw_theta = current_pose_raw[2]
            if smoothed_robot_theta_rad is None: smoothed_robot_theta_rad = raw_theta
            else:
                # Apply EMA smoothing
                error = normalize_angle(raw_theta - smoothed_robot_theta_rad)
                smoothed_robot_theta_rad = normalize_angle(smoothed_robot_theta_rad + EMA_ALPHA * error)
            # Use smoothed angle with latest real coordinates
            current_pose = (current_pose_raw[0], current_pose_raw[1], smoothed_robot_theta_rad)

        # --- State Machine Logic ---
        command_to_send = None # Determine command within states

        if current_pose is None:
            # Pose lost, ensure robot stops
            if controller_state != "IDLE":
                 print("Pose lost, commanding STOP.")
                 controller_state = "IDLE"
                 command_to_send = "STOP" # Ensure stop command is sent
        elif not current_path_real or current_waypoint_index < 0 or current_waypoint_index >= len(current_path_real):
            # No path, path finished, or invalid index
            if controller_state != "IDLE":
                 print("No active path or path finished. Commanding STOP.")
                 controller_state = "IDLE"
                 command_to_send = "STOP" # Ensure stop command is sent
        else:
            # Valid pose and path exist
            target_waypoint = current_path_real[current_waypoint_index] # Target is (rx, ry)

            # --- IDLE State: Decide whether to align or move ---
            if controller_state == "IDLE":
                heading_to_target = calculate_heading(current_pose[:2], target_waypoint)
                heading_error = normalize_angle(heading_to_target - current_pose[2])

                if abs(heading_error) > ANGLE_THRESHOLD:
                    # Need to align
                    command = "TURN_RIGHT" if heading_error < 0 else "TURN_LEFT"
                    print(f"IDLE: Need to Align (Err: {math.degrees(heading_error):.1f}). Commanding {command}")
                    command_to_send = command
                    controller_state = "ALIGNING_TO_WAYPOINT"
                else:
                    # Already aligned, start moving
                    print(f"IDLE: Aligned. Commanding MOVE_FORWARD")
                    command_to_send = "MOVE_FORWARD"
                    controller_state = "MOVING_TO_WAYPOINT"

            # --- ALIGNING State: Check if turn finished ---
            elif controller_state == "ALIGNING_TO_WAYPOINT":
                heading_to_target = calculate_heading(current_pose[:2], target_waypoint)
                heading_error = normalize_angle(heading_to_target - current_pose[2])
                # Check if alignment is now within threshold
                if abs(heading_error) < ANGLE_THRESHOLD:
                    print("ALIGNING: Alignment Complete. Commanding STOP before move.")
                    command_to_send = "STOP"
                    controller_state = "IDLE" # Go back to IDLE to transition to MOVE
                else:
                    # Continue aligning - resend the same turn command? Or assume ESP32 continues?
                    # Let's assume ESP32 continues turning until STOP is sent.
                    # No new command needed unless we want to re-issue periodically.
                    # Keep state as ALIGNING.
                    pass # command_to_send remains None

            # --- MOVING State: Check if waypoint reached ---
            elif controller_state == "MOVING_TO_WAYPOINT":
                dist_to_wp = distance(current_pose[:2], target_waypoint)
                # Check if waypoint is reached
                if dist_to_wp < WAYPOINT_RADIUS:
                    print(f"MOVING: Waypoint {current_waypoint_index} Reached. Dist: {dist_to_wp:.3f}m. Commanding STOP.")
                    command_to_send = "STOP"
                    current_waypoint_index += 1 # Move to next waypoint index
                    # Check if path is complete
                    if current_waypoint_index >= len(current_path_real):
                        print("Final Waypoint Reached! Path Complete.")
                        current_path_real = None # Clear path
                        current_waypoint_index = -1
                        controller_state = "IDLE" # Finished
                    else:
                        # More waypoints exist
                        print(f"Proceeding to next waypoint {current_waypoint_index}")
                        controller_state = "IDLE" # Go back to IDLE to align/move to the NEW waypoint
                else:
                    # Continue moving - check alignment periodically?
                    heading_to_target = calculate_heading(current_pose[:2], target_waypoint)
                    heading_error = normalize_angle(heading_to_target - current_pose[2])
                    # If we drift too far off angle while moving, maybe stop and realign?
                    if abs(heading_error) > ANGLE_THRESHOLD * 1.5: # Use a wider threshold
                        print(f"MOVING: Angle drifted (Err: {math.degrees(heading_error):.1f}). Stopping to realign.")
                        command_to_send = "STOP"
                        controller_state = "IDLE" # Go back to IDLE to force realignment
                    else:
                        # Continue moving forward - resend MOVE_FORWARD? Or assume ESP32 continues?
                        # Let's assume ESP32 continues moving until STOP is sent.
                        pass # command_to_send remains None

        # --- Send Command if determined ---
        # Send STOP immediately if needed, otherwise send determined command only if it's new
        if command_to_send:
             # Always send STOP if determined
             # Only send other commands if they are different from the last one sent
            if command_to_send == "STOP" or command_to_send != last_sent_command_type:
                command_seq += 1
                cmd_data = {"cmd": command_to_send, "seq": command_seq}
                if command_to_send == "MOVE_FORWARD":
                    cmd_data["speed"] = MOTOR_FORWARD_SPEED
                elif command_to_send == "TURN_LEFT" or command_to_send == "TURN_RIGHT":
                    cmd_data["turn_speed"] = MOTOR_TURN_SPEED

                # Send command asynchronously
                asyncio.create_task(send_coap_command(client_context, ESP32_IP, cmd_data, command_seq))
                last_sent_command_type = command_to_send # Update last sent command type

        # Update Visualization (Using the visualizer from CTE baseline)
        if visualizer:
            target_idx_viz = current_waypoint_index if current_path_real else -1
            # Pass poses (real, pixel, map) and MAP path to visualizer
            visualizer.update(current_pose, current_pose_px, current_pose_map, planned_path_map_indices, target_idx_viz, controller_state)

        # Loop Delay
        elapsed = time.time() - start_time_loop
        await asyncio.sleep(max(0, CONTROL_LOOP_DELAY - elapsed))
# --- >>> END CONTROL LOOP <<< ---

# --- Main Execution (Same as CTE Server, except no initial_start_pose_real needed) ---
async def main():
    global visualizer, planned_path_map_indices, current_path_real, current_waypoint_index
    global latest_camera_frame, latest_frame_lock

    import matplotlib
    try: matplotlib.use('TkAgg'); print("Using TkAgg backend for Matplotlib.")
    except ImportError: print("TkAgg backend not available, using default Matplotlib backend.")

    if not calibration_valid: print("FATAL: Calibration data invalid. Exiting."); return

    # Load Grid Info (Same as CTE)
    GRID_INFO_FILE = os.path.join(parent_dir, 'working_dataset/grid/grid_info.json')
    frame_width = 0; frame_height = 0
    try:
        with open(GRID_INFO_FILE, 'r') as f: grid_info_data = json.load(f)
        frame_width = int(grid_info_data['resolution']['width'])
        frame_height = int(grid_info_data['resolution']['height'])
        print(f"Loaded Frame Resolution: {frame_width}x{frame_height} from {GRID_INFO_FILE}")
        if frame_width <= 0 or frame_height <= 0: raise ValueError("Invalid dimensions")
    except Exception as e:
        print(f"Warning: Could not load frame resolution from {GRID_INFO_FILE}: {e}. Using defaults (may affect display).")
        frame_width = 1920; frame_height = 1080

    # Load Occupancy Grid (Same as CTE)
    print(f"Server checking for grid file at: {OCCUPANCY_GRID_PATH}")
    if not os.path.exists(OCCUPANCY_GRID_PATH): print(f"FATAL ERROR: Cannot find grid file at '{OCCUPANCY_GRID_PATH}'"); return
    else: print("Server successfully verified grid file existence.")
    processed_grid_for_viz = None; grid_height, grid_width = 0, 0
    try:
        grid_raw = np.load(OCCUPANCY_GRID_PATH)
        grid_height, grid_width = grid_raw.shape
        processed_grid_for_viz = (grid_raw == 1).astype(np.uint8) # 1=free, 0=obstacle
        print(f"Loaded and processed grid. Shape=({grid_height}, {grid_width}).")
    except Exception as e: print(f"Error loading/processing occupancy grid: {e}"); return
    if grid_width <= 0 or grid_height <= 0: print("FATAL: Invalid grid dimensions."); return

    # Start Camera Thread (Same as CTE)
    cam_thread = Thread(target=camera_thread_function, args=(grid_width, grid_height), daemon=True)
    cam_thread.start(); print("Camera thread started. Waiting for camera..."); await asyncio.sleep(3.0)

    # Get Start Pose and Goal Point (Same as CTE)
    start_map_indices = None; selection_snapshot = None
    print("Waiting for first ArUco marker detection to set start point...")
    while start_map_indices is None:
        _, _, detected_map_indices = get_current_pose_from_camera()
        if detected_map_indices is not None:
            start_map_indices = detected_map_indices
            print(f"*** Robot detected! Start Point (MAP Indices row, col): {start_map_indices} ***")
            print("Attempting to capture snapshot...")
            snapshot_wait_start = time.time()
            while selection_snapshot is None:
                with latest_frame_lock:
                    if latest_camera_frame is not None:
                        selection_snapshot = latest_camera_frame.copy()
                        print("Snapshot captured for goal selection.")
                        break
                if time.time() - snapshot_wait_start > 2.0:
                    print("Warning: Timed out waiting for snapshot frame after pose detection.")
                    break
                await asyncio.sleep(0.05)
            break
        else: await asyncio.sleep(0.1)
    if selection_snapshot is None: print("FATAL: Could not get a camera frame for goal selection. Exiting."); return

    goal_xy_pixels = select_point_on_image_snapshot(selection_snapshot, "Select Goal Point on Camera Snapshot")
    if goal_xy_pixels is None: print("Goal selection cancelled or failed. Exiting."); return
    print(f"Goal Point Selected (CAMERA FRAME Pixels): {goal_xy_pixels}")
    goal_map_indices_float = pixel_to_map_indices(goal_xy_pixels[0], goal_xy_pixels[1])
    if goal_map_indices_float is None: print("FATAL: Could not convert goal pixel to map indices. Exiting."); return
    goal_map_indices = (int(round(goal_map_indices_float[0])), int(round(goal_map_indices_float[1])))
    print(f"Goal Point Converted (MAP Indices row, col): {goal_map_indices}")

    # Run A* Planning (Same as CTE)
    print("--- Starting A* Path Planning ---")
    start_map_indices_int = (int(round(start_map_indices[0])), int(round(start_map_indices[1])))
    planned_path_map_indices = run_astar_on_grid(
        OCCUPANCY_GRID_PATH,
        start_map_indices_int,
        goal_map_indices,
        ASTAR_SAFETY_DISTANCE_CELLS
    )
    if not planned_path_map_indices: print("FATAL: A* planning failed. Exiting."); return
    else: print(f"--- A* Planning Complete: Path with {len(planned_path_map_indices)} MAP points found. ---")

    # Convert MAP path to REAL path for control (Same as CTE)
    try:
        current_path_real = []
        for row, col in planned_path_map_indices:
            rx, ry = map_indices_to_real(row, col)
            if rx is None: raise ValueError("Map indices to real conversion failed for path.")
            current_path_real.append((rx, ry))
        current_waypoint_index = 0 if current_path_real else -1
        if current_path_real: print(f"Converted path to real coords for control. First waypoint: {current_path_real[0]}")
    except Exception as e: print(f"Error converting map path to real path: {e}"); return

    # Initialize Visualizer (Same as CTE)
    visualizer = ServerVisualizer(processed_grid_for_viz, ROBOT_RADIUS * PX_PER_CM * CM_PER_METER)

    # Initialize CoAP Server AND Client Contexts (Same as CTE)
    root = resource.Site(); root.add_resource(['status'], StatusResource())
    print(f"Starting CoAP server on {SERVER_IP}:{SERVER_PORT}")
    server_context = await aiocoap.Context.create_server_context(root, bind=(SERVER_IP, SERVER_PORT))
    client_context = await aiocoap.Context.create_client_context()
    print("CoAP server running.")

    # Run Control Loop (This will run the EXPLICIT WAYPOINT logic now)
    print("Starting EXPLICIT WAYPOINT control loop for navigation...")
    try:
        await control_loop(client_context)
    except KeyboardInterrupt: print("\nControl loop stopped by user.")
    except Exception as e: print(f"\nError in control loop: {e}"); traceback.print_exc()
    finally:
        # Shutdown sequence (Same as CTE)
        print("Shutting down server...")
        if ESP32_IP:
            try:
                print(f"Sending final STOP command to {ESP32_IP}...")
                cmd_data = {"cmd": "STOP", "seq": 9999}
                stop_task = asyncio.create_task(send_coap_command(client_context, ESP32_IP, cmd_data, 9999))
                await asyncio.wait_for(stop_task, timeout=1.5)
            except asyncio.TimeoutError: print("Timeout sending final STOP command.")
            except Exception as e_stop: print(f"Could not send final stop: {e_stop}")
        if 'server_context' in locals(): await server_context.shutdown()
        if 'client_context' in locals(): await client_context.shutdown()
        if visualizer and visualizer.fig: plt.close(visualizer.fig)
        print("Server shutdown complete.")

if __name__ == "__main__":
    asyncio.run(main()) 