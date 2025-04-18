#!/usr/bin/env python3
"""
Move–Measure–Correct Path Follower
Micro‑burst P‑control on distance and heading, closed‑loop via ArUco pose.
"""
import asyncio
import math
import json
import time
import os
import sys
import numpy as np
import cv2
import aiocoap
import aiocoap.resource as resource
from threading import Thread, Lock
import matplotlib.pyplot as plt
import astar
from CV_app.aruco_callback import ArucoCallbackService
from simulation import normalize_angle

# --- Path hack for imports ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# ---

# --- Load calibration ---
CAL_FILE = os.path.join(parent_dir, 'CV_app/calibration.json')
try:
    with open(CAL_FILE,'r') as f:
        data = json.load(f)
    PX_PER_CM    = float(data['pixel_ratio']['value'])
    GRID_CELL_PX = float(data['grid']['cell_size_pixels'])
    GRID_CELL_CM = float(data['grid']['cell_size_cm'])
    assert PX_PER_CM>0 and GRID_CELL_PX>0
except Exception as e:
    print(f"Calibration load error: {e}")
    sys.exit(1)
CM_PER_M = 100.0

# --- Network & controller params ---
SERVER_IP      = '192.168.220.209'
SERVER_PORT    = 5683
ESP32_IP       = '192.168.220.216'
IP_WEBCAM_URL  = 'http://192.168.220.245:8080'
ROBOT_ARUCO_ID = 0

# micro‑burst P‑controller gains & timing
MAX_PWM         = 200
KP_DIST         = MAX_PWM / 1.0        # PWM per meter
KP_HEAD         = MAX_PWM / math.pi    # PWM per radian
STEP_MS         = 50                   # ms per burst
WAYPOINT_RADIUS = 0.08                 # m to snap
CONTROL_DT      = STEP_MS / 1000.0

# A* parameters
ASTAR_SAFETY_DISTANCE_REAL = 10  # cm
ASTAR_SAFETY_DISTANCE_CELLS = int(round(ASTAR_SAFETY_DISTANCE_REAL / GRID_CELL_CM))
OCCUPANCY_GRID_PATH        = os.path.join(parent_dir,'working_dataset','grid','labeled_grid.npy')

# --- Global state ---
robot_pose_real      = (0.0,0.0,0.0)
last_valid_pose_time = 0.0
robot_pose_lock      = Lock()

# --- Coordinate conversions ---
def pixel_to_map_indices(px, py):
    row = int(py / GRID_CELL_PX)
    col = int(px / GRID_CELL_PX)
    return row, col

def map_indices_to_real(row, col):
    px = col * GRID_CELL_PX
    py = row * GRID_CELL_PX
    x_cm = px / PX_PER_CM
    y_cm = py / PX_PER_CM
    return x_cm/CM_PER_M, y_cm/CM_PER_M

# --- Camera thread updates robot_pose_real ---
def camera_thread():
    global robot_pose_real, last_valid_pose_time
    service = ArucoCallbackService(
        working_dir=os.path.join(parent_dir,'working_dataset'),
        camera_yaml_path=os.path.join(parent_dir,'CV_app/camera.yaml'),
        calibration_json_path=CAL_FILE)
    if not service.initialized:
        print("Aruco init failed"); return
    cap = cv2.VideoCapture(IP_WEBCAM_URL+'/video')
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        markers = service.detect_markers_in_frame(frame)
        for m in markers:
            if m.get('id') == ROBOT_ARUCO_ID:
                gx, gy = m['grid_coords']
                ang    = m.get('angle',0)
                theta  = normalize_angle(math.radians(270-ang))
                rx, ry = map_indices_to_real(gy, gx)
                with robot_pose_lock:
                    robot_pose_real      = (rx, ry, theta)
                    last_valid_pose_time = time.time()
                break

# --- Pose fetch ---
def get_current_pose():
    with robot_pose_lock:
        pose = robot_pose_real
        t0   = last_valid_pose_time
    if time.time() - t0 > 1.0:
        return None
    return pose

# --- CoAP send & status resource ---
async def send_coap(client, ip, cmd, seq):
    msg = dict(cmd); msg['seq'] = seq
    req = aiocoap.Message(code=aiocoap.PUT,
                          uri=f"coap://{ip}/command",
                          payload=json.dumps(msg).encode())
    try:
        r  = await client.request(req).response
        js = json.loads(r.payload.decode())
        return js.get('status')=='DONE'
    except:
        return False

class StatusResource(resource.Resource):
    async def render_put(self, req):
        return aiocoap.Message(code=aiocoap.CHANGED)

# --- Control loop: Move–Measure–Correct ---
async def control_loop(client, path_real):
    seq = 0
    i   = 0
    while i < len(path_real):
        pose = get_current_pose()
        if not pose:
            await asyncio.sleep(CONTROL_DT)
            continue
        x, y, th = pose
        tx, ty   = path_real[i]
        dx, dy   = tx-x, ty-y
        d_err    = math.hypot(dx, dy)
        h_err    = normalize_angle(math.atan2(dy,dx)-th)
        if d_err < WAYPOINT_RADIUS:
            i += 1
            continue
        v_pwm = max(-MAX_PWM, min(MAX_PWM, KP_DIST*d_err))
        w_pwm = max(-MAX_PWM, min(MAX_PWM, KP_HEAD*h_err))
        l = int(max(-MAX_PWM, min(MAX_PWM, v_pwm-w_pwm)))
        r = int(max(-MAX_PWM, min(MAX_PWM, v_pwm+w_pwm)))
        seq += 1
        await send_coap(client, ESP32_IP,
            {'left_speed':l,'right_speed':r,'duration_ms':STEP_MS}, seq)
        await asyncio.sleep(CONTROL_DT)
    await send_coap(client, ESP32_IP,
        {'left_speed':0,'right_speed':0,'duration_ms':STEP_MS}, seq+1)

# --- Main ---
async def main():
    # start camera thread
    Thread(target=camera_thread,daemon=True).start()
    await asyncio.sleep(1.0)

    # load occupancy grid and choose start/goal
    grid = astar.load_grid(OCCUPANCY_GRID_PATH)
    sx, sy = astar.select_point(grid,"Select START point")
    gx, gy = astar.select_point(grid,"Select GOAL point")
    start = pixel_to_map_indices(sx, sy)
    goal  = pixel_to_map_indices(gx, gy)

    # plan path
    raw = astar.get_waypoints(grid, start, goal, ASTAR_SAFETY_DISTANCE_CELLS)
    comp = astar.approx_compress_path(raw, epsilon=0.5)
    path_real = [map_indices_to_real(r,c) for r,c in comp]

    # coap server & client
    root   = resource.Site(); root.add_resource(['status'],StatusResource())
    server = await aiocoap.Context.create_server_context(root,(SERVER_IP,SERVER_PORT))
    client = await aiocoap.Context.create_client_context()

    # drive
    await control_loop(client, path_real)
    await server.shutdown(); await client.shutdown()

if __name__=='__main__':
    asyncio.run(main())