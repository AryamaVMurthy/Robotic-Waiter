#!/usr/bin/env python3
"""
Move–Measure–Correct Path Follower
Micro‑burst P‑control on distance and heading, closed‑loop via ArUco pose.
"""
import asyncio
import math
import numpy as np
import cv2
import aiocoap
import aiocoap.resource as resource
import json
import time
import os
import sys
import traceback
from threading import Thread, Lock
from collections import deque

# --- Path hack for imports ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# ---

# --- Reuse from custom_discrete.py ---
import custom_discrete as cd
from custom_discrete import (
    get_current_pose_from_camera,
    camera_thread_function,
    send_coap_command,
    StatusResource,
    normalize_angle,
    map_indices_to_real,
    SERVER_IP,
    SERVER_PORT,
    ESP32_IP,
    IP_WEBCAM_URL,
    start_map,
    goal_map,
    safe_cells,
    OCCUPANCY_GRID_PATH
)
# micro‑burst P‑controller params
MAX_PWM         = 200
KP_DIST         = MAX_PWM / 1.0        # PWM per meter
KP_HEAD         = MAX_PWM / math.pi    # PWM per radian
STEP_MS         = 50                   # ms per burst
WAYPOINT_RADIUS = 0.08                 # m to snap
CONTROL_DT      = STEP_MS/1000.0
path_real       = []

# --- Load calibration ---
CAL_FILE = os.path.join(parent_dir, 'CV_app/calibration.json')
try:
    data = json.load(open(CAL_FILE))
    PX_PER_CM = float(data['pixel_ratio']['value'])
    GRID_CELL_PX = float(data['grid']['cell_size_pixels'])
    GRID_CELL_CM = float(data['grid']['cell_size_cm'])
    assert PX_PER_CM>0 and GRID_CELL_PX>0
    calibration_valid = True
except Exception as e:
    print(f"Calibration load error: {e}")
    sys.exit(1)
CM_PER_M = 100.

# --- Global state ---
robot_pose_real        = (0.,0.,0.)
robot_pose_lock        = Lock()
latest_frame           = None
latest_frame_lock      = Lock()

# --- Helpers: pixel↔real and pose fetch ---
def pixel_to_real(px,py):
    x_cm = px / PX_PER_CM; y_cm = py / PX_PER_CM
    return x_cm/CM_PER_M, y_cm/CM_PER_M

def map_to_real(r,c):
    px,py = c*GRID_CELL_PX, r*GRID_CELL_PX
    return pixel_to_real(px,py)

# --- Control loop: Move–Measure–Correct ---
async def control_loop(client):
    seq,i = 0,0
    while i < len(path_real):
        pose = get_current_pose_from_camera()[0]
        if not pose: await asyncio.sleep(CONTROL_DT); continue
        x,y,th = pose
        tx,ty = path_real[i]
        dx,dy = tx-x, ty-y
        d_err = math.hypot(dx,dy)
        h_err = normalize_angle(math.atan2(dy,dx)-th)
        if d_err<WAYPOINT_RADIUS:
            i+=1; continue
        v_pwm = max(-MAX_PWM,min(MAX_PWM, KP_DIST*d_err))
        w_pwm = max(-MAX_PWM,min(MAX_PWM, KP_HEAD*h_err))
        l = int(max(-MAX_PWM,min(MAX_PWM, v_pwm - w_pwm)))
        r = int(max(-MAX_PWM,min(MAX_PWM, v_pwm + w_pwm)))
        seq+=1
        await send_coap_command(client, ESP32_IP,
            {'left_speed':l,'right_speed':r,'duration_ms':STEP_MS}, seq)
        await asyncio.sleep(CONTROL_DT)
    # final stop
    await send_coap_command(client, ESP32_IP,
        {'left_speed':0,'right_speed':0,'duration_ms':STEP_MS}, seq+1)

# --- Main ---
async def main():
    global path_real
    # start camera
    grid = np.load(OCCUPANCY_GRID_PATH)
    Thread(target=lambda: camera_thread_function(grid.shape[1], grid.shape[0]),daemon=True).start(); await asyncio.sleep(1.0)
    # plan
    raw = cd.get_waypoints(grid, start_map, goal_map, safe_cells)
    comp = cd.approx_compress_path(raw,epsilon=0.5)
    path_real = [map_indices_to_real(r,c) for r,c in comp]
    # coap
    root=resource.Site(); root.add_resource(['status'],StatusResource())
    server = await aiocoap.Context.create_server_context(root,(SERVER_IP,SERVER_PORT))
    client = await aiocoap.Context.create_client_context()
    # drive
    await control_loop(client)
    await server.shutdown(); await client.shutdown()

if __name__=='__main__':
    asyncio.run(main())