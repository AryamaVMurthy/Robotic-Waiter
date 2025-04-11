import numpy as np
import matplotlib.pyplot as plt
import cv2
import heapq
import time
import os
import traceback # Added for detailed error printing

# --- Grid Loading ---
def load_grid(file_path):
    """
    Loads the occupancy grid from a .npy file.
    Assumes the grid contains labels where only label '1' (floor) is traversable.
    Converts the grid to binary: 1 for safe (floor), 0 for unsafe (obstacles/other labels).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Grid file not found: {file_path}")
    print(f"Loading grid from: {file_path}")
    grid = np.load(file_path)
    safe_value = 1
    binary_grid = (grid == safe_value).astype(np.uint8)
    print(f"Grid loaded and processed (1=safe, 0=unsafe). Shape: {binary_grid.shape}")
    return binary_grid

# --- Point Selection ---
# --- >>> Returns PIXEL coordinates <<< ---
def select_point(grid_to_display, title):
    """
    Allows the user to graphically select a point on the grid display.
    Returns the corresponding PIXEL coordinates (px, py).
    """
    print(f"\nPlease select the '{title}' on the map...")
    fig, ax = plt.subplots()
    ax.imshow(grid_to_display, cmap='gray', origin='upper') # Display the pixel grid
    ax.set_title(title + " (Click on the map)")
    point_px = None
    while point_px is None:
        pts = fig.ginput(1, timeout=-1) # Wait indefinitely for one click
        if pts:
            point_px = pts[0] # Clicked point in pixel coordinates (px, py)
            print(f"Selected Pixel Coords: ({int(point_px[0])}, {int(point_px[1])})")
        else:
            # This case might happen if the plot window is closed before clicking
            print("No point selected (window closed?). Exiting selection.")
            plt.close(fig)
            return None # Indicate failure or cancellation
    plt.close(fig)
    # Return pixel coordinates directly
    return int(point_px[0]), int(point_px[1])
# --- >>> END REVERSION <<< ---

# --- Visualization Helpers ---
def visualize_path(grid, path, title="A* Path"):
    """Visualizes the raw A* path on the grid."""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='gray', origin='upper')
    if path:
        path_y, path_x = zip(*path) # Note: A* returns (row, col), plot needs (x=col, y=row)
        plt.plot(path_x, path_y, '-r', linewidth=1.5, label='A* Path')
    plt.title(title)
    plt.legend()
    plt.show()

def visualize_approx_path(grid, approx_segments, title="Approximated Path"):
    """Visualizes the approximated (compressed) path segments on the grid."""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='gray', origin='upper')
    if approx_segments:
        # Plot each segment
        for i, seg in enumerate(approx_segments):
            start_x, start_y, end_x, end_y = seg
            label = 'Compressed Path' if i == 0 else ""
            plt.plot([start_x, end_x], [start_y, end_y], '-g', linewidth=2, label=label)
            plt.plot(start_x, start_y, 'go') # Mark start point of segment
        # Mark the final endpoint
        if approx_segments:
            plt.plot(approx_segments[-1][2], approx_segments[-1][3], 'go')
    plt.title(title)
    if approx_segments: plt.legend()
    plt.show()

def visualize_waypoints(grid, waypoints, title="Waypoints with Orientation"):
    """Visualizes the waypoints on the grid with arrows indicating orientation."""
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='gray', origin='upper')
    if waypoints:
        xs = [wp[0] for wp in waypoints]
        ys = [wp[1] for wp in waypoints]
        plt.plot(xs, ys, '-b', linewidth=1, label='Waypoints Path')
        # Plot arrows for orientation
        arrow_scale = 5 # Adjust arrow size as needed
        for i, (x, y, theta) in enumerate(waypoints):
            # Only draw arrow if orientation changed or it's the last point
            if i > 0 and waypoints[i-1][2] != theta or i == len(waypoints) - 1:
                 dx = np.cos(theta) * arrow_scale
                 dy = np.sin(theta) * arrow_scale
                 # Plotting Y is inverted because origin='upper'
                 plt.arrow(x, y, dx, -dy, head_width=arrow_scale*0.6, head_length=arrow_scale*0.4,
                           fc='red', ec='red', alpha=0.7, zorder=10)
        plt.plot(xs, ys, 'bo', markersize=3) # Mark waypoint locations
    plt.title(title)
    if waypoints: plt.legend()
    plt.show()

def visualize_safe_grid(safe_grid, title="Safe Grid", start_rc=None, goal_rc=None):
    """Visualizes the safe grid, optionally plotting start/goal points (row, col)."""
    plt.figure(figsize=(8, 8))
    plt.imshow(safe_grid, cmap='gray', origin='upper')
    plt.title(title)
    if start_rc:
        plt.plot(start_rc[1], start_rc[0], 'go', markersize=8, label='Start') # Plot col, row
    if goal_rc:
        plt.plot(goal_rc[1], goal_rc[0], 'ro', markersize=8, label='Goal')   # Plot col, row
    if start_rc or goal_rc:
        plt.legend()
    plt.show()

# --- Core A* and Path Processing Logic ---

# --- >>> MODIFIED: Uses erosion based on CELL distance <<< ---
def compute_safe_grid(grid, safe_dis_cells):
    """
    Preprocesses the grid using erosion to create a safety margin in grid cells.
    Assumes input grid has 1 for safe (floor) and 0 for obstacle.
    Returns a safe_grid where 1 is safe and 0 is unsafe.
    """
    
    if safe_dis_cells <= 0: return grid # No inflation needed
    print(f"Computing safe grid with safety distance: {safe_dis_cells} cells.")

    # Erode the safe area (1s). Obstacles (0s) effectively expand.
    kernel_size = 2 * int(round(safe_dis_cells)) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Input to erode needs safe area to be 1.
    safe_grid = cv2.erode(grid.astype(np.uint8), kernel, iterations=1)

    print(f"Safe grid computed using erosion. Unique values: {np.unique(safe_grid)}")
    return safe_grid
# --- >>> END Modification <<< ---

def heuristic(a, b):
    """
    Octile distance heuristic for an 8-connectivity grid.
    a and b are (row, col) tuples (grid coordinates).
    """
    dx = abs(a[1] - b[1]) # Difference in columns (grid_x)
    dy = abs(a[0] - b[0]) # Difference in rows (grid_y)
    cost_straight = 1
    cost_diagonal = np.sqrt(2)
    return cost_straight * (dx + dy) + (cost_diagonal - 2 * cost_straight) * min(dx, dy)

def point_line_distance(point, start, end):
    """Compute distance from 'point' to line segment 'start'-'end'. Assumes points are (row, col)."""
    px, py = point[1], point[0] # Convert to (x, y) for calculation
    sx, sy = start[1], start[0]
    ex, ey = end[1], end[0]
    line_sq_len = (ex - sx)**2 + (ey - sy)**2
    if line_sq_len == 0: return ((px - sx)**2 + (py - sy)**2)**0.5
    t = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / line_sq_len
    t = max(0, min(1, t))
    proj_x = sx + t * (ex - sx); proj_y = sy + t * (ey - sy)
    dist = ((px - proj_x)**2 + (py - proj_y)**2)**0.5
    return dist

def approx_compress_path(path, epsilon):
    """Compresses path (list of (row, col)) using RDP."""
    if len(path) < 3: return path
    start, end = path[0], path[-1]
    max_distance = 0.0; index = 0
    for i in range(1, len(path) - 1):
        distance = point_line_distance(path[i], start, end)
        if distance > max_distance: max_distance = distance; index = i
    if max_distance > epsilon:
        left_points = approx_compress_path(path[:index+1], epsilon)
        right_points = approx_compress_path(path[index:], epsilon)
        return left_points[:-1] + right_points
    else: return [start, end]

def convert_to_segments(compressed_points):
    """Converts list of points (row, col) to segments (start_col, start_row, end_col, end_row)."""
    segments = []
    for i in range(len(compressed_points) - 1):
        start_row, start_col = compressed_points[i]
        end_row, end_col = compressed_points[i+1]
        # Store as (x1, y1, x2, y2) for consistency with plotting/usage
        segments.append((start_col, start_row, end_col, end_row))
    return segments

def convert_to_waypoints(compressed_segments):
    """Converts segments (x1, y1, x2, y2) to waypoints (x, y, theta)."""
    if not compressed_segments:
        return []

    waypoints = []
    start_x, start_y = compressed_segments[0][0], compressed_segments[0][1]
    # Initial orientation can be arbitrary, often set to 0 or towards first segment
    # Let's calculate initial theta based on the first segment
    first_dx = compressed_segments[0][2] - start_x
    first_dy = compressed_segments[0][3] - start_y
    current_theta = np.arctan2(first_dy, first_dx) # Angle relative to +X axis

    waypoints.append((start_x, start_y, current_theta))

    for i, (seg_start_x, seg_start_y, seg_end_x, seg_end_y) in enumerate(compressed_segments):
        # Calculate the orientation required for this segment
        dx = seg_end_x - seg_start_x
        dy = seg_end_y - seg_start_y
        # Avoid division by zero or atan2(0,0) if segment has zero length
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
             segment_theta = current_theta # Keep previous theta if segment is just a point
        else:
             segment_theta = np.arctan2(dy, dx)

        # If orientation needs to change, add a rotation waypoint *at the start* of the segment
        # Check angle difference carefully using normalized angles if necessary
        angle_diff = abs(segment_theta - current_theta)
        # Normalize angle difference if needed, e.g., using (a - b + np.pi) % (2 * np.pi) - np.pi
        if angle_diff > 1e-4: # Use a small threshold for floating point comparison
            # Add rotation waypoint at the start point of this segment
            waypoints.append((seg_start_x, seg_start_y, segment_theta))
            current_theta = segment_theta

        # Add the movement waypoint at the end of the segment
        waypoints.append((seg_end_x, seg_end_y, current_theta))

    # Remove potential duplicate points (e.g., if rotation happened at the same spot)
    unique_waypoints = []
    if waypoints:
        unique_waypoints.append(waypoints[0])
        for i in range(1, len(waypoints)):
            # Only add if position or theta changed significantly
            pos_diff = ((waypoints[i][0] - waypoints[i-1][0])**2 + (waypoints[i][1] - waypoints[i-1][1])**2)**0.5
            theta_diff = abs(waypoints[i][2] - waypoints[i-1][2]) # Simple diff ok here? Maybe normalize.
            if pos_diff > 1e-4 or theta_diff > 1e-4:
                 unique_waypoints.append(waypoints[i])

    return unique_waypoints

# --- >>> MODIFIED: Astar function uses MAP indices (row, col) <<< ---
def astar(original_grid, safe_grid, start_map_indices, goal_map_indices):
    """
    A* search on a grid using a pre-computed safe_grid.
    Accepts start/goal as (row, col).
    Returns raw path as list of (row, col).
    """
    rows, cols = original_grid.shape
    start_row, start_col = start_map_indices
    goal_row, goal_col = goal_map_indices

    # Use (row, col) directly
    start = (int(round(start_row)), int(round(start_col)))
    goal = (int(round(goal_row)), int(round(goal_col)))

    # Safety Checks (using row, col)
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
         print(f"Error: Start point {start_map_indices} (row={start[0]}, col={start[1]}) is outside grid bounds ({rows}x{cols}).")
         return []
    if safe_grid[start[0], start[1]] == 0:
        print(f"Error: Start point {start_map_indices} is in an unsafe area (value 0 in safe_grid).")
        return []
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
         print(f"Error: Goal point {goal_map_indices} (row={goal[0]}, col={goal[1]}) is outside grid bounds ({rows}x{cols}).")
         return []
    if safe_grid[goal[0], goal[1]] == 0:
        print(f"Error: Goal point {goal_map_indices} is in an unsafe area (value 0 in safe_grid).")
        return []

    g_cost = np.full((rows, cols), np.inf, dtype=np.float32)
    g_cost[start] = 0
    came_from = {}
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    neighbor_costs = [1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]

    path_found = False
    nodes_expanded = 0
    while open_set:
        nodes_expanded += 1
        current_f, current = heapq.heappop(open_set)

        if current_f > g_cost[current] + heuristic(current, goal) + 1e-9: continue
        if current == goal: path_found = True; break

        for i, (dr, dc) in enumerate(neighbors):
            neighbor = (current[0] + dr, current[1] + dc)
            n_row, n_col = neighbor
            if 0 <= n_row < rows and 0 <= n_col < cols and safe_grid[n_row, n_col] == 1:
                move_cost = neighbor_costs[i]
                tentative_g = g_cost[current] + move_cost
                if tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_cost = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_cost, neighbor))

    raw_path_map_indices = []
    print(f"A* search completed. Nodes expanded: {nodes_expanded}")

    if path_found:
        print("Path found! Reconstructing...")
        curr = goal
        while curr != start:
            raw_path_map_indices.append(curr)
            if curr not in came_from: print("Error: Path reconstruction failed."); return []
            curr = came_from[curr]
        raw_path_map_indices.append(start)
        raw_path_map_indices.reverse()

        print(f"Path length: {len(raw_path_map_indices)} points.")
        return raw_path_map_indices
    else:
        print("No path found.")
        return []
# --- >>> END Astar Modification <<< ---

def time_astar(original_grid, safe_grid, start_map_indices, goal_map_indices):
    """Times the A* algorithm and returns its results."""
    print("\nStarting A* planning...")
    start_time = time.perf_counter()
    # Pass map indices directly to astar now
    raw_path_map_indices = astar(original_grid, safe_grid, start_map_indices, goal_map_indices)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"A* Execution Time: {elapsed_time:.6f} seconds")
    return raw_path_map_indices

# --- Main Wrapper Function ---
# --- >>> MODIFIED: Accepts map indices (row, col), cell distance <<< ---
def run_astar_on_grid(grid_file_path, start_map_indices, goal_map_indices, safety_distance_cells=3, compression_epsilon=None): # Compression not implemented here
    """
    Loads grid, computes safe grid using cell distance, runs A* from provided start/goal map indices (row, col),
    returns path in map indices [(row, col), ...].
    """
    try:
        grid = load_grid(grid_file_path)
        safe_grid = compute_safe_grid(grid, safety_distance_cells) # Use cell distance
        print(f"Running A* with Start Map Indices: {start_map_indices}, Goal Map Indices: {goal_map_indices}")
        raw_path_map_indices = time_astar(
            grid, safe_grid, start_map_indices, goal_map_indices
        )
        print("Returned from time_astar.")
        # Optional: Compression (would need adaptation)
        return raw_path_map_indices # Return path in map indices (row, col)
    except FileNotFoundError as e: print(f"Error: {e}"); return []
    except Exception as e: print(f"An unexpected error occurred during A* planning: {e}"); traceback.print_exc(); return []
# --- >>> END Modification <<< ---

# --- Example Usage ---
if __name__ == "__main__":
    print("Running astar_planner.py directly requires manual start/goal map indices (row, col) now.")
    # Example:
    # script_dir = os.path.dirname(__file__)
    # grid_path = os.path.join(script_dir, "../working_dataset/grid/labeled_grid.npy") # Adjust path
    # safety_cells = 3
    # manual_start_indices = (10, 10) # Example start indices (row, col)
    # manual_goal_indices = (50, 50) # Example goal indices (row, col)
    # planned_path_indices = run_astar_on_grid(grid_path, manual_start_indices, manual_goal_indices, safety_cells)
    # if planned_path_indices:
    #     print("Standalone test successful.")
    #     grid_for_viz = load_grid(grid_path)
    #     visualize_path(grid_for_viz, planned_path_indices) # visualize_path expects (row, col)
    # else:
    #     print("Standalone test failed.") 