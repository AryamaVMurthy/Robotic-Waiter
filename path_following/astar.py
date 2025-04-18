import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import heapq
import time

def load_grid(file_path):
    return np.load(file_path)

def select_point(grid, title):
    plt.imshow(grid, cmap='gray', origin='upper')
    plt.title(title)
    point = plt.ginput(1)[0]
    plt.close()
    return int(point[0]), int(point[1])

def visualize_path(grid, path):
    plt.imshow(grid, cmap='gray', origin='upper')
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, '-r')
    plt.show()

def visualize_approx_path(grid, approx_segments):
    """
    Visualizes the approximated (compressed) path on the grid.
    The path is provided as a list of segments, where each segment is a tuple:
    (start_x, start_y, end_x, end_y).
    
    Parameters:
      grid: 2D numpy array representing the grid.
      approx_segments: list of segments (start_x, start_y, end_x, end_y)
    """
    plt.imshow(grid, cmap='gray', origin='upper')
    
    if approx_segments:
        # Create a polyline from the segments.
        # Start with the first segment's starting point.
        xs, ys = [approx_segments[0][0]], [approx_segments[0][1]]
        for seg in approx_segments:
            xs.append(seg[2])
            ys.append(seg[3])
        plt.plot(xs, ys, '-r', linewidth=2)
        
    plt.title("Approximated Path")
    plt.show()

def visualize_safe_grid(safe_grid):
    plt.imshow(safe_grid, cmap='gray', origin='upper')
    plt.title("Safe Grid (1 = Safe, 0 = Unsafe)")
    plt.show()

def compute_safe_grid(grid, safe_dis):
    """Preprocess the grid to mark unsafe cells based on distance transform."""
    dist_transform = cv2.distanceTransform((grid * 255).astype(np.uint8), cv2.DIST_L2, 5)
    return (dist_transform > safe_dis).astype(np.uint8)

def square(x):
    return x*x

def heuristic(a, b):
    """Diagonal distance heuristic."""
    dx, dy = square(a[0] - b[0]), square(a[1] - b[1])
    return dx+dy

def point_line_distance(point, start, end):
    """
    Compute the perpendicular distance from 'point' to the line defined by 'start' and 'end'.
    """
    # If start and end are the same point, return the Euclidean distance
    if start == end:
        return ((point[0] - start[0])**2 + (point[1] - start[1])**2) ** 0.5
    # Calculate distance using the area method
    num = abs((end[1] - start[1]) * point[0] -
              (end[0] - start[0]) * point[1] +
              end[0] * start[1] - end[1] * start[0])
    den = ((end[1] - start[1])**2 + (end[0] - start[0])**2) ** 0.5
    return num / den

def approx_compress_path(path, epsilon=0.1):
    """
    Compresses a list of (x, y) points into a simplified version using
    the Ramer-Douglas-Peucker algorithm, with a tolerance epsilon.
    
    Parameters:
        path: List of (x, y) tuples representing the original path.
        epsilon: Tolerance threshold for approximation.
    
    Returns:
        A list of (x, y) points that approximates the original path.
    """
    if len(path) < 3:
        return path
    
    # Find the point with the maximum distance from the line between start and end.
    start, end = path[0], path[-1]
    max_distance = 0.0
    index = 0
    for i in range(1, len(path)-1):
        distance = point_line_distance(path[i], start, end)
        if distance > max_distance:
            max_distance = distance
            index = i

    # If the maximum distance is above the threshold, recursively simplify.
    if max_distance > epsilon:
        left_points = approx_compress_path(path[:index+1], epsilon)
        right_points = approx_compress_path(path[index:], epsilon)
        # Combine the results, omitting the duplicate point at the junction.
        return left_points[:-1] + right_points
    else:
        # Otherwise, approximate the entire segment as a straight line.
        return [start, end]

def convert_to_segments(compressed_points):
    """
    Converts a simplified list of (x, y) points into line segments.
    Each segment is represented as (start_x, start_y, end_x, end_y).
    """
    segments = []
    for i in range(len(compressed_points) - 1):
        start = compressed_points[i]
        end = compressed_points[i+1]
        segments.append((start[0], start[1], end[0], end[1]))
    return segments

def convert_to_waypoints(compressed_segments):
    """
    Converts compressed line segments into a sequence of (x, y, theta) waypoints
    where the robot either:
    1. Moves in a straight line (position changes, theta fixed)
    2. Rotates in place (position fixed, theta changes)
    
    Parameters:
        compressed_segments: List of (start_x, start_y, end_x, end_y) tuples
    
    Returns:
        List of (x, y, theta) tuples representing the robot's path
    """
    if not compressed_segments:
        return []
    
    waypoints = []
    
    for start_x, start_y, end_x, end_y in compressed_segments:
        waypoints.append((start_x,start_y))

    waypoints.append((compressed_segments[-1][2],compressed_segments[-1][3]))

    # Ensure the final waypoint has orientation 0
    return waypoints

def visualize_waypoints(grid, waypoints):
    """
    Visualizes the waypoints on the grid with arrows indicating orientation.
    
    Parameters:
        grid: 2D numpy array representing the grid
        waypoints: List of (x, y, theta) tuples
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='gray', origin='upper')
    
    if waypoints:
        # Extract x, y coordinates for the path
        xs = [wp[0] for wp in waypoints]
        ys = [wp[1] for wp in waypoints]
        
        # Plot the path
        plt.plot(xs, ys, '-b', linewidth=1)
        
    plt.title("Waypoints")
    plt.show()

def astar(grid, startt, goalt, safe_dis):
    """
    A* search on a grid with safe-distance enforcement.
    Finds path to goal or closest reachable point if goal is unreachable.
    grid: 2D numpy array where allowed cells are 1 and obstacles 0.
    startt: (x, y) tuple for the starting cell.
    goalt: (x, y) tuple for the target cell.
    safe_dis: Minimum safe distance from obstacles.

    Returns a tuple: (path, compressed_path, waypoints, safe_grid)
      path: list of (x, y) tuples from start to goal/closest node.
      compressed_path: Simplified segments of the path.
      waypoints: Waypoints derived from the compressed path.
      safe_grid: the preprocessed grid.
    """
    # Ensure inputs are tuples and use (x, y) consistently internally
    start = (startt[1],startt[0])
    goal = (goalt[1],goalt[0])
    rows, cols = grid.shape

    # Compute safe grid using distance transform.
    safe_grid = compute_safe_grid(grid, safe_dis)

    # Check if start node is valid
    # NumPy indexing uses [row, col] which corresponds to [y, x]
    if not (0 <= start[1] < rows and 0 <= start[0] < cols and safe_grid[start[1], start[0]] == 1):
         print(f"Start node {start} is outside grid bounds or unsafe.")
         return [], [], [], safe_grid

    # Check if goal node is valid initially (optional, could be unreachable later)
    goal_is_initially_safe = (0 <= goal[1] < rows and 0 <= goal[0] < cols and safe_grid[goal[1], goal[0]] == 1)
    if not goal_is_initially_safe:
         print(f"Warning: Goal node {goal} is outside grid bounds or initially unsafe.")
         # We will still proceed to find the closest reachable point.

    # Initialize the cost arrays.
    g_cost = np.full((rows, cols), np.inf, dtype=np.float32)
    # Use [y, x] for numpy indexing
    g_cost[start[1], start[0]] = 0

    # Parent pointer array: store (prev_x, prev_y) for each cell.
    came_from = np.full((rows, cols, 2), -1, dtype=np.int32)

    # --- Modification Start: Track closest node ---
    min_h_to_goal = heuristic(start, goal) # Heuristic needs (x,y) tuples
    closest_node_to_goal = start
    # --- Modification End ---

    # Open set as a priority queue. Each item is (f_cost, (x, y)).
    open_set = []
    # Push (f_cost, (x, y)) onto heap
    heapq.heappush(open_set, (min_h_to_goal, start)) # f_cost = g_cost (0) + h_cost

    iter = 0
    goal_found = False
    final_target_node = None

    # Neighbors are defined as (dx, dy)
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while open_set:
        iter += 1
        current_f, current = heapq.heappop(open_set) # current is (x, y)

        # Optimization: If we already found a path to this node with lower cost, skip
        # Check g_cost using [y, x] indexing
        if current_f > g_cost[current[1], current[0]] + heuristic(current, goal):
             continue # Skip if a better path to 'current' was already found and processed

        # --- Modification Start: Update closest node ---
        current_h = heuristic(current, goal)
        # Update if this node is heuristically closer to the goal than any node seen so far.
        # Ensure the node was actually reached (g_cost is finite) - implicit by being popped.
        if current_h < min_h_to_goal:
            min_h_to_goal = current_h
            closest_node_to_goal = current
        # --- Modification End ---

        # Check if we reached the *original* goal
        if current == goal:
            print(f"Goal {goal} reached! Iter = {iter}")
            final_target_node = goal
            goal_found = True
            break # Exit loop successfully

        # Current node coordinates (x, y)
        x, y = current
        # Explore the 4-connected neighbors (up, down, left, right).
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            # Check bounds.
            if 0 <= nx < cols and 0 <= ny < rows:
                # Only consider safe cells (use [y, x] for numpy access).
                if safe_grid[ny, nx] == 1:
                    tentative_g = g_cost[y, x] + 1 # uniform cost (1 per step)

                    # Check if this path to neighbor is better than any previous one
                    # Access g_cost using [y, x] indexing
                    if tentative_g < g_cost[ny, nx]:
                        g_cost[ny, nx] = tentative_g
                        # Store parent as (x, y)
                        came_from[ny, nx] = current # Store the current node (x, y) as parent
                        # Calculate f_cost for priority queue
                        f_cost = tentative_g + heuristic(neighbor, goal) # Heuristic needs (x,y)
                        heapq.heappush(open_set, (f_cost, neighbor)) # Push neighbor (x,y)

    # --- Loop finished ---

    if goal_found:
        target_node = final_target_node
        print(f"Path reconstruction target: Original Goal {target_node}")
    elif closest_node_to_goal is not None:
        # If goal wasn't found, use the closest node identified during search
        target_node = closest_node_to_goal
        print(f"Goal {goal} unreachable. Path reconstruction target: Closest Node {target_node}")
    else:
        # This case should ideally not be reached if start is valid.
        print(f"Goal {goal} unreachable, and no closest node found (start node might be isolated?).")
        return [], [], [], safe_grid


    # --- Reconstruct path from target_node to start using came_from ---
    path = []
    # Start reconstruction from the determined target node (x, y)
    cx, cy = target_node
    while (cx, cy) != (-1, -1): # Stop when we trace back past the start node
        # Check if the current node in reconstruction was actually reached
        # Use [y, x] for g_cost access
        if math.isinf(g_cost[cy, cx]) and (cx, cy) != start:
             print(f"Error during path reconstruction: Node {(cx, cy)} has infinite g_cost.")
             # Decide fallback: Return empty path or path up to this point?
             # Let's return empty path for safety.
             return [], [], [], safe_grid

        # Add the current node (x, y) to the path
        path.append((cx, cy))

        # Get the parent (px, py) using [y, x] indexing for came_from
        px, py = tuple(came_from[cy, cx])

        # If the parent is (-1, -1), it means we've reached the start node's marker
        if (px, py) == (-1, -1):
            # Ensure start node is in the path if it wasn't the target itself
            if (cx,cy) != start:
                 # This check might be redundant depending on exact came_from init
                 pass
            break

        # Move to the parent node for the next iteration
        cx, cy = px, py

        # Safety break
        if len(path) > rows * cols:
            print("Error: Path reconstruction exceeded maximum possible length.")
            return [], [], [], safe_grid

    if not path:
        print("Path reconstruction failed.")
        return [], [], [], safe_grid

    path.reverse() # Reverse the path to go from start -> target

    # --- Path Post-Processing (Using your existing functions) ---
    # Ensure these functions correctly handle (x, y) tuples
    simplified_points = approx_compress_path(path, epsilon=0.1) # Using your specified epsilon
    compressed_path = convert_to_segments(simplified_points)
    waypoints = convert_to_waypoints(compressed_path) # Assuming this takes segments

    print("Path lengths (to goal or closest point)")
    print(f"Original: {len(path)}")
    print(f"Compressed Segments: {len(compressed_path)}") # Assuming this is what was meant by 'Compressed'
    print(f"Waypoints: {len(waypoints)}")

    # Return the results in the specified format
    return path, compressed_path, waypoints, safe_grid

def time_astar(grid, start, goal, safe_dis):
    start_time = time.perf_counter()
    path, compressed_path, waypoints, safe_grid = astar(grid, start, goal, safe_dis)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    print(f"Astar Execution Time: {elapsed_time:.6f} seconds")

    return path, compressed_path, waypoints, safe_grid

def get_waypoints(grid, start, goal, safe_dis):
    path, compressed_path, waypoints, safe_grid = astar(grid, start, goal, safe_dis)
    waypoints = [(y, x) for x, y in path]
    return waypoints

import os
def run():
    
    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    file_path = os.path.abspath(os.path.join(parent_dir, 'working_dataset/grid/labeled_grid.npy'))
    grid = load_grid(file_path)
    
    start = select_point(grid, "Select Start Point")
    goal = select_point(grid, "Select Goal Point")
    safe_dis = 25

    path, compressed_path, waypoints, safe_grid = time_astar(grid, start, goal, safe_dis)
    visualize_safe_grid(safe_grid)
    visualize_path(grid, path)
    visualize_approx_path(grid, compressed_path)
    visualize_waypoints(grid, waypoints)
    
if __name__ == "__main__":
    run()
