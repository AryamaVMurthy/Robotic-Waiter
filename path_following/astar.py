
import numpy as np
import matplotlib.pyplot as plt
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
    grid: 2D numpy array where allowed cells are 1 and obstacles 0.
    start: (x, y) tuple for the starting cell.
    goal: (x, y) tuple for the target cell.
    safe_dis: Minimum safe distance from obstacles.
    
    Returns a tuple: (path, safe_grid)
      path: list of (x, y) tuples from start to goal (empty list if no path exists).
      safe_grid: the preprocessed grid where cells within safe_dis of an obstacle are blocked.
    """
    start = startt[1], startt[0]
    goal = goalt[1], goalt[0]
    rows, cols = grid.shape
    
    # Compute safe grid using distance transform.
    safe_grid = compute_safe_grid(grid, safe_dis)

    # If start or goal are unsafe, we cannot find a path.
    if safe_grid[start[1], start[0]] == 0 or safe_grid[goal[1], goal[0]] == 0:
        return [], [], [], safe_grid  # Modified return to include waypoints

    # Initialize the cost arrays.
    g_cost = np.full((rows, cols), np.inf, dtype=np.float32)
    g_cost[start[1], start[0]] = 0

    # Parent pointer array: store (prev_x, prev_y) for each cell.
    came_from = np.full((rows, cols, 2), -1, dtype=np.int32)

    # Open set as a priority queue. Each item is (f_cost, (x, y)).
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    iter = 0
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while open_set:
        iter += 1
        current_f, current = heapq.heappop(open_set)
        if current == goal:
            print(f"Iter = {iter}")

            # Reconstruct path from goal to start using came_from.
            length = int(g_cost[goal[1], goal[0]]) + 1
            path = [None] * length

            # Fill the path list in reverse order.
            i = length - 1
            cx, cy = goal
            while (cx, cy) != (-1, -1) and i >= 0:
                path[i] = (cx, cy)
                i -= 1
                cx, cy = tuple(came_from[cy, cx])

            simplified_points = approx_compress_path(path,epsilon=0.1)
            compressed_path = convert_to_segments(simplified_points)
            waypoints = convert_to_waypoints(compressed_path)

            print("Path lengths")
            print(f"Original: {len(path)}")
            print(f"Compressed: {len(compressed_path)}")
            print(f"Waypoints: {len(waypoints)}")

            return path, compressed_path, waypoints, safe_grid

        x, y = current
        # Explore the 4-connected neighbors (up, down, left, right).
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            # Check bounds.
            if 0 <= nx < cols and 0 <= ny < rows:
                # Only consider safe cells.
                if safe_grid[ny, nx] == 1:
                    tentative_g = g_cost[y, x] + 1  # uniform cost (1 per step)
                    if tentative_g < g_cost[ny, nx]:
                        g_cost[ny, nx] = tentative_g
                        came_from[ny, nx] = (x, y)
                        f_cost = tentative_g + heuristic((nx, ny), goal)
                        heapq.heappush(open_set, (f_cost, (nx, ny)))
                        
    # If we exit the loop, no path was found.
    return [], [], [], safe_grid  # Modified return to include waypoints

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
