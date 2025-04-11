import argparse
import os
import sys
import numpy as np
import cv2
import torch
import time
import json

# Set CUDA device explicitly
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# --- Remove sys.path manipulation if sam2 is installed via pip ---
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sam2_dir = os.path.join(script_dir, "sam2") # Assuming sam_callback is in the same dir
# sys.path.insert(0, sam2_dir)

# Import from local sam2 directory (or globally installed package)
try:
    from sam_callback import SAMCallbackService
except ImportError as e:
     print(f"Error importing SAMCallbackService: {e}")
     print("Ensure sam_callback.py is in the same directory or sam2 package is installed.")
     sys.exit(1)

# Removed load_grid_dimensions function as it's handled within SAMCallbackService

def pixel_to_grid(px, py, img_width, img_height, grid_cols, grid_rows):
    """Converts pixel coordinates to grid cell indices [col, row]."""
    if grid_cols is None or grid_rows is None or grid_cols <= 0 or grid_rows <= 0:
        # print("[WARN] Invalid grid dimensions for pixel_to_grid conversion.")
        return None # Return None if dimensions are invalid

    # Ensure img_width and img_height are valid
    if img_width <= 0 or img_height <= 0:
        return None

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

def main():
    parser = argparse.ArgumentParser(description="Test SAM2 segmentation with a single point")
    parser.add_argument("--px", type=int, required=True, help="X pixel coordinate (robot center)")
    parser.add_argument("--py", type=int, required=True, help="Y pixel coordinate (robot center)")
    parser.add_argument("--dist", type=float, required=True, help="Obstacle distance in cm")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (relative to script)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output grid (relative to script)")
    parser.add_argument("--model", type=str, default="facebook/sam2-hiera-tiny", help="Hugging Face model name")
    args = parser.parse_args()

    # Get the directory of this script (e.g., CV_app/)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths for image and output relative to script dir
    image_path = os.path.abspath(os.path.join(script_dir, args.image))
    output_path = os.path.abspath(os.path.join(script_dir, args.output))
    # Paths for SAMCallbackService constructor are relative to script_dir
    working_dir_rel = "../working_dataset"
    calibration_json_rel = "calibration.json"

    print(f"Script directory: {script_dir}")
    print(f"Image path: {image_path}")
    print(f"Output path: {output_path}")
    print(f"Working dir (relative): {working_dir_rel}")
    print(f"Calibration JSON (relative): {calibration_json_rel}")
    print(f"Using HF Model: {args.model}")

    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Failed to load image from {image_path}")
        return
    img_h, img_w = frame.shape[:2]
    print(f"Loaded image size: {img_w}x{img_h}")

    # Initialize SAMCallbackService
    print("Initializing SAMCallbackService...")
    sam_service = SAMCallbackService(
        hf_model_name=args.model,
        working_dir=working_dir_rel,          # Relative path from CV_app/
        calibration_json_path=calibration_json_rel # Relative path from CV_app/
    )

    if not sam_service.initialized:
        print("Error: Failed to initialize SAMCallbackService")
        return

    # Check if grid dimensions were loaded
    if sam_service.grid_cols is None or sam_service.grid_rows is None:
         print("Error: Grid dimensions not loaded by SAMCallbackService. Cannot proceed.")
         return

    # Convert pixel coordinates (assumed robot center) to grid coordinates
    robot_grid_coords = pixel_to_grid(args.px, args.py, img_w, img_h,
                                      sam_service.grid_cols, sam_service.grid_rows)

    if robot_grid_coords is None:
         print(f"Error converting pixel coordinates ({args.px}, {args.py}) to grid coordinates.")
         return

    print(f"Robot Pixel coordinates ({args.px}, {args.py}) -> Robot Grid coordinates {robot_grid_coords}") # Note: [col, row] format

    # Calculate obstacle direction (assuming obstacle is directly in front of the robot's pixel coords)
    # For simplicity, let's assume "front" means negative Y direction in pixel space
    # This might need adjustment based on actual robot orientation vs image
    obstacle_direction_vector = (0, -1) # Directly "up" in image coordinates

    # Update grid with obstacle
    print("Updating grid with obstacle...")
    start_update_time = time.time()
    updated_grid = sam_service.update_grid_with_obstacle(
        frame=frame,
        robot_grid_coords=robot_grid_coords, # Pass as [col, row]
        obstacle_distance_cm=args.dist,
        obstacle_direction_vector=obstacle_direction_vector
    )
    end_update_time = time.time()

    if updated_grid is not None:
        print(f"Grid update took {end_update_time - start_update_time:.3f} seconds.")
        print(f"Saving updated grid to {output_path}")
        try:
            np.save(output_path, updated_grid)
            print("Save successful!")
        except Exception as e:
            print(f"Error saving updated grid: {e}")
        print("Done!")
    else:
        print("Error: Failed to update grid")

if __name__ == '__main__':
    main()
