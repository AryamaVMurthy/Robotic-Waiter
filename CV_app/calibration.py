import cv2
import numpy as np
import glob
import sys # Import sys to handle potential errors if not enough images found

# --- Configuration ---
chessboard_size = (10, 15)  # Inner corners (width, height)
square_size = 25.0         # Size of a square side in your chosen units (e.g., mm)
image_folder_path = 'cal/*.jpg' # Path pattern to your calibration images
# ---------------------

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1*sq_size,0,0), ..., ((w-1)*sq_size,(h-1)*sq_size,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size # Scale by square size

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Load all images
images = glob.glob(image_folder_path)

if not images:
    print(f"Error: No images found matching pattern '{image_folder_path}'")
    sys.exit(1)

print(f"Found {len(images)} images. Processing...")

image_size = None # To store image dimensions

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Could not read image {fname}. Skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1] # Get (width, height)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        print(f"  Corners found in {fname}")
        objpoints.append(objp)
        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Optional: Draw and display the corners (can be slow)
        # cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        # cv2.imshow('Calibration Check', img)
        # cv2.waitKey(50) # Show for a short time
    else:
        print(f"  Corners *not* found in {fname}")

# cv2.destroyAllWindows() # Close window if you were showing images

if not objpoints or not imgpoints:
    print("\nError: No valid chessboard corners were found in any image. Cannot calibrate.")
    sys.exit(1)

if image_size is None:
    print("\nError: Could not determine image size.") # Should not happen if images were processed
    sys.exit(1)

print(f"\nCalibrating camera using {len(objpoints)} valid images...")

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

if not ret:
    print("Error: Camera calibration failed.")
    sys.exit(1)

print("Calibration successful.")

# --- Format and Print Output ---

print("\n--- Calibration Results ---")

# Camera Matrix Formatting
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

print("camera_matrix:")
print(f"  data: [{fx}, 0.0, {cx}, 0.0, {fy}, {cy}, 0.0, 0.0, 1.0]") # Explicitly use floats for zeros

# Distortion Coefficients Formatting
# Ensure dist_coeffs is treated as a flat list/array for easier indexing
dist_coeffs_flat = dist_coeffs.flatten()

# Assign default values in case calibration returns fewer coefficients than expected
k1 = dist_coeffs_flat[0] if len(dist_coeffs_flat) > 0 else 0.0
k2 = dist_coeffs_flat[1] if len(dist_coeffs_flat) > 1 else 0.0
p1 = dist_coeffs_flat[2] if len(dist_coeffs_flat) > 2 else 0.0
p2 = dist_coeffs_flat[3] if len(dist_coeffs_flat) > 3 else 0.0
k3 = dist_coeffs_flat[4] if len(dist_coeffs_flat) > 4 else 0.0

print("distortion_coefficients:")
print(f"  k1: {k1}")
print(f"  k2: {k2}")
print(f"  p1: {p1}")
print(f"  p2: {p2}")
print(f"  k3: {k3}")
print("---------------------------\n")

# Optional: Print reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Mean Reprojection Error: {mean_error / len(objpoints)}")