"""
calibrate_charuco.py
====================
Camera calibration using a ChArUco board.

WHAT IS CAMERA CALIBRATION?
----------------------------
Every physical camera lens introduces two types of distortion that must be
corrected before accurate 3D measurements can be made:

  1. Intrinsic parameters (the camera matrix K):
     Describes the camera's internal geometry:
       - fx, fy : focal lengths in pixels (how strongly the lens converges light)
       - cx, cy : optical center (principal point) in pixels
     The camera matrix maps a 3D point in the camera frame to a 2D pixel.

  2. Distortion coefficients:
     Describes how the lens warps the image:
       - k1, k2, k3 : radial distortion (barrel or pincushion effect)
       - p1, p2     : tangential distortion (lens/sensor misalignment)

WHY CALIBRATION IS NECESSARY FOR SOLVEPNP
------------------------------------------
solvePnP() computes the 3D pose of a marker by solving the equation:
    pixel = K * (R * world_point + t)

If K (the camera matrix) is wrong, or if distortion is not corrected, the
pixel coordinates fed into solvePnP will be inaccurate. This directly causes
errors in the estimated position and orientation of the marker.

Calibration is done once per camera. The resulting camera_matrix.npy and
dist_coeffs.npy files are then loaded by detect_pose.py at runtime.

WHY CHARUCO IS USED FOR CALIBRATION ONLY
-----------------------------------------
ChArUco is excellent for calibration because:
  - Partial board views are still usable (ArUco IDs identify each corner)
  - Sub-pixel accurate corner positions (chessboard corners)
  - Robust to varying lighting and partial occlusion

ChArUco is NOT used at runtime because:
  - Runtime detection uses small standalone ArUco markers on physical posts
  - Standalone ArUco markers are compact, easy to print, and easy to attach
  - solvePnP only needs the four corners of a square marker
  - There is no benefit to using a full calibration board at runtime

PIPELINE
---------
1. Load calibration images from calibration_images/
2. Detect ArUco markers in each image
3. Interpolate ChArUco corners (chessboard corners adjacent to detected markers)
4. Apply sub-pixel corner refinement using cornerSubPix
5. Accumulate valid corners and IDs across all images
6. Run cv2.aruco.calibrateCameraCharuco() to solve for K and distortion
7. Save camera_matrix.npy and dist_coeffs.npy

Usage:
    python calibrate_charuco.py
    python calibrate_charuco.py --images calibration_images/ --show
    python calibrate_charuco.py --images calibration_images/ --out calibration/
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

from generate_charuco import create_charuco_board


# Sub-pixel corner refinement parameters.
# cornerSubPix refines each detected corner to sub-pixel accuracy by searching
# for the point where the gradient of image intensity changes sign.
# HALF_WIN defines the search window size around each coarse corner estimate.
# Larger windows handle more blur and lower resolution images.
# Smaller windows are faster and more precise when corners are sharp.

SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,      # Maximum number of refinement iterations
    0.001    # Stop when corner moves less than this many pixels
)
SUBPIX_HALF_WIN = (11, 11)   # Half-window size for the gradient search
SUBPIX_ZERO_ZONE = (-1, -1)  # Disable the dead zone inside the search window


# Step 1: Load calibration images from disk

def load_images(image_dir):
    """
    Load all JPG and PNG images from the given directory.

    Parameters
    ----------
    image_dir : str
        Path to the folder containing calibration images.

    Returns
    -------
    images : list of (str, numpy.ndarray) tuples
        Each tuple contains (file_path, BGR_image).

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    RuntimeError
        If no images are found or all fail to load.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            "Image directory not found: '" + image_dir + "'\n"
            "Create the directory and place calibration images inside it."
        )

    search_patterns = [
        os.path.join(image_dir, "*.jpg"),
        os.path.join(image_dir, "*.jpeg"),
        os.path.join(image_dir, "*.png"),
    ]

    paths = []
    for pattern in search_patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))

    if not paths:
        raise RuntimeError(
            "No JPG or PNG images found in '" + image_dir + "'.\n"
            "Photograph the ChArUco board from 15-30 different angles and distances,\n"
            "then place the images in that directory."
        )

    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print("  [WARNING] Failed to read: " + path + " -- skipping.")
            continue
        images.append((path, img))
        print("  [loaded] " + os.path.basename(path) + "  " +
              str(img.shape[1]) + " x " + str(img.shape[0]))

    if not images:
        raise RuntimeError("All image loads failed. Check file permissions and formats.")

    return images


# Step 2: Detect ArUco markers and interpolate ChArUco corners

def detect_charuco_corners(images, board, dictionary, show=False):
    """
    Detect ArUco markers and interpolate ChArUco corners in every calibration image.

    How this works:
      1. detectMarkers() finds the coarse pixel positions of ArUco markers.
         This gives us a rough location of where the board is in the image.
      2. CharucoDetector.detectBoard() uses the detected markers to find the
         exact pixel positions of the inner chessboard corners that sit at the
         intersections between squares. These corners are more accurate than
         marker corners because they are defined by two perpendicular edges.
      3. cornerSubPix() refines each corner position to sub-pixel accuracy by
         iteratively searching for the local intensity gradient zero-crossing.

    Parameters
    ----------
    images     : list of (str, numpy.ndarray), from load_images()
    board      : cv2.aruco.CharucoBoard
    dictionary : cv2.aruco.Dictionary
    show       : bool, if True display each image with detected corners

    Returns
    -------
    all_charuco_corners : list of numpy.ndarray, shape (N, 1, 2), dtype float32
    all_charuco_ids     : list of numpy.ndarray, shape (N, 1), dtype int32
    image_size          : tuple of (width, height) in pixels
    """
    # Configure detector parameters for robust detection under varied lighting.
    # Adaptive thresholding adjusts the binarization threshold locally per pixel,
    # which handles uneven illumination (shadows, highlights) much better than
    # a single global threshold would.
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.adaptiveThreshWinSizeMin  = 3
    detector_params.adaptiveThreshWinSizeMax  = 23
    detector_params.adaptiveThreshWinSizeStep = 10
    detector_params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_CONTOUR

    aruco_detector   = cv2.aruco.ArucoDetector(dictionary, detector_params)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    all_charuco_corners = []
    all_charuco_ids     = []
    image_size          = None
    total               = len(images)

    for idx, (path, img) in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        label = "[" + str(idx + 1) + "/" + str(total) + "] " + os.path.basename(path)
        print("\n  " + label)

        # Detect the ArUco markers embedded inside the ChArUco board.
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) == 0:
            print("    [SKIP] No ArUco markers detected in this image.")
            continue

        print("    Markers detected: " + str(len(marker_ids)))

        # Interpolate the chessboard-style inner corners using the known marker
        # positions as anchors. The board object provides the physical 3D
        # coordinates of each corner based on SQUARE_LENGTH and MARKER_LENGTH.
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < 4:
            n = len(charuco_ids) if charuco_ids is not None else 0
            print("    [SKIP] Only " + str(n) + " ChArUco corners found (minimum 4 required).")
            continue

        print("    ChArUco corners: " + str(len(charuco_ids)))

        # Refine corner positions to sub-pixel accuracy.
        # This step significantly improves calibration accuracy and is always
        # recommended when pixel-level precision matters.
        charuco_corners_refined = cv2.cornerSubPix(
            gray,
            charuco_corners.astype(np.float32),
            SUBPIX_HALF_WIN,
            SUBPIX_ZERO_ZONE,
            SUBPIX_CRITERIA,
        )

        all_charuco_corners.append(charuco_corners_refined)
        all_charuco_ids.append(charuco_ids)

        # Optionally display the detected corners for visual verification.
        if show:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners_refined, charuco_ids)
            display = _fit_to_screen(vis, max_dim=900)
            cv2.imshow("ChArUco Corners - " + os.path.basename(path) + "  (any key to continue)", display)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:  # ESC aborts display mode for remaining images
                show = False

    return all_charuco_corners, all_charuco_ids, image_size


# Step 3: Solve for camera intrinsics and distortion

def calibrate_camera(all_charuco_corners, all_charuco_ids, board, image_size):
    """
    Run cv2.aruco.calibrateCameraCharuco() and return the calibration results.

    This function minimizes the reprojection error: the average pixel-space
    distance between observed corner positions and where the calibrated camera
    model predicts they should appear.

    A reprojection error below 0.5 pixels is considered acceptable for most
    robotics applications. Below 0.3 pixels is excellent.

    Parameters
    ----------
    all_charuco_corners : list of numpy.ndarray
    all_charuco_ids     : list of numpy.ndarray
    board               : cv2.aruco.CharucoBoard
    image_size          : (width, height) tuple

    Returns
    -------
    rms_error     : float, mean reprojection error in pixels
    camera_matrix : numpy.ndarray, shape (3, 3), dtype float64
    dist_coeffs   : numpy.ndarray, shape (1, 5), dtype float64

    Raises
    ------
    RuntimeError if fewer than 3 valid images are available.
    """
    if len(all_charuco_corners) < 3:
        raise RuntimeError(
            "Need at least 3 valid calibration images, got " + str(len(all_charuco_corners)) + ".\n"
            "Capture more images and make sure the board is clearly visible."
        )

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,   # Pass None to let OpenCV initialize the camera matrix automatically
        None,   # Pass None to let OpenCV initialize distortion coefficients to zero
    )

    return rms_error, camera_matrix, dist_coeffs


# Step 4: Save calibration output

def save_calibration(camera_matrix, dist_coeffs, out_dir="calibration"):
    """
    Save camera_matrix and dist_coeffs as NumPy .npy files.

    Parameters
    ----------
    camera_matrix : numpy.ndarray, shape (3, 3)
    dist_coeffs   : numpy.ndarray
    out_dir       : str, output directory path
    """
    os.makedirs(out_dir, exist_ok=True)
    matrix_path = os.path.join(out_dir, "camera_matrix.npy")
    dist_path   = os.path.join(out_dir, "dist_coeffs.npy")
    np.save(matrix_path, camera_matrix)
    np.save(dist_path,   dist_coeffs)
    print("\n[OK] Camera matrix saved -> " + matrix_path)
    print("[OK] Distortion coefficients saved -> " + dist_path)


def _fit_to_screen(img, max_dim=900):
    """Resize image so its largest dimension is max_dim, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def main():
    parser = argparse.ArgumentParser(description="Calibrate camera using a ChArUco board.")
    parser.add_argument(
        "--images", default=os.path.join("test_images", "calibration_imgs"),
        help="Directory containing calibration images (default: test_images/calibration_imgs/)"
    )
    parser.add_argument(
        "--out", default="calibration",
        help="Output directory for calibration .npy files (default: calibration/)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display detected corners for each image interactively"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  ChArUco Camera Calibration")
    print("=" * 50)

    board, dictionary = create_charuco_board()

    # Step 1: Load images
    print("\n[1/4] Loading calibration images from '" + args.images + "'...")
    try:
        images = load_images(args.images)
    except (FileNotFoundError, RuntimeError) as err:
        print("\n[ERROR] " + str(err))
        sys.exit(1)
    print("       " + str(len(images)) + " image(s) loaded.")

    # Step 2: Detect ChArUco corners
    print("\n[2/4] Detecting ChArUco corners...")
    try:
        all_corners, all_ids, img_size = detect_charuco_corners(
            images, board, dictionary, show=args.show
        )
    except Exception as err:
        print("\n[ERROR] Detection failed: " + str(err))
        sys.exit(1)
    print("\n       Valid images for calibration: " + str(len(all_corners)) + " / " + str(len(images)))

    # Step 3: Run calibration
    print("\n[3/4] Running calibration...")
    try:
        rms, camera_matrix, dist_coeffs = calibrate_camera(
            all_corners, all_ids, board, img_size
        )
    except RuntimeError as err:
        print("\n[ERROR] " + str(err))
        sys.exit(1)

    # Print results
    print("\n" + "=" * 50)
    print("  Calibration Results")
    print("=" * 50)
    print("  RMS Reprojection Error: " + str(round(rms, 4)) + " pixels")
    print("")
    print("  Camera Matrix (K):")
    for row in camera_matrix:
        print("    " + str(row))
    print("")
    print("  Distortion Coefficients:")
    print("    " + str(dist_coeffs.flatten()))
    print("=" * 50)

    if rms > 1.0:
        print("\n[WARNING] Reprojection error > 1 px. Consider:")
        print("  - Capturing more calibration images (15-30 is recommended)")
        print("  - Ensuring the board is flat, rigid, and fully visible in each image")
        print("  - Shooting from a wider variety of angles and distances")
    elif rms < 0.3:
        print("\n[OK] Excellent reprojection error (< 0.3 px).")
    else:
        print("\n[OK] Good reprojection error.")

    # Step 4: Save calibration
    print("\n[4/4] Saving calibration files...")
    save_calibration(camera_matrix, dist_coeffs, args.out)

    print("\n[Done] Calibration complete. Run detect_pose.py next.")


if __name__ == "__main__":
    main()
