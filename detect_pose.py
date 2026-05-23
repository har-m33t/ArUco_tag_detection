"""
detect_pose.py
==============
Standalone ArUco marker detection and 6-DOF pose estimation.

This script is the runtime perception pipeline for the rover.
It does NOT use ChArUco boards. It only detects standalone ArUco markers.

WHAT IS 6-DOF POSE ESTIMATION?
--------------------------------
6 degrees of freedom means we estimate both:
  - 3 translational DOF: X, Y, Z position of the marker relative to the camera
  - 3 rotational DOF: roll, pitch, yaw of the marker relative to the camera

Together, these six numbers fully describe where the marker is in 3D space
and how it is oriented, from the camera's perspective.

HOW solvePnP WORKS
--------------------
solvePnP() solves the Perspective-n-Point problem.

Given:
  - N 3D world-space points   (the known positions of the marker corners in meters)
  - N 2D image-space points   (the detected pixel positions of those same corners)
  - The camera matrix K       (intrinsic parameters: focal length, optical center)
  - The distortion coefficients

Find:
  - rvec : rotation vector (Rodrigues notation, 3 elements encoding axis and angle)
  - tvec : translation vector (3 elements: X, Y, Z in meters)

Such that the projection equation is satisfied:
    s * [u, v, 1]^T = K * [R | t] * [X_world, Y_world, Z_world, 1]^T

where s is a scaling factor and R is the rotation matrix from rvec.

The solution tells us exactly where the marker is relative to the camera.

WHAT RVEC AND TVEC REPRESENT
------------------------------
  tvec = [tx, ty, tz]
    tx > 0 : marker is to the right of the camera center
    ty > 0 : marker is below the camera center (Y axis points down)
    tz > 0 : marker is in front of the camera (positive Z = forward)

  rvec = [rx, ry, rz]
    Direction of the vector = axis of rotation
    Magnitude of the vector = angle of rotation in radians

  distance = sqrt(tx^2 + ty^2 + tz^2)

COORDINATE CONVENTION (OpenCV camera frame)
--------------------------------------------
  +X : right
  +Y : down
  +Z : forward (into the scene, away from camera)

This is the standard OpenCV convention. Note that it differs from some other
robotics conventions where +Z points up.

WHY WE USE SOLVEPNP_IPPE_SQUARE
---------------------------------
IPPE_SQUARE (Infinitesimal Plane-based Pose Estimation for squares) is a
closed-form analytical solver specifically designed for planar square targets.
It is:
  - Faster than the iterative SOLVEPNP_ITERATIVE method
  - More accurate for square markers because it exploits the known square geometry
  - Returns two candidate solutions (the ambiguous cases) but we take the best one

WHY UNDISTORTION IS APPLIED BEFORE DETECTION
---------------------------------------------
ArUco corner detection operates on pixel coordinates. If the image is distorted,
the corner positions will be shifted from their true positions, which degrades
both detection reliability and solvePnP accuracy.

By undistorting the image first, we ensure:
  - The detector sees geometrically correct straight edges
  - The corners passed to solvePnP are at their true undistorted pixel locations
  - We then pass zero distortion coefficients to solvePnP and projectPoints

Usage:
    # Static image
    python detect_pose.py --image test_images/sample.jpg

    # Live webcam
    python detect_pose.py --webcam

    # Custom marker size, different calibration directory
    python detect_pose.py --image test.jpg --marker-size 0.15 --calib calibration/

    # Webcam without undistortion (faster, slightly less accurate)
    python detect_pose.py --webcam --no-undistort

    # Webcam with second camera
    python detect_pose.py --webcam --cam-index 1
"""

import argparse
import glob
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

from utils import (
    load_calibration,
    compute_distance,
    draw_text_overlay,
    draw_projected_arrow,
    format_pose_text,
    undistort_image,
)


# Configuration constants.
# MARKER_SIZE must match the actual printed marker size in meters.
# The competition posts have 20 cm x 20 cm faces, so the marker size is 0.20 m.
ARUCO_DICT_ID         = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_SIZE   = 0.20    # 20 cm competition face
AXIS_LENGTH           = 0.06    # Length of the drawn coordinate axes in meters
ARROW_LENGTH          = 0.12    # Length of the 3D pose arrow in meters
POSE_SMOOTHING_WINDOW = 5       # Number of frames to average for pose smoothing


# Step 1: Build the ArUco marker detector

def build_detector():
    """
    Create and configure an ArucoDetector tuned for outdoor robotics use.

    Parameter choices:
      - adaptiveThreshWinSize range: small min catches high-res fine detail;
        large max handles blur and low-contrast markers at distance
      - CORNER_REFINE_SUBPIX: after coarse detection, refines each corner to
        sub-pixel accuracy using the image gradient. This directly improves
        the accuracy of solvePnP because the input 2D points are more precise.
      - minMarkerPerimeterRate: allows detecting small/distant markers

    Returns
    -------
    detector : cv2.aruco.ArucoDetector
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    params = cv2.aruco.DetectorParameters()

    # Adaptive thresholding handles uneven lighting (bright sun, shadow patches).
    # The algorithm computes a local threshold for each pixel based on the mean
    # intensity in a surrounding window.
    params.adaptiveThreshWinSizeMin     = 3
    params.adaptiveThreshWinSizeMax     = 53
    params.adaptiveThreshWinSizeStep    = 4
    params.adaptiveThreshConstant       = 7

    # Allow detection of smaller (more distant) markers.
    params.minMarkerPerimeterRate       = 0.01
    params.maxMarkerPerimeterRate       = 4.0

    # Sub-pixel corner refinement: after the marker is found, search within a
    # small window around each corner for the exact sub-pixel position.
    params.cornerRefinementMethod       = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize      = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy  = 0.01

    return cv2.aruco.ArucoDetector(dictionary, params)


# Step 2: Detect standalone ArUco markers

def detect_markers(image, detector):
    """
    Run ArUco marker detection on a BGR image.

    The detector converts the image to grayscale internally, then applies
    adaptive thresholding to create a binary image. Contours are extracted
    and filtered by shape. Candidate quadrilaterals are decoded against the
    ArUco dictionary.

    Parameters
    ----------
    image    : numpy.ndarray, BGR input image
    detector : cv2.aruco.ArucoDetector

    Returns
    -------
    corners  : tuple of numpy.ndarray, one (1, 4, 2) float32 array per marker
               Corners are ordered: top-left, top-right, bottom-right, bottom-left
    ids      : numpy.ndarray, shape (N, 1), int32 marker IDs, or None
    rejected : tuple of candidate quadrilaterals that failed decoding
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids, rejected


# Step 3: Build the 3D object point model for a square ArUco marker

def make_marker_object_points(marker_size):
    """
    Define the known 3D positions of the four corners of a square ArUco marker.

    The marker is modeled as a flat square centered at the world origin,
    lying in the Z=0 plane. The physical size of the marker (in meters)
    determines the actual 3D coordinates.

    Corner ordering matches what OpenCV's ArUco detector returns:
      Index 0 : top-left     (-half, +half, 0)
      Index 1 : top-right    (+half, +half, 0)
      Index 2 : bottom-right (+half, -half, 0)
      Index 3 : bottom-left  (-half, -half, 0)

    IMPORTANT: If marker_size is wrong, all distance estimates will be wrong.
    Measure the actual printed marker and set this value accordingly.

    Parameters
    ----------
    marker_size : float, physical side length of the marker in meters

    Returns
    -------
    obj_pts : numpy.ndarray, shape (4, 1, 3), dtype float32
    """
    half = marker_size / 2.0
    obj_pts = np.array([
        [-half,  half, 0.0],   # Top-left
        [ half,  half, 0.0],   # Top-right
        [ half, -half, 0.0],   # Bottom-right
        [-half, -half, 0.0],   # Bottom-left
    ], dtype=np.float32).reshape(4, 1, 3)
    return obj_pts


# Step 4: Estimate the 6-DOF pose of a single detected marker

def estimate_pose(corners, obj_pts, camera_matrix, dist_coeffs):
    """
    Estimate the 6-DOF pose of one ArUco marker using solvePnP.

    This function solves the Perspective-n-Point problem:
    given N 3D world points and their corresponding 2D image projections,
    find the rotation (rvec) and translation (tvec) that relate the marker
    coordinate frame to the camera coordinate frame.

    We use SOLVEPNP_IPPE_SQUARE, which is a closed-form solver designed for
    planar square objects. It is faster and more accurate than the generic
    iterative solver for this specific case.

    Parameters
    ----------
    corners       : numpy.ndarray, shape (1, 4, 2), float32
                    Detected 2D corner positions in pixel coordinates.
    obj_pts       : numpy.ndarray, shape (4, 1, 3), float32
                    Known 3D corner positions in the marker's local frame.
    camera_matrix : numpy.ndarray, shape (3, 3), float64
    dist_coeffs   : numpy.ndarray, distortion coefficients

    Returns
    -------
    success : bool, True if solvePnP found a valid solution
    rvec    : numpy.ndarray, shape (3, 1), float64, Rodrigues rotation vector
    tvec    : numpy.ndarray, shape (3, 1), float64, translation vector in meters
    """
    img_pts = corners.reshape(4, 1, 2).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    return success, rvec, tvec


# Step 5: Draw all visualization elements on the image

def draw_pose(image, corners, rvec, tvec, marker_id,
              camera_matrix, dist_coeffs,
              axis_length=AXIS_LENGTH, arrow_length=ARROW_LENGTH,
              hud_origin=(15, 35)):
    """
    Render all pose visualization elements for one detected marker.

    Draws in-place on the image:
      1. Marker bounding box: cyan quadrilateral showing the detected corners
      2. Corner dots: color-coded circles at each corner position
      3. Coordinate axes: drawn by cv2.drawFrameAxes()
           Red   = marker local +X (right)
           Green = marker local +Y (down)
           Blue  = marker local +Z (forward, out of the marker surface)
      4. 3D arrow: projects arrow_length meters along local +Z
      5. Text HUD: marker ID, X/Y/Z position, distance, rotation angles

    Parameters
    ----------
    image         : numpy.ndarray, BGR image (modified in-place)
    corners       : numpy.ndarray, shape (1, 4, 2), float32
    rvec          : numpy.ndarray, shape (3, 1), Rodrigues rotation vector
    tvec          : numpy.ndarray, shape (3, 1), translation vector in meters
    marker_id     : int, ArUco marker ID
    camera_matrix : numpy.ndarray, shape (3, 3)
    dist_coeffs   : numpy.ndarray
    axis_length   : float, coordinate axis length in meters
    arrow_length  : float, 3D arrow length in meters
    hud_origin    : (int, int), pixel position of the top of the text HUD
    """
    pts = corners.reshape(4, 2).astype(int)

    # Draw the marker bounding box as a closed quadrilateral.
    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # Color-coded corner dots help identify marker orientation at a glance.
    corner_colors = [
        (0,   0, 255),   # Top-left     : red
        (0, 255,   0),   # Top-right    : green
        (255, 0,   0),   # Bottom-right : blue
        (255, 255, 0),   # Bottom-left  : yellow
    ]
    for i, (px, py) in enumerate(pts):
        cv2.circle(image, (int(px), int(py)), 5, corner_colors[i], -1)

    # Draw the coordinate axes.
    # cv2.drawFrameAxes() projects three unit vectors along X, Y, Z from the
    # marker origin into 2D pixel space using the camera model. It draws:
    #   Red line   = +X direction (right in marker frame)
    #   Green line = +Y direction (down in marker frame)
    #   Blue line  = +Z direction (forward, toward camera)
    cv2.drawFrameAxes(
        image,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        axis_length,
        thickness=3,
    )

    # Draw the 3D arrow projecting forward from the marker.
    draw_projected_arrow(
        image, tvec, rvec,
        camera_matrix, dist_coeffs,
        arrow_length=arrow_length,
    )

    # Render the text HUD with pose information.
    lines = format_pose_text(tvec, rvec, marker_id=marker_id)
    draw_text_overlay(image, lines, origin=hud_origin)


# Pose smoother for webcam mode

class PoseSmoother:
    """
    Exponential moving average smoother for rvec and tvec over time.

    Without smoothing, solvePnP can produce slightly different results each
    frame due to sub-pixel detection noise. This class maintains a rolling
    buffer of the last N pose estimates and returns their element-wise mean.

    The effect is a stable, jitter-free AR overlay that is much more readable
    in live video than the raw per-frame estimates.

    Usage:
        smoother = PoseSmoother(window=5)
        smooth_rvec, smooth_tvec = smoother.update(rvec, tvec)
    """

    def __init__(self, window=POSE_SMOOTHING_WINDOW):
        self._rvecs = deque(maxlen=window)
        self._tvecs = deque(maxlen=window)

    def update(self, rvec, tvec):
        """Add a new pose estimate and return the smoothed pose."""
        self._rvecs.append(np.array(rvec, dtype=np.float64).flatten())
        self._tvecs.append(np.array(tvec, dtype=np.float64).flatten())
        smooth_r = np.mean(self._rvecs, axis=0).reshape(3, 1)
        smooth_t = np.mean(self._tvecs, axis=0).reshape(3, 1)
        return smooth_r, smooth_t

    def reset(self):
        """Clear the buffer when a marker disappears from view."""
        self._rvecs.clear()
        self._tvecs.clear()


# High-level pipeline: static image

def run_on_image(image_path, camera_matrix, dist_coeffs, marker_size,
                 undistort=True, save_output=False):
    """
    Run the full detection and pose estimation pipeline on a static image.

    Parameters
    ----------
    image_path    : str, path to the input image
    camera_matrix : numpy.ndarray, shape (3, 3)
    dist_coeffs   : numpy.ndarray
    marker_size   : float, physical marker side length in meters
    undistort     : bool, whether to undistort the image before detection
    save_output   : bool, whether to save the annotated result image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found: '" + image_path + "'")

    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError("cv2.imread() could not open: '" + image_path + "'")

    print("[OK] Image loaded: " + image_path +
          "  (" + str(frame.shape[1]) + " x " + str(frame.shape[0]) + ")")

    # Undistort the frame and get the refined camera matrix for the corrected image.
    # After undistortion, we pass zero distortion to solvePnP and projectPoints.
    if undistort:
        frame, camera_matrix = undistort_image(frame, camera_matrix, dist_coeffs)
        dist_coeffs = np.zeros(5, dtype=np.float64)
        print("[OK] Undistortion applied.")

    detector = build_detector()
    obj_pts  = make_marker_object_points(marker_size)

    corners, ids, _ = detect_markers(frame, detector)
    output = frame.copy()

    if ids is None or len(ids) == 0:
        print("[INFO] No ArUco markers detected in the image.")
        draw_text_overlay(output, ["No markers detected"], color=(0, 0, 255))
    else:
        print("[OK] Detected " + str(len(ids)) + " marker(s).")

        for i, (corner, marker_id) in enumerate(zip(corners, ids.flatten())):
            success, rvec, tvec = estimate_pose(
                corner, obj_pts, camera_matrix, dist_coeffs
            )
            if not success:
                print("  [WARNING] solvePnP failed for marker ID " + str(marker_id))
                continue

            dist = compute_distance(tvec)
            print("  ID " + str(marker_id) + ":  "
                  "X=" + str(round(float(tvec[0]), 4)) + "  "
                  "Y=" + str(round(float(tvec[1]), 4)) + "  "
                  "Z=" + str(round(float(tvec[2]), 4)) + "  "
                  "dist=" + str(round(dist, 4)) + " m")

            # Offset each marker's HUD downward so they do not overlap.
            hud_y = 35 + i * 210
            draw_pose(
                output, corner, rvec, tvec, marker_id,
                camera_matrix, dist_coeffs,
                hud_origin=(15, hud_y),
            )

    if save_output:
        base, ext = os.path.splitext(image_path)
        out_path = base + "_pose" + ext
        cv2.imwrite(out_path, output)
        print("[OK] Annotated image saved -> " + out_path)

    display = _fit_to_screen(output, max_dim=1000)
    cv2.imshow("ArUco Pose Estimation  (any key to close)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# High-level pipeline: live webcam

def run_webcam(camera_matrix, dist_coeffs, marker_size,
               cam_index=0, smooth=True, undistort=True):
    """
    Run real-time ArUco detection and pose estimation from a webcam feed.

    Keyboard controls:
      q or ESC  : quit
      s         : save a snapshot to snapshots/
      u         : toggle undistortion on/off
      p         : toggle pose smoothing on/off

    Parameters
    ----------
    camera_matrix : numpy.ndarray, shape (3, 3)
    dist_coeffs   : numpy.ndarray
    marker_size   : float, physical marker side length in meters
    cam_index     : int, OpenCV camera device index
    smooth        : bool, enable pose smoothing
    undistort     : bool, apply undistortion each frame
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera index " + str(cam_index))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Pre-compute the undistortion remapping for efficiency.
    # initUndistortRectifyMap computes a pixel-to-pixel lookup table so each
    # frame can be corrected with a fast remap() call instead of recomputing
    # the correction from scratch.
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_matrix, (w, h), cv2.CV_16SC2
    )

    detector   = build_detector()
    obj_pts    = make_marker_object_points(marker_size)
    smoothers  = {}   # Maps marker_id -> PoseSmoother

    fps_buffer = deque(maxlen=30)
    prev_time  = time.time()
    do_smooth  = smooth
    do_undist  = undistort

    os.makedirs("snapshots", exist_ok=True)
    print("[OK] Webcam " + str(cam_index) + " opened. Press 'q' to quit, 's' to save.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from webcam.")
            break

        # Update FPS counter.
        now = time.time()
        fps_buffer.append(1.0 / max(now - prev_time, 1e-9))
        prev_time = now
        fps = float(np.mean(fps_buffer))

        # Undistort the frame if enabled.
        if do_undist:
            proc_frame  = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            active_K    = new_matrix
            active_dist = np.zeros(5, dtype=np.float64)
        else:
            proc_frame  = frame
            active_K    = camera_matrix
            active_dist = dist_coeffs

        corners, ids, _ = detect_markers(proc_frame, detector)
        output = proc_frame.copy()

        detected_ids = set()
        if ids is not None and len(ids) > 0:
            for corner, marker_id in zip(corners, ids.flatten()):
                detected_ids.add(int(marker_id))
                success, rvec, tvec = estimate_pose(
                    corner, obj_pts, active_K, active_dist
                )
                if not success:
                    continue

                if do_smooth:
                    if marker_id not in smoothers:
                        smoothers[marker_id] = PoseSmoother(POSE_SMOOTHING_WINDOW)
                    rvec, tvec = smoothers[marker_id].update(rvec, tvec)

                draw_pose(output, corner, rvec, tvec, int(marker_id), active_K, active_dist)

        # Clear smoothers for markers that are no longer visible.
        for mid in list(smoothers.keys()):
            if mid not in detected_ids:
                smoothers[mid].reset()

        # Status HUD in the top-right corner.
        n_detected = len(ids) if ids is not None else 0
        status_lines = [
            "FPS: " + str(round(fps, 1)),
            "Markers: " + str(n_detected),
            "Undistort: " + ("ON" if do_undist else "OFF") + "  [u]",
            "Smooth: " + ("ON" if do_smooth else "OFF") + "  [p]",
            "Quit [q]  Save [s]",
        ]
        hud_x = output.shape[1] - 270
        draw_text_overlay(output, status_lines, origin=(hud_x, 35),
                          font_scale=0.55, color=(200, 200, 200))

        cv2.imshow("ArUco Pose Estimation - Live  (q=quit)", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join("snapshots", "snapshot_" + ts + ".jpg")
            cv2.imwrite(path, output)
            print("[OK] Snapshot saved -> " + path)
        elif key == ord('u'):
            do_undist = not do_undist
            print("[INFO] Undistortion: " + ("enabled" if do_undist else "disabled"))
        elif key == ord('p'):
            do_smooth = not do_smooth
            for s in smoothers.values():
                s.reset()
            print("[INFO] Pose smoothing: " + ("enabled" if do_smooth else "disabled"))

    cap.release()
    cv2.destroyAllWindows()


def _fit_to_screen(img, max_dim=1000):
    """Resize image so its largest dimension is at most max_dim pixels."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def main():
    parser = argparse.ArgumentParser(
        description="Detect standalone ArUco markers and estimate 6-DOF pose."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--image", "-i", default=None,
        help="Path to a static test image"
    )
    source.add_argument(
        "--webcam", "-w", action="store_true",
        help="Use live webcam feed"
    )
    parser.add_argument(
        "--cam-index", type=int, default=0,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--marker-size", type=float, default=DEFAULT_MARKER_SIZE,
        help="Physical marker side length in meters (default: " + str(DEFAULT_MARKER_SIZE) + ")"
    )
    parser.add_argument(
        "--calib", default="calibration",
        help="Directory containing camera_matrix.npy and dist_coeffs.npy"
    )
    parser.add_argument(
        "--no-undistort", action="store_true",
        help="Skip image undistortion (faster but less accurate)"
    )
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="Disable pose smoothing in webcam mode"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the annotated result image (static mode only)"
    )
    args = parser.parse_args()

    # Load calibration
    matrix_path = os.path.join(args.calib, "camera_matrix.npy")
    dist_path   = os.path.join(args.calib, "dist_coeffs.npy")
    try:
        camera_matrix, dist_coeffs = load_calibration(matrix_path, dist_path)
    except FileNotFoundError as err:
        print("\n[ERROR] " + str(err))
        sys.exit(1)

    print("=" * 50)
    print("  ArUco Pose Estimation")
    print("=" * 50)
    print("  Dictionary   : DICT_4X4_50")
    print("  Marker size  : " + str(args.marker_size) + " m  (" +
          str(int(args.marker_size * 100)) + " cm)")
    print("  Calibration  : " + args.calib + "/")
    print("=" * 50)

    undistort = not args.no_undistort

    if args.webcam:
        try:
            run_webcam(
                camera_matrix, dist_coeffs,
                args.marker_size,
                cam_index=args.cam_index,
                smooth=not args.no_smooth,
                undistort=undistort,
            )
        except RuntimeError as err:
            print("\n[ERROR] " + str(err))
            sys.exit(1)
    else:
        # Static image mode.
        # If no path is given, automatically pick the first image found in
        # test_images/aruco_tags/. Drop phone photos of the ArUco tags into
        # that folder and run without any arguments.
        image_path = args.image
        if image_path is None:
            search_dir = os.path.join("test_images", "aruco_tags")
            candidates = sorted(
                glob.glob(os.path.join(search_dir, "*.jpg"))  +
                glob.glob(os.path.join(search_dir, "*.jpeg")) +
                glob.glob(os.path.join(search_dir, "*.png"))
            )
            if candidates:
                image_path = candidates[0]
                print("[INFO] No --image given. Using: " + image_path)
            else:
                print("[ERROR] No --image path given and no images found in '" + search_dir + "'.")
                print("  Drop ArUco tag photos from your phone into that folder, then re-run.")
                print("  Usage: python detect_pose.py --image path/to/image.jpg")
                print("         python detect_pose.py --webcam")
                sys.exit(1)

        try:
            run_on_image(
                image_path,
                camera_matrix, dist_coeffs,
                args.marker_size,
                undistort=undistort,
                save_output=args.save,
            )
        except (FileNotFoundError, RuntimeError) as err:
            print("\n[ERROR] " + str(err))
            sys.exit(1)


if __name__ == "__main__":
    main()
