"""
utils.py
========
Shared helper functions for the ArUco pose estimation pipeline.

All functions in this module are pure utilities with no global state.
They are imported by detect_pose.py and calibrate_charuco.py.

Functions
---------
load_calibration        : Load camera_matrix.npy and dist_coeffs.npy
rvec_to_rotation_matrix : Convert Rodrigues rotation vector to 3x3 matrix
compute_distance        : Euclidean norm of a translation vector
draw_text_overlay       : Render multi-line text HUD on an image
draw_projected_arrow    : Project a 3D arrow along the marker Z axis and draw it
format_pose_text        : Format rvec and tvec into a list of display strings
undistort_image         : Apply camera undistortion to an image
"""

import os

import cv2
import numpy as np


# Load calibration files

def load_calibration(matrix_path="calibration/camera_matrix.npy",
                     dist_path="calibration/dist_coeffs.npy"):
    """
    Load the camera intrinsic matrix and distortion coefficients from disk.

    The camera matrix K has the form:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    where:
      fx, fy : focal lengths in pixels
      cx, cy : optical center (principal point) in pixels

    These values are specific to each physical camera and lens combination.
    They are determined during calibration and must not be changed manually.

    Parameters
    ----------
    matrix_path : str, path to camera_matrix.npy
    dist_path   : str, path to dist_coeffs.npy

    Returns
    -------
    camera_matrix : numpy.ndarray, shape (3, 3), dtype float64
    dist_coeffs   : numpy.ndarray, shape (1, 5) or (5,), dtype float64

    Raises
    ------
    FileNotFoundError if either file is missing.
    ValueError if loaded arrays have unexpected shapes.
    """
    for path in (matrix_path, dist_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Calibration file not found: '" + path + "'\n"
                "Run calibrate_charuco.py first to generate calibration data."
            )

    camera_matrix = np.load(matrix_path).astype(np.float64)
    dist_coeffs   = np.load(dist_path).astype(np.float64)

    if camera_matrix.shape != (3, 3):
        raise ValueError(
            "Expected camera_matrix shape (3, 3), got " + str(camera_matrix.shape)
        )

    return camera_matrix, dist_coeffs


# Geometry helpers

def rvec_to_rotation_matrix(rvec):
    """
    Convert a Rodrigues rotation vector to a 3x3 rotation matrix.

    OpenCV represents orientation as a compact 3-element Rodrigues vector.
    Its direction is the axis of rotation.
    Its magnitude is the angle of rotation in radians.

    cv2.Rodrigues() converts it to a full 3x3 rotation matrix R such that:
        p_camera = R @ p_marker + t

    Parameters
    ----------
    rvec : array-like, shape (3,) or (3, 1)

    Returns
    -------
    R : numpy.ndarray, shape (3, 3), dtype float64
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64).flatten())
    return R


def compute_distance(tvec):
    """
    Compute the straight-line distance from the camera to the marker.

    The translation vector tvec = [tx, ty, tz] gives the marker's origin
    position in the camera coordinate frame (in meters).
    The Euclidean distance is the L2 norm: sqrt(tx^2 + ty^2 + tz^2).

    Parameters
    ----------
    tvec : array-like, shape (3,) or (3, 1), in meters

    Returns
    -------
    distance : float, in meters
    """
    return float(np.linalg.norm(np.array(tvec, dtype=np.float64).flatten()))


# Visualization helpers

def draw_text_overlay(image, lines, origin=(15, 35), font_scale=0.65,
                      thickness=2, color=(0, 255, 80), shadow=True):
    """
    Render a multi-line text HUD on an image in-place.

    A dark drop-shadow is drawn one pixel offset behind the foreground text
    so that the text remains legible regardless of background color.

    Parameters
    ----------
    image      : numpy.ndarray, BGR image to annotate (modified in-place)
    lines      : list of str, one string per line
    origin     : (int, int), pixel (x, y) of the first line's baseline
    font_scale : float, OpenCV font scale factor
    thickness  : int, text stroke width in pixels
    color      : (int, int, int), BGR foreground color
    shadow     : bool, whether to draw a black shadow behind the text
    """
    font     = cv2.FONT_HERSHEY_SIMPLEX
    line_gap = int(font_scale * 38)  # Vertical spacing between lines
    x0, y0  = origin

    for i, line in enumerate(lines):
        y = y0 + i * line_gap
        if shadow:
            cv2.putText(
                image, line, (x0 + 1, y + 1),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA
            )
        cv2.putText(
            image, line, (x0, y),
            font, font_scale, color, thickness, cv2.LINE_AA
        )


def draw_projected_arrow(image, tvec, rvec, camera_matrix, dist_coeffs,
                         arrow_length=0.12, color=(0, 140, 255),
                         thickness=3, tip_fraction=0.25):
    """
    Project a 3D arrow from the marker origin along its local Z axis and draw it.

    This visualizes the direction the marker is facing in 3D space.
    Two 3D points are defined: the marker origin and a point arrow_length
    meters in front of it along the marker's local +Z axis.
    cv2.projectPoints() transforms both points to 2D pixel coordinates.
    An arrowedLine is then drawn between them.

    OpenCV coordinate convention for the marker frame:
      +X : right
      +Y : down
      +Z : forward (out of the marker surface, toward the camera)

    Parameters
    ----------
    image         : numpy.ndarray, BGR image (modified in-place)
    tvec          : array-like, translation vector (3,) or (3, 1), in meters
    rvec          : array-like, rotation vector (3,) or (3, 1), in radians
    camera_matrix : numpy.ndarray, shape (3, 3)
    dist_coeffs   : numpy.ndarray, distortion coefficients
    arrow_length  : float, length of the 3D arrow in meters
    color         : (int, int, int), BGR arrow color
    thickness     : int, line thickness in pixels
    tip_fraction  : float, arrowhead size as fraction of total arrow length
    """
    # Define the origin and tip in the marker's local 3D coordinate frame.
    # Z = 0 is the marker plane. Positive Z points out toward the camera.
    pts_3d = np.array([
        [0.0, 0.0, 0.0],           # Marker origin
        [0.0, 0.0, arrow_length],   # Tip arrow_length meters along local +Z
    ], dtype=np.float32)

    # Project the 3D points into 2D pixel coordinates using the camera model.
    pts_2d, _ = cv2.projectPoints(
        pts_3d,
        np.array(rvec, dtype=np.float64),
        np.array(tvec, dtype=np.float64),
        camera_matrix,
        dist_coeffs,
    )

    p_start = tuple(pts_2d[0].ravel().astype(int))
    p_end   = tuple(pts_2d[1].ravel().astype(int))

    cv2.arrowedLine(
        image, p_start, p_end,
        color, thickness,
        tipLength=tip_fraction,
        line_type=cv2.LINE_AA,
    )


def format_pose_text(tvec, rvec, marker_id=None):
    """
    Format pose information as a list of strings for HUD display.

    Parameters
    ----------
    tvec      : array-like, translation vector in meters
    rvec      : array-like, rotation vector in radians
    marker_id : int or None, optional ArUco marker ID to include

    Returns
    -------
    lines : list of str
        Ready to pass to draw_text_overlay().
    """
    t    = np.array(tvec, dtype=np.float64).flatten()
    r    = np.array(rvec, dtype=np.float64).flatten()
    dist = compute_distance(t)

    lines = []
    if marker_id is not None:
        lines.append("Marker ID : " + str(marker_id))
    lines += [
        "X  :  " + ("{:+.4f}".format(t[0])) + " m",
        "Y  :  " + ("{:+.4f}".format(t[1])) + " m",
        "Z  :  " + ("{:+.4f}".format(t[2])) + " m",
        "Dist: " + ("{:.4f}".format(dist)) + " m",
        "Rx : " + ("{:+.1f}".format(float(np.degrees(r[0])))) + " deg",
        "Ry : " + ("{:+.1f}".format(float(np.degrees(r[1])))) + " deg",
        "Rz : " + ("{:+.1f}".format(float(np.degrees(r[2])))) + " deg",
    ]
    return lines


# Image preprocessing

def undistort_image(image, camera_matrix, dist_coeffs, crop=True):
    """
    Remove lens distortion from an image using the camera calibration.

    Lens distortion bends straight lines in the real world into curves in the
    image. This must be corrected before making accurate geometric measurements.

    There are two ways to handle distortion in the pipeline:
      (a) Undistort the image first (this function), then pass zero distortion
          coefficients to solvePnP.
      (b) Pass the original distortion coefficients directly to solvePnP and
          skip undistortion.

    Option (a) is used here because it also improves ArUco corner detection
    accuracy, since the detector operates on the corrected pixel positions.

    Parameters
    ----------
    image         : numpy.ndarray, BGR input image
    camera_matrix : numpy.ndarray, shape (3, 3)
    dist_coeffs   : numpy.ndarray, distortion coefficients
    crop          : bool, if True crop away the black border introduced by undistortion

    Returns
    -------
    undistorted  : numpy.ndarray, corrected BGR image
    new_matrix   : numpy.ndarray, shape (3, 3), refined camera matrix for the
                   undistorted image (use this instead of the original K after undistortion)
    """
    h, w = image.shape[:2]

    # getOptimalNewCameraMatrix returns a refined K and a crop region of interest.
    # alpha=0 means no black border pixels appear in the output.
    # alpha=1 keeps all original pixels but shows black borders at the edges.
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0
    )

    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)

    if crop and roi != (0, 0, 0, 0):
        x, y, rw, rh = roi
        undistorted = undistorted[y:y + rh, x:x + rw]

    return undistorted, new_matrix
