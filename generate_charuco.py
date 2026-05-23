"""
generate_charuco.py
===================
Generates a ChArUco calibration board and saves it as a printable PNG image.

PURPOSE OF A CHARUCO BOARD
---------------------------
A ChArUco board is used exclusively for camera calibration.
It is never used during live ArUco marker detection or pose estimation.

Why ChArUco is better than a plain chessboard for calibration:
  - Each square in the board contains a unique ArUco marker ID.
    This means that even a partially visible board can still be used,
    because each visible corner can be matched to a specific known 3D location
    via its adjacent marker ID.
  - Plain chessboards require all inner corners to be fully visible.
    If even one corner is occluded, the entire image is unusable.
  - Chessboard corners provide sub-pixel accurate 2D corner positions.
  - The combination of ArUco ID matching + chessboard corner accuracy
    makes ChArUco the most robust calibration target available in OpenCV.

Why ChArUco is NOT used during runtime pose estimation:
  - Pose estimation uses small standalone ArUco markers attached to physical targets.
  - Those markers are simple, compact, and have well-defined 3D corner positions
    based on their printed physical size.
  - A ChArUco board is large and complex. It is impractical to attach to a post.
  - solvePnP() only needs the four corners of one flat square marker.
    ChArUco's multi-corner structure provides no additional benefit at runtime.

Usage:
    python generate_charuco.py
    python generate_charuco.py --output boards/charuco_board.png
    python generate_charuco.py --width 2480 --height 3508
"""

import argparse
import os

import cv2
import numpy as np


# Board parameters.
# These must stay identical across generate_charuco.py and calibrate_charuco.py.
# Changing them after calibration invalidates the saved calibration files.

SQUARES_X     = 5       # Number of chessboard squares along the X axis
SQUARES_Y     = 7       # Number of chessboard squares along the Y axis
SQUARE_LENGTH = 0.04    # Physical size of one chessboard square, in meters
MARKER_LENGTH = 0.02    # Physical size of the ArUco marker inside each square, in meters
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50  # Dictionary shared with all ArUco scripts


def create_charuco_board():
    """
    Build and return a CharucoBoard object and its associated dictionary.

    The CharucoBoard object encodes:
      - The grid dimensions (squaresX x squaresY)
      - The physical square and marker sizes
      - The ArUco dictionary used for the embedded markers

    Returns
    -------
    board      : cv2.aruco.CharucoBoard
    dictionary : cv2.aruco.Dictionary
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    return board, dictionary


def generate_board_image(board, width_px=2480, height_px=3508):
    """
    Render the ChArUco board to a grayscale image.

    Default dimensions are approximately A4 at 300 DPI (2480 x 3508 pixels).
    Print at exactly that DPI to match the physical square size.

    Parameters
    ----------
    board     : cv2.aruco.CharucoBoard
    width_px  : int, output image width in pixels
    height_px : int, output image height in pixels

    Returns
    -------
    board_img : numpy.ndarray, dtype uint8, shape (height_px, width_px)
    """
    board_img = board.generateImage(
        outSize=(width_px, height_px),
        marginSize=20,   # White border around the board in pixels
        borderBits=1,    # Thickness of the black border around each ArUco marker
    )
    return board_img


def save_board(board_img, output_path):
    """
    Save the board image to disk. Creates parent directories if needed.

    Parameters
    ----------
    board_img   : numpy.ndarray
    output_path : str, file path ending in .png or .jpg
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cv2.imwrite(output_path, board_img)
    print("[OK] ChArUco board saved -> " + output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate a ChArUco calibration board.")
    parser.add_argument(
        "--output", default=os.path.join("boards", "charuco_board.png"),
        help="Output path for the board image (default: boards/charuco_board.png)"
    )
    parser.add_argument(
        "--width", type=int, default=2480,
        help="Image width in pixels (default: 2480, which is A4 at 300 DPI)"
    )
    parser.add_argument(
        "--height", type=int, default=3508,
        help="Image height in pixels (default: 3508, which is A4 at 300 DPI)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  ChArUco Calibration Board Generator")
    print("=" * 50)
    print("  Dictionary   : DICT_4X4_50")
    print("  Squares X    : " + str(SQUARES_X))
    print("  Squares Y    : " + str(SQUARES_Y))
    print("  Square size  : " + str(SQUARE_LENGTH * 100) + " cm")
    print("  Marker size  : " + str(MARKER_LENGTH * 100) + " cm")
    print("  Image size   : " + str(args.width) + " x " + str(args.height) + " px")
    print("=" * 50)

    board, _ = create_charuco_board()
    board_img = generate_board_image(board, args.width, args.height)
    save_board(board_img, args.output)

    # Resize to fit on screen for display only. Does not affect the saved file.
    preview_w = 700
    preview_h = int(700 * args.height / args.width)
    display = cv2.resize(board_img, (preview_w, preview_h))
    cv2.imshow("ChArUco Board Preview - press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("")
    print("Print the board at " + str(args.width) + " x " + str(args.height) + " px on flat, rigid material.")
    print("Measure the printed square size and update SQUARE_LENGTH if it differs from 4 cm.")


if __name__ == "__main__":
    main()
