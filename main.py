"""
main.py
=======
Unified command-line entry point for the ArUco pose estimation pipeline.

Sub-commands
------------
  generate-board   : Generate the ChArUco calibration board image
  calibrate        : Calibrate the camera from board photos
  generate-tags    : Generate printable standalone ArUco marker images
  detect           : Detect ArUco markers and estimate 6-DOF pose

Usage
-----
  python main.py generate-board
  python main.py calibrate --images calibration_images/ --show
  python main.py generate-tags --ids 0 1 2 3 --show
  python main.py detect --image test_images/sample.jpg
  python main.py detect --webcam
  python main.py --help

Full Workflow
-------------
  Step 1: python main.py generate-board
          Print boards/charuco_board.png on flat rigid material.

  Step 2: Photograph the ChArUco board 15-30 times from varied angles.
          Drop the photos from your phone into test_images/calibration_imgs/.

  Step 3: python main.py calibrate --show
          Saves calibration/camera_matrix.npy and calibration/dist_coeffs.npy.

  Step 4: python main.py generate-tags --ids 0 1
          Print generated_tags/tag_0.png and tag_1.png.
          Attach to the competition post faces.

  Step 5: Drop ArUco tag photos from your phone into test_images/aruco_tags/.
          python main.py detect
          Or with a specific image: python main.py detect --image path/to/img.jpg
          Or live webcam:          python main.py detect --webcam
"""

import sys


USAGE = """
ArUco Pose Estimation Pipeline

Usage:
  python main.py <command> [options]

Commands:
  generate-board   Generate the ChArUco calibration board image
  calibrate        Calibrate camera from ChArUco board photos
  generate-tags    Generate printable standalone ArUco marker images
  detect           Detect markers and estimate 6-DOF pose

Run  python main.py <command> --help  for per-command options.

Workflow:
  1. python main.py generate-board
  2. Photograph the board, drop photos into  test_images/calibration_imgs/
  3. python main.py calibrate --show
  4. python main.py generate-tags --ids 0 1
  5. Drop ArUco tag photos into  test_images/aruco_tags/
     python main.py detect
"""


def run_generate_board(argv):
    sys.argv = ["generate_charuco.py"] + argv
    from generate_charuco import main
    main()


def run_calibrate(argv):
    sys.argv = ["calibrate_charuco.py"] + argv
    from calibrate_charuco import main
    main()


def run_generate_tags(argv):
    sys.argv = ["generate_aruco_tags.py"] + argv
    from generate_aruco_tags import main
    main()


def run_detect(argv):
    sys.argv = ["detect_pose.py"] + argv
    from detect_pose import main
    main()


COMMANDS = {
    "generate-board": run_generate_board,
    "calibrate":      run_calibrate,
    "generate-tags":  run_generate_tags,
    "detect":         run_detect,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print("[ERROR] Unknown command: '" + command + "'")
        print(USAGE)
        sys.exit(1)

    COMMANDS[command](sys.argv[2:])


if __name__ == "__main__":
    main()
