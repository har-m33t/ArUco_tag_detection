# ArUco Tag Detection - 6DOF Pose Estimation

A production-quality Python computer vision project for standalone ArUco marker
detection and 6-DOF pose estimation using a calibrated monocular camera.

Built for a robotics rover that must locate and approach competition posts
equipped with ArUco markers in an outdoor environment.

---

## Project Structure

```
ArUco_tag_detection/
|
|-- generate_charuco.py         Step 1: Generate ChArUco calibration board
|-- calibrate_charuco.py        Step 2: Calibrate camera from board photos
|-- calibrate.py                Alias for calibrate_charuco.py
|-- generate_aruco_tags.py      Step 3: Generate printable ArUco marker images
|-- detect_pose.py              Step 4: Detect markers + estimate 6-DOF pose
|-- utils.py                    Shared helper functions
|-- main.py                     Unified CLI entry point
|-- requirements.txt
|
|-- boards/
|   `-- charuco_board.png       Print this board for camera calibration
|
|-- test_images/
|   |-- calibration_imgs/       Drop ChArUco board photos here from your phone
|   `-- aruco_tags/             Drop ArUco tag photos here from your phone
|
|-- generated_tags/
|   |-- tag_0.png               Print these and attach to competition posts
|   |-- tag_1.png
|   `-- ...
|
`-- calibration/
    |-- camera_matrix.npy       Saved automatically by calibrate_charuco.py
    `-- dist_coeffs.npy
```

---

## Installation

**Requirements:** Python 3.10 or newer.

```
pip install -r requirements.txt
```

> **Important:** Install `opencv-contrib-python`, NOT `opencv-python`.
> The contrib build includes `cv2.aruco`. If you already have `opencv-python`,
> remove it first:
>
> ```
> pip uninstall opencv-python
> pip install opencv-contrib-python
> ```

---

## Complete Workflow

### Step 1 - Generate the ChArUco calibration board

```
python main.py generate-board
```

This saves `boards/charuco_board.png` — a 5x7 ChArUco board using DICT_4X4_50.

Print the board at 300 DPI on flat, rigid material (foam board or laminated card).
Measure the printed square size with calipers and update `SQUARE_LENGTH` in
`generate_charuco.py` if it differs from 4 cm.

---

### Step 2 - Capture calibration images

Photograph the ChArUco board with your phone. Aim for 15-30 images that cover:
- A range of distances (30 cm to 1.5 m from the board)
- Varied tilt angles (left/right/up/down, rotated in plane)
- All regions of the camera frame (corners, center, edges)

Transfer the photos to:

```
test_images/calibration_imgs/
```

---

### Step 3 - Calibrate the camera

```
python main.py calibrate
python main.py calibrate --show        # visualize detections image-by-image
```

Output:
- Prints the reprojection error (target: less than 0.5 px; excellent: less than 0.3 px)
- Saves `calibration/camera_matrix.npy`
- Saves `calibration/dist_coeffs.npy`

If the reprojection error is above 1.0 px, capture more calibration images
from more varied angles and re-run.

---

### Step 4 - Generate printable ArUco tags

```
python main.py generate-tags
python main.py generate-tags --ids 0 1 2 3
python main.py generate-tags --ids 0 1 --size 1000 --margin 50 --show
```

This creates `generated_tags/tag_0.png`, `tag_1.png`, etc.

**Competition setup:**
- Post 1: print tag_0.png on all three faces (same ID for 360-degree visibility)
- Post 2: print tag_1.png on all three faces
- Print at 100% scale so the black marker area is exactly 20 cm x 20 cm
- Laminate or mount on rigid foam for outdoor use

---

### Step 5 - Detect markers and estimate pose

**Transfer photos of the ArUco tags from your phone to:**

```
test_images/aruco_tags/
```

**Then run:**

```
python main.py detect                                           # auto-picks first image
python main.py detect --image test_images/aruco_tags/img.jpg   # specific image
python main.py detect --image test_images/aruco_tags/img.jpg --save  # save annotated result
python main.py detect --webcam                                  # live webcam feed
python main.py detect --webcam --cam-index 1                    # second camera
```

**Webcam keyboard controls:**

| Key     | Action                                  |
|---------|-----------------------------------------|
| q / ESC | Quit                                    |
| s       | Save snapshot to snapshots/             |
| u       | Toggle undistortion on/off              |
| p       | Toggle pose smoothing on/off            |

---

## Visualization Output

The annotated image shows:

| Element            | Description                                          |
|--------------------|------------------------------------------------------|
| Cyan box           | Detected marker boundary                             |
| Colored corner dots| Red=top-left, Green=top-right, Blue=bottom-right, Yellow=bottom-left |
| Red axis line      | Marker local +X (right)                              |
| Green axis line    | Marker local +Y (down)                               |
| Blue axis line     | Marker local +Z (forward, out of marker surface)     |
| Orange arrow       | 3D arrow projected along local +Z                    |
| Text HUD           | Marker ID, X/Y/Z in meters, distance, rotation angles|

---

## Key Computer Vision Concepts

### Architecture: ChArUco for calibration only, standalone ArUco at runtime

ChArUco boards are used **only** for camera calibration. They are never used
during live detection or pose estimation. At runtime, the system detects only
small standalone ArUco markers.

**Why ChArUco for calibration:**
- Embeds unique ArUco marker IDs inside a chessboard grid
- Partial board views are still usable because IDs identify each visible corner
- Sub-pixel accurate corner positions from the chessboard geometry
- Robust to partial occlusion and uneven lighting

### Camera intrinsic calibration

The camera matrix K maps 3D world points to 2D pixel positions:

```
pixel = K * [R | t] * world_point

K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]

fx, fy : focal lengths in pixels
cx, cy : optical center (principal point) in pixels
```

Distortion coefficients (k1, k2, p1, p2, k3) describe how the lens warps
straight lines. Both K and the distortion coefficients are measured during
calibration and saved to disk as .npy files.

### solvePnP and the Perspective-n-Point problem

solvePnP() finds the rotation (rvec) and translation (tvec) that project the
known 3D marker corners onto the detected 2D image corners. This is the
Perspective-n-Point (PnP) problem:

```
s * [u, v, 1]^T = K * [R | t] * [X, Y, Z, 1]^T
```

We use SOLVEPNP_IPPE_SQUARE, a closed-form solver optimized for planar square
targets. It is faster and more accurate than the generic iterative solver.

### Coordinate system (OpenCV camera frame)

```
+X : right
+Y : down
+Z : forward (into the scene, away from the camera)

tvec = [tx, ty, tz]
  tx > 0  ->  marker is to the right of center
  ty > 0  ->  marker is below the camera center
  tz > 0  ->  marker is in front of the camera

distance = sqrt(tx^2 + ty^2 + tz^2)
```

### Why physical marker size matters

solvePnP uses the known 3D corner positions (derived from `marker_size`) and
the detected 2D pixel positions to compute pose. If `marker_size` is wrong,
all distance and position estimates scale by the same wrong factor. Always
measure the printed marker and pass the correct value via `--marker-size`.

---

## Tuning for Outdoor Use

The detector uses adaptive thresholding, which adjusts the binarization
threshold locally per pixel region. This handles bright sunlight, shadows, and
glare much better than a single global threshold.

Key parameters in `build_detector()` inside `detect_pose.py`:

| Parameter                  | Value | Reason                                        |
|----------------------------|-------|-----------------------------------------------|
| adaptiveThreshWinSizeMax   | 53    | Handles blur and low contrast at long range   |
| cornerRefinementMethod     | SUBPIX| Sub-pixel corners improve solvePnP accuracy   |
| minMarkerPerimeterRate     | 0.01  | Allows detection of small/distant markers     |

---

## API Reference

### utils.py

| Function                                  | Description                                      |
|-------------------------------------------|--------------------------------------------------|
| `load_calibration(matrix_path, dist_path)`| Load .npy calibration files from disk            |
| `rvec_to_rotation_matrix(rvec)`           | Rodrigues rotation vector to 3x3 matrix          |
| `compute_distance(tvec)`                  | Euclidean distance to marker in meters           |
| `draw_text_overlay(image, lines, ...)`    | Multi-line text HUD with drop-shadow             |
| `draw_projected_arrow(image, tvec, ...)`  | Project and draw 3D arrow along marker Z axis    |
| `format_pose_text(tvec, rvec, marker_id)` | Format pose data as a list of display strings    |
| `undistort_image(image, K, dist)`         | Apply lens undistortion to an image              |

### detect_pose.py

| Function                                  | Description                                      |
|-------------------------------------------|--------------------------------------------------|
| `build_detector()`                        | Create a tuned ArucoDetector                     |
| `detect_markers(image, detector)`         | Detect markers, return corners and IDs           |
| `make_marker_object_points(marker_size)`  | Build 3D corner array for solvePnP               |
| `estimate_pose(corners, obj_pts, K, dist)`| Run solvePnP, return rvec and tvec               |
| `draw_pose(image, corners, rvec, tvec, ...)`| Draw all visualizations in-place               |
| `run_on_image(...)`                       | Full static image detection pipeline             |
| `run_webcam(...)`                         | Full live webcam detection pipeline              |

---

## Troubleshooting

| Problem                          | Solution                                                              |
|----------------------------------|-----------------------------------------------------------------------|
| ImportError: cv2.aruco not found | Install `opencv-contrib-python`, not `opencv-python`                 |
| No markers detected              | Verify DICT_4X4_50 matches the printed marker; check lighting        |
| Reprojection error > 1 px        | Capture more calibration images from more varied angles               |
| Pose is noisy in webcam mode     | Enable smoothing with the `p` key                                     |
| Wrong distance estimate          | Measure the printed marker and pass the correct `--marker-size` value |

---

## Requirements

```
opencv-contrib-python >= 4.8.0
numpy >= 1.24.0
Python >= 3.10
```
