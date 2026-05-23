"""
generate_aruco_tags.py
======================
Generates printable standalone ArUco marker images matching the physical
competition constraints.

PHYSICAL COMPETITION CONSTRAINTS
----------------------------------
  - Two posts in the field
  - Each post has three faces (for 360-degree visibility)
  - Each face is 20 cm x 20 cm
  - Each face displays one ArUco marker from DICT_4X4_50
  - The same marker ID appears on all three faces of a given post

This script generates high-resolution marker images that can be printed,
cut out, and attached to the physical post faces.

WHAT ARE ARUCO MARKER DICTIONARIES?
--------------------------------------
An ArUco dictionary defines:
  - How many bits are in each marker's binary grid (here: 4x4 = 16 bits)
  - How many unique IDs the dictionary contains (here: 50 IDs, numbered 0-49)
  - The specific binary pattern for each ID, chosen to maximize the Hamming
    distance between any two patterns so misdetection is minimized

DICT_4X4_50 was chosen because:
  - A 4x4 grid is easy to detect even at medium distances and modest resolution
  - 50 IDs is more than enough for two-post competition setups
  - Larger dictionaries (6x6, 7x7) are more robust but require higher resolution
    cameras and closer range to reliably decode at 20-30+ meters

WHY UNIQUE IDs MATTER
-----------------------
Each post should carry a marker with a unique ID. This allows the detection
pipeline to:
  - Identify which post it is looking at (by ID)
  - Avoid confusing returns from different posts
  - Simultaneously track multiple markers when both posts are visible

HOW PHYSICAL MARKER SIZE AFFECTS POSE ESTIMATION
--------------------------------------------------
The solvePnP() function computes 3D pose by comparing:
  - The known 3D positions of the marker corners in the real world (meters)
  - The observed 2D positions of those corners in the image (pixels)

The 3D positions are derived directly from PHYSICAL_MARKER_SIZE:
  corner[0] = (-half,  half, 0)  # top-left
  corner[1] = ( half,  half, 0)  # top-right
  corner[2] = ( half, -half, 0)  # bottom-right
  corner[3] = (-half, -half, 0)  # bottom-left

If PHYSICAL_MARKER_SIZE is wrong (e.g., you measure 20 cm but set 10 cm),
all distance and position estimates will be scaled by the wrong factor.
Always measure the printed marker and set this value to the actual size.

Usage:
    python generate_aruco_tags.py
    python generate_aruco_tags.py --ids 0 1 2 3
    python generate_aruco_tags.py --out generated_tags/ --size 800 --margin 40
    python generate_aruco_tags.py --no-label
"""

import argparse
import os

import cv2
import numpy as np


# The physical face size for the competition posts.
# This value is only used for labeling purposes here.
# The detect_pose.py script uses this same value as the marker_size parameter
# when calling solvePnP(), so they must match.
PHYSICAL_MARKER_SIZE = 0.20  # 20 cm face = 0.20 meters

# ArUco dictionary to use. Must match detect_pose.py exactly.
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# IDs to generate by default (one per post face; posts could use 0, 1 or more)
DEFAULT_IDS = list(range(10))  # IDs 0 through 9


def generate_aruco_image(marker_id, image_size_px=800, margin_px=40, include_label=True):
    """
    Generate a single high-resolution ArUco marker image ready for printing.

    The marker is centered in the image with an optional white margin on all
    four sides. A text label showing the marker ID is printed below the marker.

    Parameters
    ----------
    marker_id      : int, ArUco marker ID (must be valid for DICT_4X4_50: 0-49)
    image_size_px  : int, total image size in pixels (square canvas)
    margin_px      : int, white margin around the marker in pixels
    include_label  : bool, if True print the marker ID below the marker image

    Returns
    -------
    canvas : numpy.ndarray, shape (image_size_px, image_size_px), dtype uint8
             Grayscale image suitable for saving as PNG.

    Raises
    ------
    ValueError if marker_id is out of range for the dictionary.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    # The marker rendering area is the canvas minus margins on all four sides.
    # If a label is included, reserve extra space at the bottom.
    label_height = 60 if include_label else 0
    marker_area  = image_size_px - 2 * margin_px - label_height

    if marker_area <= 0:
        raise ValueError(
            "margin_px is too large for the given image_size_px. "
            "Reduce margin or increase image size."
        )

    # Generate the raw marker image.
    # cv2.aruco.generateImageMarker() renders the binary pattern for the given
    # marker ID as a square grayscale image. The borderBits parameter adds a
    # mandatory black border around the data cells; this border is part of the
    # marker specification and is required for detection.
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_area, borderBits=1)

    # Create a white canvas and paste the marker in the center.
    canvas = np.ones((image_size_px, image_size_px), dtype=np.uint8) * 255

    y_start = margin_px
    y_end   = margin_px + marker_area
    x_start = margin_px
    x_end   = margin_px + marker_area

    canvas[y_start:y_end, x_start:x_end] = marker_img

    # Draw a thin black border around the marker (visual aid for cutting).
    border_thickness = 2
    cv2.rectangle(
        canvas,
        (x_start - border_thickness, y_start - border_thickness),
        (x_end   + border_thickness, y_end   + border_thickness),
        color=0,
        thickness=border_thickness,
    )

    # Add a text label below the marker showing the ID and physical size.
    if include_label:
        label = "ARUCO ID: " + str(marker_id) + "  |  DICT_4X4_50  |  " + \
                str(int(PHYSICAL_MARKER_SIZE * 100)) + " cm face"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness  = 2

        # Center the text horizontally.
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(0, (image_size_px - text_w) // 2)
        text_y = y_end + margin_px // 2 + text_h

        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            font,
            font_scale,
            color=0,  # Black text
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return canvas


def generate_all_tags(ids, out_dir, image_size_px=800, margin_px=40, include_label=True):
    """
    Generate and save marker images for every ID in the provided list.

    Parameters
    ----------
    ids            : list of int, ArUco marker IDs to generate
    out_dir        : str, output directory path
    image_size_px  : int, canvas size for each marker image
    margin_px      : int, white margin in pixels
    include_label  : bool, whether to include the ID label below the marker

    Returns
    -------
    paths : list of str, file paths of all saved images
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    print("Generating " + str(len(ids)) + " ArUco tag(s) -> " + out_dir + "/")
    print("Canvas: " + str(image_size_px) + " x " + str(image_size_px) + " px  |  "
          "Margin: " + str(margin_px) + " px  |  "
          "Physical size: " + str(int(PHYSICAL_MARKER_SIZE * 100)) + " cm")
    print("")

    for marker_id in ids:
        try:
            img = generate_aruco_image(
                marker_id,
                image_size_px=image_size_px,
                margin_px=margin_px,
                include_label=include_label,
            )
        except ValueError as err:
            print("  [ERROR] ID " + str(marker_id) + ": " + str(err))
            continue

        filename = "tag_" + str(marker_id) + ".png"
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, img)
        paths.append(path)
        print("  [OK] Saved: " + filename)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate printable ArUco marker images for the competition posts."
    )
    parser.add_argument(
        "--ids", nargs="+", type=int, default=DEFAULT_IDS,
        help="Marker IDs to generate (default: 0 through 9). Example: --ids 0 1 2"
    )
    parser.add_argument(
        "--out", default="generated_tags",
        help="Output directory for generated tag images (default: generated_tags/)"
    )
    parser.add_argument(
        "--size", type=int, default=800,
        help="Canvas size in pixels, square (default: 800). Higher = sharper print."
    )
    parser.add_argument(
        "--margin", type=int, default=40,
        help="White margin around marker in pixels (default: 40)"
    )
    parser.add_argument(
        "--no-label", action="store_true",
        help="Omit the ID label below each marker"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display each generated marker in a window"
    )
    args = parser.parse_args()

    # Validate IDs against the dictionary size (DICT_4X4_50 has IDs 0-49)
    max_id = 49
    invalid = [i for i in args.ids if i < 0 or i > max_id]
    if invalid:
        print("[ERROR] Invalid IDs for DICT_4X4_50 (valid range: 0-49): " + str(invalid))
        return

    print("=" * 55)
    print("  ArUco Tag Generator")
    print("=" * 55)
    print("  Dictionary       : DICT_4X4_50")
    print("  IDs to generate  : " + str(args.ids))
    print("  Physical face sz : " + str(int(PHYSICAL_MARKER_SIZE * 100)) + " cm (set PHYSICAL_MARKER_SIZE)")
    print("  Output directory : " + args.out + "/")
    print("=" * 55)
    print("")

    paths = generate_all_tags(
        ids=args.ids,
        out_dir=args.out,
        image_size_px=args.size,
        margin_px=args.margin,
        include_label=not args.no_label,
    )

    if args.show:
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                preview = img if max(img.shape) <= 900 else \
                    cv2.resize(img, (700, 700))
                name = os.path.basename(path)
                cv2.imshow(name + "  (any key to advance)", preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    print("")
    print("[Done] " + str(len(paths)) + " tag(s) saved to '" + args.out + "/'.")
    print("")
    print("Printing instructions:")
    print("  - Print each tag at 100% scale on flat, white paper")
    print("  - Measure the black marker area after printing")
    print("  - Update the marker_size parameter in detect_pose.py to match the measured value")
    print("  - Laminate or mount on rigid foam for outdoor use")


if __name__ == "__main__":
    main()
