"""
Run the improved stitching algorithm on left and right images.

Usage:
    python run_stitch.py <left_image> <right_image> [output_image]

Example:
    python run_stitch.py left.jpg right.jpg result.jpg
"""
import os
import sys
import cv2 as cv
from stitching import Stitcher

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_stitch.py <left_image> <right_image> [output_image]")
        sys.exit(1)

    left_path = sys.argv[1]
    right_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "stitched_result.jpg"

    left = cv.imread(left_path)
    right = cv.imread(right_path)

    if left is None:
        print(f"Error: Could not read {left_path}")
        sys.exit(1)
    if right is None:
        print(f"Error: Could not read {right_path}")
        sys.exit(1)

    print(f"Left image:  {left.shape[1]}x{left.shape[0]}")
    print(f"Right image: {right.shape[1]}x{right.shape[0]}")

    calibration_file = os.path.join(SCRIPT_DIR, "calibration", "16mp", "config.json")

    settings = {
        "detector": "sift",
        "medium_megapix": 1,
        "low_megapix": 0.3,
        "confidence_threshold": 0.1,
        "megapixels": "16",
        "calibration_file": calibration_file,
    }

    stitcher = Stitcher(**settings)
    result = stitcher.stitch([left, right])

    print(f"Alignment correction (low res): {stitcher._alignment_correction}")
    if stitcher._strip_corrections is not None:
        y_centers, dy = stitcher._strip_corrections
        print("Strip corrections (low res):")
        for yc, d in zip(y_centers, dy):
            print(f"  y={yc:.0f}  dy={d:.3f}")
    else:
        print("Strip corrections: none computed")
    print(f"Result: {result.shape[1]}x{result.shape[0]}")

    cv.imwrite(output_path, result)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
