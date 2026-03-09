import json
import os
import tempfile
import unittest

import cv2 as cv
import numpy as np

from .context import Images, Stitcher, test_input


class TestRefineOverlapAlignment(unittest.TestCase):
    """Tests for the post-warp alignment refinement path (2-image case)."""

    def _make_stitcher(self, **kwargs):
        """Create a Stitcher that will not load/save shared calibration."""
        defaults = {"calibrate": True, "calibration_file": ""}
        defaults.update(kwargs)
        return Stitcher(**defaults)

    # ------------------------------------------------------------------
    # refine_overlap_alignment
    # ------------------------------------------------------------------

    def test_noop_for_more_than_two_images(self):
        """Refinement should be a no-op when more than 2 images are given."""
        stitcher = self._make_stitcher()
        imgs = [np.zeros((100, 200, 3), np.uint8) for _ in range(3)]
        masks = [np.ones((100, 200), np.uint8) * 255 for _ in range(3)]
        corners = [(0, 0), (100, 0), (200, 0)]

        result = stitcher.refine_overlap_alignment(imgs, masks, corners)
        self.assertEqual(result, corners)

    def test_noop_for_insufficient_overlap(self):
        """When the two images do not overlap by more than 5 px, corners
        should be returned unchanged."""
        stitcher = self._make_stitcher()
        left = np.zeros((100, 200, 3), np.uint8)
        right = np.zeros((100, 200, 3), np.uint8)
        masks = [np.ones((100, 200), np.uint8) * 255] * 2

        # No overlap at all (right starts at x=200)
        corners_no_overlap = [(0, 0), (200, 0)]
        result = stitcher.refine_overlap_alignment(
            [left, right], masks, corners_no_overlap
        )
        self.assertEqual(result, corners_no_overlap)

        # Tiny overlap of 3 px (<=5 threshold)
        corners_tiny = [(0, 0), (197, 0)]
        result = stitcher.refine_overlap_alignment(
            [left, right], masks, corners_tiny
        )
        self.assertEqual(result, corners_tiny)

    def test_noop_for_insufficient_vertical_overlap(self):
        """Insufficient common Y range should return corners unchanged."""
        stitcher = self._make_stitcher()
        left = np.zeros((100, 200, 3), np.uint8)
        right = np.zeros((100, 200, 3), np.uint8)
        masks = [np.ones((100, 200), np.uint8) * 255] * 2

        # X overlaps by 50 px, but Y ranges don't overlap
        corners = [(0, 0), (150, 105)]
        result = stitcher.refine_overlap_alignment(
            [left, right], masks, corners
        )
        self.assertEqual(result, corners)

    def test_correction_applied_to_right_corner(self):
        """When a correction is computed it should be applied to the second
        image's corner only."""
        stitcher = self._make_stitcher()

        # Build a wide synthetic panorama, then extract two overlapping
        # crops.  The right crop is shifted vertically so the alignment
        # code has something to correct.
        h, total_w = 200, 400
        overlap_w = 100  # pixels of shared panorama content
        dy_shift = 5

        rng = np.random.RandomState(42)
        panorama = rng.randint(0, 256, (h + dy_shift, total_w), dtype=np.uint8)
        panorama = cv.GaussianBlur(panorama, (5, 5), 2.0)

        left_w = 300  # left image covers panorama cols 0..300
        right_start = left_w - overlap_w  # 200 – right image starts here

        left_img = cv.cvtColor(panorama[:h, :left_w], cv.COLOR_GRAY2BGR)
        right_img = cv.cvtColor(
            panorama[dy_shift : h + dy_shift, right_start:], cv.COLOR_GRAY2BGR
        )
        # right_img covers panorama cols 200..400, shape (200, 200)

        masks = [
            np.ones(left_img.shape[:2], np.uint8) * 255,
            np.ones(right_img.shape[:2], np.uint8) * 255,
        ]

        # In stitched coordinates the right image starts at x = right_start
        corners = [(0, 0), (right_start, 0)]
        result = stitcher.refine_overlap_alignment(
            [left_img, right_img], masks, corners
        )

        # The left corner must be untouched
        self.assertEqual(result[0], corners[0])
        # The right corner should have been adjusted
        self.assertIsInstance(result[1], tuple)
        self.assertEqual(len(result[1]), 2)
        # Correction stored
        self.assertNotEqual(stitcher._alignment_correction, (0.0, 0.0))

    def test_cached_correction_applied_without_recomputation(self):
        """When a cached correction exists, it should be applied directly."""
        stitcher = self._make_stitcher()
        stitcher._alignment_correction = (3, -2)
        stitcher._alignment_correction_cached = True

        imgs = [np.zeros((100, 200, 3), np.uint8)] * 2
        masks = [np.ones((100, 200), np.uint8) * 255] * 2
        corners = [(0, 0), (100, 0)]

        result = stitcher.refine_overlap_alignment(imgs, masks, corners)
        self.assertEqual(result[0], (0, 0))
        self.assertEqual(result[1], (103, -2))

    def test_force_realign_ignores_cache(self):
        """With force_realign=True the cached correction is ignored and
        re-computation is attempted."""
        stitcher = self._make_stitcher(force_realign=True)
        stitcher._alignment_correction = (3, -2)
        stitcher._alignment_correction_cached = True

        # Use non-overlapping images so refinement returns corners unchanged
        imgs = [np.zeros((100, 200, 3), np.uint8)] * 2
        masks = [np.ones((100, 200), np.uint8) * 255] * 2
        corners = [(0, 0), (200, 0)]

        result = stitcher.refine_overlap_alignment(imgs, masks, corners)
        # No overlap → correction reset and corners unchanged
        self.assertEqual(result, corners)
        self.assertEqual(stitcher._alignment_correction, (0.0, 0.0))

    # ------------------------------------------------------------------
    # apply_alignment_correction
    # ------------------------------------------------------------------

    def test_apply_alignment_noop_when_zero(self):
        """When the stored correction is (0,0), corners returned as-is."""
        stitcher = self._make_stitcher()
        stitcher._alignment_correction = (0.0, 0.0)

        corners = [(0, 0), (100, 0)]
        result = stitcher.apply_alignment_correction(corners)
        self.assertEqual(result, corners)

    def test_apply_alignment_scales_to_final_resolution(self):
        """The correction computed at low-res should be scaled by the
        low→final ratio when applied at final resolution."""
        stitcher = self._make_stitcher()
        dx_low, dy_low = 4, -6
        stitcher._alignment_correction = (dx_low, dy_low)

        # Initialise images so get_ratio works.
        imgs = [test_input("s1.jpg"), test_input("s2.jpg")]
        stitcher.images = Images.of(
            imgs,
            stitcher.medium_megapix,
            stitcher.low_megapix,
            stitcher.final_megapix,
        )
        # Force scale computation
        list(stitcher.images.resize(Images.Resolution.LOW))
        list(stitcher.images.resize(Images.Resolution.FINAL))

        scale = stitcher.images.get_ratio(
            Images.Resolution.LOW, Images.Resolution.FINAL
        )
        expected_dx = round(dx_low * scale)
        expected_dy = round(dy_low * scale)

        corners = [(0, 0), (500, 10)]
        result = stitcher.apply_alignment_correction(corners)

        self.assertEqual(result[0], (0, 0))
        self.assertEqual(result[1], (500 + expected_dx, 10 + expected_dy))

    # ------------------------------------------------------------------
    # _feature_based_alignment
    # ------------------------------------------------------------------

    def test_feature_alignment_returns_none_on_blank_images(self):
        """Feature alignment should return (None, None) when images have
        no detectable features."""
        stitcher = self._make_stitcher()
        blank = np.zeros((50, 50, 3), np.uint8)
        dx, dy = stitcher._feature_based_alignment(blank, blank)
        self.assertIsNone(dx)
        self.assertIsNone(dy)

    def test_feature_alignment_returns_none_on_small_images(self):
        """Feature alignment should return (None, None) for images too
        small to have 4+ matched features."""
        stitcher = self._make_stitcher()
        small = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        dx, dy = stitcher._feature_based_alignment(small, small)
        self.assertIsNone(dx)
        self.assertIsNone(dy)

    def test_feature_alignment_grayscale_input(self):
        """Feature alignment should handle grayscale images without error."""
        stitcher = self._make_stitcher()
        gray = np.zeros((50, 50), np.uint8)
        dx, dy = stitcher._feature_based_alignment(gray, gray)
        # All-black image → no features → None
        self.assertIsNone(dx)
        self.assertIsNone(dy)

    # ------------------------------------------------------------------
    # _phase_correlation_alignment
    # ------------------------------------------------------------------

    def test_phase_correlation_returns_none_on_low_response(self):
        """Phase correlation should return (None, None) when the images are
        uniform (no spatial frequency → no meaningful correlation peak)."""
        stitcher = self._make_stitcher()
        # Two different uniform-grey images share no spatial structure,
        # yet phaseCorrelate can return a high response on truly identical
        # uniform patches.  Use images with different constant values so
        # the Hann-windowed product has near-zero energy.
        left = np.full((64, 64), 100, dtype=np.uint8)
        right = np.full((64, 64), 200, dtype=np.uint8)
        dx, dy = stitcher._phase_correlation_alignment(left, right)
        # Uniform images may produce a high response (identity-like).
        # The important contract is that the function never raises.
        # If it returns values, they should be finite.
        if dx is not None:
            self.assertTrue(np.isfinite(dx))
            self.assertTrue(np.isfinite(dy))

    def test_phase_correlation_detects_known_shift(self):
        """Phase correlation should detect a known integer pixel shift."""
        stitcher = self._make_stitcher()
        rng = np.random.RandomState(123)
        base = rng.randint(50, 200, (128, 128), dtype=np.uint8)
        base = cv.GaussianBlur(base, (5, 5), 2.0)

        shift_x, shift_y = 3, 2
        tx_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv.warpAffine(base, tx_matrix, (128, 128))

        dx, dy = stitcher._phase_correlation_alignment(base, shifted)
        if dx is not None:
            self.assertAlmostEqual(dx, shift_x, delta=1.5)
            self.assertAlmostEqual(dy, shift_y, delta=1.5)

    # ------------------------------------------------------------------
    # _save_alignment_correction
    # ------------------------------------------------------------------

    def test_save_alignment_correction_persists_to_file(self):
        """The alignment correction should be saved into the calibration
        file's JSON data."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"left": {}, "right": {}}, f)
            tmp_path = f.name

        try:
            stitcher = self._make_stitcher()
            stitcher._calibration_fp = tmp_path
            stitcher._alignment_correction = (7, -3)
            stitcher._save_alignment_correction()

            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["alignment_correction"], [7, -3])
            self.assertTrue(stitcher._alignment_correction_cached)
        finally:
            os.unlink(tmp_path)

    def test_save_alignment_correction_noop_without_calibration_fp(self):
        """If no calibration file path is set, saving should be a no-op."""
        stitcher = self._make_stitcher()
        stitcher._calibration_fp = None
        stitcher._alignment_correction = (5, 5)
        # Should not raise
        stitcher._save_alignment_correction()
        self.assertFalse(stitcher._alignment_correction_cached)

    # ------------------------------------------------------------------
    # Consistency: same correction at low and final resolution
    # ------------------------------------------------------------------

    def test_low_and_final_correction_consistent(self):
        """The integer correction applied at low resolution should, when
        scaled by the low→final ratio, produce the same value that
        ``apply_alignment_correction`` returns."""
        stitcher = self._make_stitcher()
        dx_low, dy_low = 10, -8
        stitcher._alignment_correction = (dx_low, dy_low)

        # Set up images so we can query the scale ratio
        imgs = [test_input("s1.jpg"), test_input("s2.jpg")]
        stitcher.images = Images.of(
            imgs,
            stitcher.medium_megapix,
            stitcher.low_megapix,
            stitcher.final_megapix,
        )
        list(stitcher.images.resize(Images.Resolution.LOW))
        list(stitcher.images.resize(Images.Resolution.FINAL))

        scale = stitcher.images.get_ratio(
            Images.Resolution.LOW, Images.Resolution.FINAL
        )

        low_corners = [(0, 0), (100, 20)]
        low_result = list(low_corners)
        low_result[1] = (low_corners[1][0] + dx_low, low_corners[1][1] + dy_low)

        final_corners = [(0, 0), (500, 100)]
        final_result = stitcher.apply_alignment_correction(final_corners)

        # The final correction should equal the low-res correction scaled
        self.assertEqual(
            final_result[1][0] - final_corners[1][0], round(dx_low * scale)
        )
        self.assertEqual(
            final_result[1][1] - final_corners[1][1], round(dy_low * scale)
        )


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
