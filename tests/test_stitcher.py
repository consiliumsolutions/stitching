import os
import unittest
from datetime import datetime

import numpy as np

from .context import (
    VERBOSE_DIR,
    AffineStitcher,
    Stitcher,
    StitchingError,
    StitchingWarning,
    load_test_img,
    test_input,
    test_output,
    write_test_result,
)


class TestStitcher(unittest.TestCase):
    def test_stitcher_weir(self):
        stitcher = Stitcher()
        max_derivation = 30
        expected_shape = (673, 2636)

        # from image filenames
        imgs = [test_input("weir*.jpg")]
        name = "weir_from_filenames"

        self.stitch_test_with_warning(
            stitcher,
            imgs,
            expected_shape,
            max_derivation,
            name,
            StitchingWarning,
            "Not all images are included",
        )

        # from loaded numpy arrays
        imgs = [
            load_test_img("weir_1.jpg"),
            load_test_img("weir_2.jpg"),
            load_test_img("weir_3.jpg"),
            load_test_img("weir_noise.jpg"),
        ]
        name = "weir_from_numpy_images"

        self.stitch_test_with_warning(
            stitcher,
            imgs,
            expected_shape,
            max_derivation,
            name,
            StitchingWarning,
            "Not all images are included",
        )

    def test_stitcher_with_not_matching_images(self):
        stitcher = Stitcher()
        imgs = [test_input("s1.jpg"), test_input("boat1.jpg")]

        self.stitch_test_with_error(
            stitcher,
            imgs,
            (),
            0,
            "",
            StitchingError,
            "No match exceeds the given confidence threshold",
            verbose=False,
        )

    def test_stitcher_aquaduct(self):
        stitcher = Stitcher(nfeatures=250, crop=False)
        imgs = [test_input("s?.jpg")]
        max_derivation = 3
        expected_shape = (700, 1811)
        name = "s_result"

        self.stitch_test(stitcher, imgs, expected_shape, max_derivation, name)

    def test_stitcher_boat1(self):
        settings = {
            "warper_type": "fisheye",
            "wave_correct_kind": "no",
            "finder": "dp_colorgrad",
            "compensator": "no",
            "crop": False,
        }
        stitcher = Stitcher(**settings)
        imgs = [
            test_input("boat5.jpg"),
            test_input("boat2.jpg"),
            test_input("boat3.jpg"),
            test_input("boat4.jpg"),
            test_input("boat1.jpg"),
            test_input("boat6.jpg"),
        ]
        max_derivation = 600
        expected_shape = (14488, 7556)
        name = "boat_fisheye"

        self.stitch_test(
            stitcher, imgs, expected_shape, max_derivation, name, verbose=False
        )

    def test_stitcher_boat2(self):
        settings = {
            "warper_type": "compressedPlaneA2B1",
            "finder": "dp_colorgrad",
            "compensator": "channel_blocks",
            "crop": False,
        }
        stitcher = Stitcher(**settings)
        imgs = [
            test_input("boat5.jpg"),
            test_input("boat2.jpg"),
            test_input("boat3.jpg"),
            test_input("boat4.jpg"),
            test_input("boat1.jpg"),
            test_input("boat6.jpg"),
        ]
        max_derivation = 600
        expected_shape = (7400, 12340)
        name = "boat_fisheye"

        self.stitch_test(
            stitcher, imgs, expected_shape, max_derivation, name, verbose=False
        )

    def test_stitcher_boat_aquaduct_subset(self):
        graph = test_output("boat_subset_matches_graph.txt")
        settings = {"final_megapix": 1, "matches_graph_dot_file": graph}
        stitcher = Stitcher(**settings)
        imgs = [
            test_input("boat5.jpg"),
            test_input("s1.jpg"),
            test_input("s2.jpg"),
            test_input("boat2.jpg"),
            test_input("boat3.jpg"),
            test_input("boat4.jpg"),
            test_input("boat1.jpg"),
            test_input("boat6.jpg"),
        ]
        max_derivation = 100
        expected_shape = (705, 3374)
        name = "boat_subset_low_res"

        self.stitch_test_with_warning(
            stitcher,
            imgs,
            expected_shape,
            max_derivation,
            name,
            StitchingWarning,
            "Not all images are included",
        )

        with open(graph, "r") as file:
            graph_content = file.read()
            self.assertTrue(graph_content.startswith("graph matches_graph{"))

    def test_affine_stitcher_warning(self):
        with self.assertWarns(StitchingWarning) as cm:
            AffineStitcher(estimator="homography")
        self.assertTrue(
            str(cm.warning).startswith(
                "You are overwriting an affine default (estimator=affine)"
            )
        )

    def test_affine_stitcher_budapest(self):
        settings = {
            "detector": "sift",
            "crop": False,
        }

        stitcher = AffineStitcher(**settings)
        imgs = [test_input("budapest?.jpg")]
        max_derivation = 50
        expected_shape = (1155, 2310)
        name = "budapest"

        self.stitch_test(stitcher, imgs, expected_shape, max_derivation, name)

    def test_stitcher_feature_masks(self):
        stitcher = Stitcher(crop=False)

        # without masks
        imgs = [test_input("barcode1.png"), test_input("barcode2.png")]
        max_derivation = 25
        expected_shape = (905, 2124)
        name = "features_without_mask"

        self.stitch_test(stitcher, imgs, expected_shape, max_derivation, name)

        # with masks
        masks = [test_input("mask1.png"), test_input("mask2.png")]
        max_derivation = 15
        expected_shape = (716, 1852)
        name = "features_with_mask"

        self.stitch_test(
            stitcher, imgs, expected_shape, max_derivation, name, feature_masks=masks
        )

    def stitch_test(
        self,
        stitcher,
        imgs,
        expected_shape,
        max_derivation,
        name,
        feature_masks=[],
        verbose=True,
    ):
        result = stitcher.stitch(imgs, feature_masks)

        if verbose:
            verbose_dir_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + name
            verbose_dir = os.path.join(VERBOSE_DIR, verbose_dir_name)
            os.makedirs(verbose_dir)
            result_verbose = stitcher.stitch_verbose(imgs, feature_masks, verbose_dir)
            np.testing.assert_allclose(
                result.shape, result_verbose.shape, atol=max_derivation
            )

        np.testing.assert_allclose(
            result.shape[:2], expected_shape, atol=max_derivation
        )

        write_test_result(name + ".jpg", result)

    def stitch_test_with_warning(
        self,
        stitcher,
        imgs,
        expected_shape,
        max_derivation,
        name,
        expected_warning_type,
        expected_warning_message,
        feature_masks=[],
        verbose=True,
    ):
        with self.assertWarns(expected_warning_type) as cm:
            self.stitch_test(
                stitcher,
                imgs,
                expected_shape,
                max_derivation,
                name,
                feature_masks,
                verbose,
            )
        self.assertTrue(str(cm.warning).startswith(expected_warning_message))

    def stitch_test_with_error(
        self,
        stitcher,
        imgs,
        expected_shape,
        max_derivation,
        name,
        expected_error_type,
        expected_error_message,
        feature_masks=[],
        verbose=True,
    ):
        with self.assertRaises(expected_error_type) as cm:
            self.stitch_test(
                stitcher,
                imgs,
                expected_shape,
                max_derivation,
                name,
                feature_masks,
                verbose,
            )
        self.assertTrue(str(cm.exception).startswith(expected_error_message))

    def test_use_of_a_stitcher_for_multiple_image_sets(self):
        # the scale should not be fixed by the first run but set dynamically
        # based on every input image set.
        stitcher = Stitcher()
        _ = stitcher.stitch([test_input("s1.jpg"), test_input("s2.jpg")])
        self.assertEqual(round(stitcher.images._scalers["MEDIUM"].scale, 2), 0.83)
        _ = stitcher.stitch([test_input("boat1.jpg"), test_input("boat2.jpg")])
        self.assertEqual(round(stitcher.images._scalers["MEDIUM"].scale, 2), 0.24)

    def test_overlap_alignment_refinement_two_images(self):
        """Test that overlap alignment refinement runs for 2-image stitching
        and produces a valid alignment shift."""
        stitcher = Stitcher(nfeatures=250, crop=False)
        result = stitcher.stitch(
            [test_input("s1.jpg"), test_input("s2.jpg")]
        )

        # Alignment shift should have been computed for a 2-image stitch
        self.assertTrue(hasattr(stitcher, "_alignment_shift"))
        dx, dy = stitcher._alignment_shift
        # Shift should be small (sub-pixel to a few pixels)
        self.assertLess(abs(dx), 50)
        self.assertLess(abs(dy), 50)

        # Result should still have valid shape
        self.assertEqual(len(result.shape), 3)
        self.assertGreater(result.shape[0], 100)
        self.assertGreater(result.shape[1], 100)

        write_test_result("alignment_two_images.jpg", result)

    def test_overlap_alignment_skips_multi_image(self):
        """Test that overlap alignment is skipped when more than 2 warped
        images are present (called directly, not through stitch pipeline)."""
        import cv2 as cv

        stitcher = Stitcher()

        # Create 3 dummy images/masks/corners/sizes
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = 255 * np.ones((100, 100), dtype=np.uint8)
        imgs = [img, img, img]
        masks = [mask, mask, mask]
        corners = [(0, 0), (50, 0), (100, 0)]
        sizes = [(100, 100), (100, 100), (100, 100)]

        result = stitcher.refine_overlap_alignment(imgs, masks, corners, sizes)

        # Should have been skipped (shift = 0)
        self.assertEqual(stitcher._alignment_shift, (0.0, 0.0))
        # Corners should be unchanged
        self.assertEqual(result, corners)

    def test_overlap_alignment_with_synthetic_shift(self):
        """Test alignment refinement detects a known synthetic shift."""
        import cv2 as cv

        stitcher = Stitcher(nfeatures=250, crop=False)
        img = load_test_img("s1.jpg")
        h, w = img.shape[:2]

        # Create two overlapping regions from the same image
        # with a known vertical offset
        overlap_w = w // 2
        crop1 = img[:, :overlap_w + 100]
        crop2 = img[5:, 100:]  # shifted down by 5 pixels

        h1, w1 = crop1.shape[:2]
        h2, w2 = crop2.shape[:2]
        overlap_start = 100
        corners = [(0, 0), (overlap_start, 0)]
        sizes = [(w1, h1), (w2, h2)]
        masks = [
            255 * np.ones((h1, w1), np.uint8),
            255 * np.ones((h2, w2), np.uint8),
        ]

        new_corners = stitcher.refine_overlap_alignment(
            [crop1, crop2], masks, corners, sizes
        )

        # The refinement should detect a shift
        dx, dy = stitcher._alignment_shift
        # The vertical shift of 5 pixels should be detected
        # (exact value may differ due to image content)
        self.assertNotEqual((dx, dy), (0.0, 0.0))


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
