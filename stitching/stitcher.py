import warnings
from types import SimpleNamespace
import cv2 as cv
import numpy as np
import json
import os
from cv2.detail import CameraParams
from pathlib import Path

from .blender import Blender
from .camera_adjuster import CameraAdjuster
from .camera_estimator import CameraEstimator
from .camera_wave_corrector import WaveCorrector
from .cropper import Cropper
from .exposure_error_compensator import ExposureErrorCompensator
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .images import Images
from .seam_finder import SeamFinder
from .stitching_error import StitchingError, StitchingWarning
from .subsetter import Subsetter
from .timelapser import Timelapser
from .verbose import verbose_stitching
from .warper import Warper

def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

class Stitcher:
    DEFAULT_SETTINGS = {
        "medium_megapix": Images.Resolution.MEDIUM.value,
        "detector": FeatureDetector.DEFAULT_DETECTOR,
        "nfeatures": 500,
        "matcher_type": FeatureMatcher.DEFAULT_MATCHER,
        "range_width": FeatureMatcher.DEFAULT_RANGE_WIDTH,
        "try_use_gpu": False,
        "match_conf": None,
        "calibrate": False,
        "calibration_file": None,
        "megapixels": '16',
        "confidence_threshold": Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
        "matches_graph_dot_file": Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
        "estimator": CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
        "adjuster": CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
        "refinement_mask": CameraAdjuster.DEFAULT_REFINEMENT_MASK,
        "wave_correct_kind": WaveCorrector.DEFAULT_WAVE_CORRECTION,
        "warper_type": Warper.DEFAULT_WARP_TYPE,
        "low_megapix": Images.Resolution.LOW.value,
        "crop": Cropper.DEFAULT_CROP,
        "compensator": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
        "nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
        "block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
        "finder": SeamFinder.DEFAULT_SEAM_FINDER,
        "final_megapix": Images.Resolution.FINAL.value,
        "blender_type": Blender.DEFAULT_BLENDER,
        "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
        "timelapse": Timelapser.DEFAULT_TIMELAPSE,
        "timelapse_prefix": Timelapser.DEFAULT_TIMELAPSE_PREFIX,
    }

    def __init__(self, **kwargs):
        self.initialize_stitcher(**kwargs)

    def initialize_stitcher(self, **kwargs):
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.validate_kwargs(kwargs)
        self.kwargs = kwargs
        self.settings.update(kwargs)

        args = SimpleNamespace(**self.settings)
        self.medium_megapix = args.medium_megapix
        self.low_megapix = args.low_megapix
        self.final_megapix = args.final_megapix
        if args.detector in ("orb", "sift"):
            self.detector = FeatureDetector(args.detector, nfeatures=args.nfeatures)
        else:
            self.detector = FeatureDetector(args.detector)
        match_conf = FeatureMatcher.get_match_conf(args.match_conf, args.detector)
        self.matcher = FeatureMatcher(
            args.matcher_type,
            args.range_width,
            try_use_gpu=args.try_use_gpu,
            match_conf=match_conf,
        )
        self.subsetter = Subsetter(
            args.confidence_threshold, args.matches_graph_dot_file
        )
        self.megapixels = args.megapixels

        self.megapixel_options = {
            '4': Path(os.path.expanduser('~/stitching/calibration/4mp/config.json')),
            '16': Path(os.path.expanduser('~/stitching/calibration/16mp/config.json')),
            '64': Path(os.path.expanduser('~/stitching/calibration/64mp/config.json'))
        }

        self.camera_estimator = CameraEstimator(args.estimator)
        self.camera_adjuster = CameraAdjuster(
            args.adjuster, args.refinement_mask, args.confidence_threshold
        )
        self.wave_corrector = WaveCorrector(args.wave_correct_kind)
        self.warper = Warper(args.warper_type)
        self.cropper = Cropper(args.crop)
        self.compensator = ExposureErrorCompensator(
            args.compensator, args.nr_feeds, args.block_size
        )
        self.seam_finder = SeamFinder(args.finder)
        self.blender = Blender(args.blender_type, args.blend_strength)
        self.timelapser = Timelapser(args.timelapse, args.timelapse_prefix)

        self._alignment_shift = (0.0, 0.0)
        self._alignment_low_sizes = []

        if args.calibrate is True:
            self.run_calibration = True
            self.cameras = None
            self.cameras_registered = False
            self.calibration_file = args.calibration_file
        else:
            # Check if calibration file exists, and if it
            if args.calibration_file is not None and args.calibration_file != "":
                if not os.path.isabs(args.calibration_file):
                    fp = os.path.expanduser(
                        f"~/stitching/calibration/{self.megapixels}mp/{args.calibration_file}"
                    )
                else:
                    fp = args.calibration_file
            else:
                fp = self.megapixel_options[self.megapixels]
            file_exists = os.path.exists(fp)
            self.cameras = []
            if file_exists:
                with open(fp, 'r') as f:
                    data = json.load(f)

                    left_params = data['left']
                    right_params = data['right']

                    left_cam = CameraParams()
                    right_cam = CameraParams()
                    self.cameras = [self.setup_cam(left_cam, left_params), self.setup_cam(right_cam, right_params)]

                self.cameras_registered = True
                self.run_calibration = False
                self.estimate_scale(self.cameras)

            else:
                self.cameras = None
                self.cameras_registered = False
                self.run_calibration = True
                self.calibration_file = args.calibration_file

    def setup_cam(self, cam, cam_config):
        cam.aspect = cam_config['aspect']
        cam.focal = cam_config['focal']
        cam.ppx = cam_config['ppx']
        cam.ppy = cam_config['ppy']
        cam.t = np.array(cam_config['t'], dtype=np.float32)
        cam.R = np.array(cam_config['R'], dtype=np.float32)

        return cam

    def stitch_verbose(self, images, feature_masks=[], verbose_dir=None):
        return verbose_stitching(self, images, feature_masks, verbose_dir)

    def calibrate(self, feature_masks):

        imgs = self.resize_medium_resolution()
        features = self.find_features(imgs, feature_masks)
        matches = self.match_features(features)
        imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        self.estimate_scale(cameras)
        self.cameras = cameras
        
        camera_dict = {}

        for idx, camera in enumerate(cameras):

            if idx == 0:
                cam = 'left'
            else:
                cam = 'right'

            camera_dict[cam] = {
                'aspect': camera.aspect, 
                'focal': camera.focal, 
                'ppx': camera.ppx, 
                'ppy': camera.ppy,
                't': camera.t,
                'R': camera.R
            }

        # save to the right calibration file
        if self.calibration_file is not None and self.calibration_file != "":
            if not os.path.isabs(self.calibration_file):
                fp = os.path.expanduser(
                    f"~/stitching/calibration/{self.megapixels}mp/{self.calibration_file}"
                )
            else:
                fp = self.calibration_file
        else:
            fp = self.megapixel_options[self.megapixels]

        with open(fp, 'w') as f:
            json.dump(camera_dict, f, default=convert)

        self.cameras_registered = True
        self.run_calibration = False

    def stitch(self, images, feature_masks=[]):
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )

        if not self.cameras_registered or self.run_calibration:
            self.calibrate(feature_masks)

        imgs = self.resize_low_resolution()
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, self.cameras)
        corners = self.refine_overlap_alignment(imgs, masks, corners, sizes)
        self.prepare_cropper(imgs, masks, corners, sizes)
        imgs, masks, corners, sizes = self.crop_low_resolution(
            imgs, masks, corners, sizes
        )
        self.estimate_exposure_errors(corners, imgs, masks)
        seam_masks = self.find_seam_masks(imgs, corners, masks)

        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, self.cameras)
        corners = self.apply_final_alignment(corners, sizes)
        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )
        self.set_masks(masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        return self.create_final_panorama()

    def resize_medium_resolution(self):
        return list(self.images.resize(Images.Resolution.MEDIUM))

    def find_features(self, imgs, feature_masks=[]):
        if len(feature_masks) == 0:
            return self.detector.detect(imgs)
        else:
            feature_masks = Images.of(
                feature_masks, self.medium_megapix, self.low_megapix, self.final_megapix
            )
            feature_masks = list(feature_masks.resize(Images.Resolution.MEDIUM))
            feature_masks = [Images.to_binary(mask) for mask in feature_masks]
            return self.detector.detect_with_masks(imgs, feature_masks)

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, imgs, features, matches):
        indices = self.subsetter.subset(self.images.names, features, matches)
        imgs = Subsetter.subset_list(imgs, indices)
        features = Subsetter.subset_list(features, indices)
        matches = Subsetter.subset_matches(matches, indices)
        self.images.subset(indices)
        return imgs, features, matches

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def refine_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)

    def estimate_scale(self, cameras):
        self.warper.set_scale(cameras)

    def refine_overlap_alignment(self, imgs, masks, corners, sizes):
        """Refine alignment between warped images using phase correlation.

        For a two-camera fixed setup, after warping with calibrated
        camera parameters, there may be small misalignment in the overlap
        region. This method detects the misalignment using phase
        correlation and adjusts the corner positions to correct it.

        The computed shift is stored internally and can be applied to
        final-resolution corners via apply_final_alignment().
        """
        self._alignment_shift = (0.0, 0.0)
        self._alignment_low_sizes = [tuple(s) for s in sizes]

        if len(imgs) != 2:
            return corners

        img1, img2 = imgs[0], imgs[1]
        mask1, mask2 = masks[0], masks[1]

        x1, y1 = corners[0]
        w1, h1 = sizes[0]
        x2, y2 = corners[1]
        w2, h2 = sizes[1]

        ox_start = max(x1, x2)
        oy_start = max(y1, y2)
        ox_end = min(x1 + w1, x2 + w2)
        oy_end = min(y1 + h1, y2 + h2)

        overlap_w = ox_end - ox_start
        overlap_h = oy_end - oy_start

        min_overlap = 32
        if overlap_w < min_overlap or overlap_h < min_overlap:
            return corners

        roi1_x = ox_start - x1
        roi1_y = oy_start - y1
        roi2_x = ox_start - x2
        roi2_y = oy_start - y2

        crop1 = img1[roi1_y:roi1_y + overlap_h, roi1_x:roi1_x + overlap_w]
        crop2 = img2[roi2_y:roi2_y + overlap_h, roi2_x:roi2_x + overlap_w]

        mcrop1 = mask1[roi1_y:roi1_y + overlap_h, roi1_x:roi1_x + overlap_w]
        mcrop2 = mask2[roi2_y:roi2_y + overlap_h, roi2_x:roi2_x + overlap_w]

        combined_mask = cv.bitwise_and(mcrop1, mcrop2)
        valid_pixels = np.count_nonzero(combined_mask)
        total_pixels = overlap_w * overlap_h
        if valid_pixels / total_pixels < 0.3:
            return corners

        if len(crop1.shape) == 3:
            gray1 = cv.cvtColor(crop1, cv.COLOR_BGR2GRAY).astype(np.float64)
            gray2 = cv.cvtColor(crop2, cv.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray1 = crop1.astype(np.float64)
            gray2 = crop2.astype(np.float64)

        if gray1.shape[0] < 2 or gray1.shape[1] < 2:
            return corners

        mask_float = combined_mask.astype(np.float64) / 255.0
        gray1 = gray1 * mask_float
        gray2 = gray2 * mask_float

        hann = cv.createHanningWindow(
            (gray1.shape[1], gray1.shape[0]), cv.CV_64F
        )
        gray1 = gray1 * hann
        gray2 = gray2 * hann

        (dx, dy), response = cv.phaseCorrelate(gray1, gray2)

        if response < 0.05:
            return corners

        max_shift = min(overlap_w, overlap_h) * 0.15
        dx = float(np.clip(dx, -max_shift, max_shift))
        dy = float(np.clip(dy, -max_shift, max_shift))

        self._alignment_shift = (dx, dy)

        new_corners = list(corners)
        new_corners[1] = (
            corners[1][0] - int(round(dx)),
            corners[1][1] - int(round(dy)),
        )
        return new_corners

    def apply_final_alignment(self, corners, sizes):
        """Apply the alignment shift computed at low resolution to
        final-resolution corners, scaling appropriately."""
        dx, dy = self._alignment_shift
        if dx == 0.0 and dy == 0.0:
            return corners
        if len(corners) != 2:
            return corners

        low_w = self._alignment_low_sizes[0][0]
        low_h = self._alignment_low_sizes[0][1]
        final_w = sizes[0][0]
        final_h = sizes[0][1]

        if low_w == 0 or low_h == 0:
            return corners

        scale_x = final_w / low_w
        scale_y = final_h / low_h

        dx_final = dx * scale_x
        dy_final = dy * scale_y

        new_corners = list(corners)
        new_corners[1] = (
            corners[1][0] - int(round(dx_final)),
            corners[1][1] - int(round(dy_final)),
        )
        return new_corners

    def resize_low_resolution(self, imgs=None):
        return list(self.images.resize(Images.Resolution.LOW, imgs))

    def warp_low_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.LOW)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.LOW
        )
        imgs, masks, corners, sizes = self.warp(imgs, cameras, sizes, camera_aspect)
        return list(imgs), list(masks), corners, sizes

    def warp_final_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.FINAL)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )
        return self.warp(imgs, cameras, sizes, camera_aspect)

    def warp(self, imgs, cameras, sizes, aspect=1):
        imgs = self.warper.warp_images(imgs, cameras, aspect)
        masks = self.warper.create_and_warp_masks(sizes, cameras, aspect)
        corners, sizes = self.warper.warp_rois(sizes, cameras, aspect)
        return imgs, masks, corners, sizes

    def prepare_cropper(self, imgs, masks, corners, sizes):
        self.cropper.prepare(imgs, masks, corners, sizes)

    def crop_low_resolution(self, imgs, masks, corners, sizes):
        imgs, masks, corners, sizes = self.crop(imgs, masks, corners, sizes)
        return list(imgs), list(masks), corners, sizes

    def crop_final_resolution(self, imgs, masks, corners, sizes):
        lir_aspect = self.images.get_ratio(
            Images.Resolution.LOW, Images.Resolution.FINAL
        )
        return self.crop(imgs, masks, corners, sizes, lir_aspect)

    def crop(self, imgs, masks, corners, sizes, aspect=1):
        masks = self.cropper.crop_images(masks, aspect)
        imgs = self.cropper.crop_images(imgs, aspect)
        corners, sizes = self.cropper.crop_rois(corners, sizes, aspect)
        return imgs, masks, corners, sizes

    def estimate_exposure_errors(self, corners, imgs, masks):
        self.compensator.feed(corners, imgs, masks)

    def find_seam_masks(self, imgs, corners, masks):
        return self.seam_finder.find(imgs, corners, masks)

    def resize_final_resolution(self):
        return self.images.resize(Images.Resolution.FINAL)

    def compensate_exposure_errors(self, corners, imgs):
        for idx, (corner, img) in enumerate(zip(corners, imgs)):
            yield self.compensator.apply(idx, corner, img, self.get_mask(idx))

    def resize_seam_masks(self, seam_masks):
        for idx, seam_mask in enumerate(seam_masks):
            yield SeamFinder.resize(seam_mask, self.get_mask(idx))

    def set_masks(self, mask_generator):
        self.masks = mask_generator
        self.mask_index = -1

    def get_mask(self, idx):
        if idx == self.mask_index + 1:
            self.mask_index += 1
            self.mask = next(self.masks)
            return self.mask
        elif idx == self.mask_index:
            return self.mask
        else:
            raise StitchingError("Invalid Mask Index!")

    def initialize_composition(self, corners, sizes):
        if self.timelapser.do_timelapse:
            self.timelapser.initialize(corners, sizes)
        else:
            self.blender.prepare(corners, sizes)

    def blend_images(self, imgs, masks, corners):
        for idx, (img, mask, corner) in enumerate(zip(imgs, masks, corners)):
            if self.timelapser.do_timelapse:
                self.timelapser.process_and_save_frame(
                    self.images.names[idx], img, corner
                )
            else:
                self.blender.feed(img, mask, corner)

    def create_final_panorama(self):
        if not self.timelapser.do_timelapse:
            panorama, _ = self.blender.blend()
            return panorama

    def validate_kwargs(self, kwargs):
        for arg in kwargs:
            if arg not in self.DEFAULT_SETTINGS:
                raise StitchingError("Invalid Argument: " + arg)


class AffineStitcher(Stitcher):
    AFFINE_DEFAULTS = {
        "estimator": "affine",
        "wave_correct_kind": "no",
        "matcher_type": "affine",
        "adjuster": "affine",
        "warper_type": "affine",
        "compensator": "no",
    }

    DEFAULT_SETTINGS = Stitcher.DEFAULT_SETTINGS.copy()
    DEFAULT_SETTINGS.update(AFFINE_DEFAULTS)

    def initialize_stitcher(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.AFFINE_DEFAULTS and value != self.AFFINE_DEFAULTS[key]:
                warnings.warn(
                    f"You are overwriting an affine default ({key}={self.AFFINE_DEFAULTS[key]}) with another value ({value}). Make sure this is intended",  # noqa: E501
                    StitchingWarning,
                )
        super().initialize_stitcher(**kwargs)
