from types import SimpleNamespace

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
from .stitching_error import StitchingError
from .subsetter import Subsetter
from .timelapser import Timelapser
from .verbose import verbose_stitching
from .warper import Warper

from .timer import Timer
import logging
import gc
gc.disable()

logging.info("LOADED MODULE: stitching/stitcher.py")

class Stitcher:
    DEFAULT_SETTINGS = {
        "medium_megapix": Images.Resolution.MEDIUM.value,
        "detector": FeatureDetector.DEFAULT_DETECTOR,
        "nfeatures": 500,
        "matcher_type": FeatureMatcher.DEFAULT_MATCHER,
        "range_width": FeatureMatcher.DEFAULT_RANGE_WIDTH,
        "try_use_gpu": False,
        "match_conf": None,
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

        self.cameras = None
        self.cameras_registered = False

    def stitch_verbose(self, images, feature_masks=[], verbose_dir=None):
        return verbose_stitching(self, images, feature_masks, verbose_dir)

    def stitch(self, images, feature_masks=[]):
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )

        if not self.cameras_registered:
            imgs = self.resize_medium_resolution()
            features = self.find_features(imgs, feature_masks)
            matches = self.match_features(features)
            imgs, features, matches = self.subset(imgs, features, matches)
            cameras = self.estimate_camera_parameters(features, matches)
            cameras = self.refine_camera_parameters(features, matches, cameras)
            cameras = self.perform_wave_correction(cameras)
            self.estimate_scale(cameras)
            self.cameras = cameras
            self.cameras_registered = True

        low_res_timer = Timer("stitcher.stitch() func: resize_low_resolution")
        imgs = self.resize_low_resolution()
        
        low_res_timer.stop()
        warp_timer = Timer("stitcher.stitch() func: warp_low_resolution")
        
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, self.cameras)
        
        warp_timer.stop()
        prepare_cropper_timer = Timer("stitcher.stitch() func: prepare_cropper")
        
        self.prepare_cropper(imgs, masks, corners, sizes)
        
        prepare_cropper_timer.stop()
        crop_low_res_timer = Timer("stitcher.stitch() func: crop_low_resolution")
        
        imgs, masks, corners, sizes = self.crop_low_resolution(
            imgs, masks, corners, sizes
        )
        
        crop_low_res_timer.stop()
        exposure_error_timer = Timer("stitcher.stitch() func: estimate_exposure_errors")

        self.estimate_exposure_errors(corners, imgs, masks)

        exposure_error_timer.stop()
        seam_finder_timer = Timer("stitcher.stitch() func: find_seam_masks")

        seam_masks = self.find_seam_masks(imgs, corners, masks)

        seam_finder_timer.stop()
        resize_final_res_timer = Timer("stitcher.stitch() func: resize_final_resolution")

        imgs = self.resize_final_resolution()

        resize_final_res_timer.stop()
        warp_final_res_timer = Timer("stitcher.stitch() func: warp_final_resolution")

        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, self.cameras)

        warp_final_res_timer.stop()
        crop_final_res_timer = Timer("stitcher.stitch() func: crop_final_resolution")

        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )

        crop_final_res_timer.stop()
        set_mask_timer = Timer("stitcher.stitch() func: set_masks")

        self.set_masks(masks)
        
        set_mask_timer.stop()
        initialize_composition_timer = Timer("stitcher.stitch() func: initialize_composition")

        #imgs = self.compensate_exposure_errors(corners, imgs)
        
        initialize_composition_timer.stop()
        seam_masks_timer = Timer("stitcher.stitch() func: resize_seam_masks")

        seam_masks = self.resize_seam_masks(seam_masks)

        seam_masks_timer.stop()
        initial_composition_timer = Timer("stitcher.stitch() func: initialize_composition")

        self.initialize_composition(corners, sizes)
        
        initial_composition_timer.stop()
        blend_images_timer = Timer("stitcher.stitch() func: blend_images")
        self.blend_images(imgs, seam_masks, corners)

        blend_images_timer.stop()
        final_panorama_timer = Timer("stitcher.stitch() func: create_final_panorama")

        pano = self.create_final_panorama()

        final_panorama_timer.stop()

        return pano

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
        print("stitcher.blend_images: Type of imgs: ", type(imgs))
        print("stitcher.blend_images: Type of masks: ", type(masks))
        print("stitcher.blend_images: Type of corners: ", type(corners))

        print("stitcher.blend_images: Items in imgs: ", imgs)
        print("stitcher.blend_images: Items in masks: ", masks)
        print("stitcher.blend_images: Items in corners: ", corners)       

        counter = 0
        for_loop_timer = Timer("stitcher.blend_images: for loop")
        #for (img, mask, corner) in (zip(imgs, masks, corners)):
        for corner in corners:
            iter_timer = Timer("stitcher.blend_images: Iteration " + str(counter))
            mask_next_timer = Timer("stitcher.blend_images: next(masks)")
            mask = next(masks)
            mask_next_timer.stop()
            img_next_timer = Timer("stitcher.blend_images: next(imgs)")
            img = next(imgs)
            img_next_timer.stop()
            if self.timelapser.do_timelapse:
                print("TIMELAPSING!")
                self.timelapser.process_and_save_frame(
                    self.images.names[idx], img, corner
                )
            else:
                print("FEEEDING!")
                #print("stitcher.blend_images: img: ", img)
                #print("stitcher.blend_images: mask: ", mask)
                #print("stitcher.blend_images: corner: ", corner)

                self.blender.feed(img, mask, corner)
            counter += 1
            iter_timer.stop()
        for_loop_timer.stop()
        print("stitcher.blend_images: Counter: ", counter)

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

