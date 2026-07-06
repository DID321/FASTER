"""
@time: 2026/02/10
@file: dataset.py
@author: WD                     ___       __   ________
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ /
                                ____/|__/      /_____/

SAR datasets
"""
# import sys
# sys.path.append('..')
from PIL import Image
import os
import cv2
from pathlib import Path
import math
import torch
import random
import psutil
import numpy as np
from torch.utils.data import Dataset
from multiprocessing.pool import ThreadPool
from itertools import repeat
from tqdm import tqdm
from copy import deepcopy
from Pretrain.utils import LOGGER, LOCAL_RANK, DATASET_CACHE_VERSION
from Pretrain.data.data_utils import paser_image_label, load_dataset_cache_file, get_hash, save_dataset_cache_file
from Pretrain.data.augment import classify_augmentations, classify_transforms


class SupCLSARDataSet(Dataset):
    """
    SAR dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference
    in object classification tasks.

    Attributes:
        img_txt_path (str): Path to the txt file containing image paths.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        prefix (str): Prefix to print in log messages.
        fraction (float): Fraction of dataset to utilize.
        im_files (List[str]): List of image file paths.
        labels (List[Dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (List[Path]): List of numpy file paths.
        cache (str): Cache images to RAM or disk during training.

    Methods:
        get_img_files: Read image files from the specified path.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        get_labels: Get labels method to be implemented by subclasses.
    """

    def __init__(
            self,
            data_root,
            img_txt_path,
            imgsz=224,
            cache=False,
            augment=False,
            prefix="",
            classes=None,
            sar_config=None,
            fraction=1.0,
            crop_fraction=1.0,
            transform=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.img_txt_path = img_txt_path
        self.imgsz = imgsz
        self.augment = augment
        self.prefix = prefix
        self.class_names_dict = {}
        if classes is not None:
            for k, v in classes.items():
                self.class_names_dict[v] = k
        self.sar_scene_dict = {}
        self.sar_polar_dict = {}
        self.sar_band_dict = {}
        self.sar_pitch_range = []
        self.sar_angle_range = []
        self.init_sar_config(sar_config)
        self.fraction = fraction
        self.crop_fraction = crop_fraction
        self.im_files = self.get_img_files(self.img_txt_path, self.data_root)
        self.sar_info_dict = {"cls": [], "scene": [], "band": [], "pitch": [], "angle": [], "polar": []}
        self.labels = self.get_labels()
        self.ni = len(self.labels)  # number of images

        self.angle_array = np.array(self.sar_info_dict["angle"])

        self.cls_to_indices = {}
        self.diff_cls_to_indices = {}
        self.composite_to_indices = {}
        self.diff_scene_composite_to_indices = {}
        self.diff_pitch_composite_to_indices = {}
        self.diff_polar_composite_to_indices = {}
        self.diff_polar_raw_idx = {}

        # Cache self.diff_scene_composite_to_indices self.diff_pitch_composite_to_indices self.diff_polar_composite_to_indices self.diff_polar_raw_idx
        self.get_index_mappings()

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None

        if self.cache == "ram" and self.check_cache_ram():
            LOGGER.warning(
                "WARNING ⚠️ cache='ram' may produce non-deterministic training results. "
                "Consider cache='disk' as a deterministic alternative if your disk space allows."
            )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        if transform is None:
            self.transforms = (
                classify_augmentations(
                    size=self.imgsz,
                    scale=None,
                    hflip=0.5,
                    vflip=0.0,
                    erasing=0.0,
                    auto_augment=None,
                    hsv_h=0.015,
                    hsv_s=0.4,
                    hsv_v=0.4,
                )
                if augment
                else classify_transforms(size=self.imgsz, crop_fraction=self.crop_fraction)
            )
        else:
            self.transforms = transform

    def init_sar_config(self, sar_config):
        self.sar_pitch_range = sar_config['pitch']
        self.sar_angle_range = sar_config['angle']
        for i, v in enumerate(sar_config['band']):
            self.sar_band_dict[v] = i
        for i, v in enumerate(sar_config['scene']):
            self.sar_scene_dict[v] = i
        for i, v in enumerate(sar_config['polar']):
            self.sar_polar_dict[v] = i

    def cache_index_mappings(self, path):
        """Precompute index mappings for efficient sample lookup in contrastive learning."""
        path = Path(path)
        x = {"numbers": 7}
        LOGGER.info(f"Total: {x['numbers']} samples. Cache index mappings for efficient sample lookup...")

        # Convert lists to numpy arrays for faster operations
        cls_array = np.array(self.sar_info_dict["cls"])
        scene_array = np.array(self.sar_info_dict["scene"])
        polar_array = np.array(self.sar_info_dict["polar"])
        pitch_array = np.array(self.sar_info_dict["pitch"])

        # Precompute class-based mappings using numpy for better performance
        for cls_idx in np.unique(cls_array):
            self.cls_to_indices[cls_idx] = np.where(cls_array == cls_idx)[0]

        LOGGER.info(f"[1/{x['numbers']}] Precomputed {len(self.cls_to_indices)} class mappings...")

        for cls_idx in np.unique(cls_array):
            self.diff_cls_to_indices[cls_idx] = np.where(cls_array != cls_idx)[0]
        LOGGER.info(f"[2/{x['numbers']}] Precomputed {len(self.diff_cls_to_indices)} different class mappings...")
        # Precompute composite keys for complex queries
        # Key: (cls, pitch, scene, polar) -> indices
        for idx in range(self.ni):
            key = (cls_array[idx], pitch_array[idx], scene_array[idx], polar_array[idx])
            if key not in self.composite_to_indices:
                self.composite_to_indices[key] = []
            self.composite_to_indices[key].append(idx)
        LOGGER.info(f"[3/{x['numbers']}] Precomputed {len(self.composite_to_indices)} composite mappings...")

        # (same cls, pitch, polar, diff scene) -> indices
        for key in self.composite_to_indices:
            cls_, pitch_, scene_, polar_ = key
            if key not in self.diff_scene_composite_to_indices:
                self.diff_scene_composite_to_indices[key] = []
            for scene_idx in np.unique(scene_array):
                if scene_idx != scene_:
                    diff_scene_key = (cls_, pitch_, scene_idx, polar_)
                    # Add indices from the different scene key (but the same cls, pitch, polar)
                    if diff_scene_key in self.composite_to_indices:
                        self.diff_scene_composite_to_indices[key].extend(self.composite_to_indices[diff_scene_key])
                    else:
                        print(diff_scene_key, 'diff_scene_key not exist!')
        LOGGER.info(
            f"[4/{x['numbers']}] Precomputed {len(self.diff_scene_composite_to_indices)} different scene composite mappings...")

        # (same cls, polar, scene, diff pitch) -> indices
        for key in self.composite_to_indices:
            cls_, pitch_, scene_, polar_ = key
            if key not in self.diff_pitch_composite_to_indices:
                self.diff_pitch_composite_to_indices[key] = []
            for pitch in np.unique(pitch_array):
                if pitch != pitch_:
                    diff_pitch_key = (cls_, pitch, scene_, polar_)
                    # Add indices from the different pitch key (but the same cls, polar, scene)
                    if diff_pitch_key in self.composite_to_indices:
                        self.diff_pitch_composite_to_indices[key].extend(self.composite_to_indices[diff_pitch_key])
                    else:
                        print(diff_pitch_key, 'diff_pitch_key not exist!')
        LOGGER.info(
            f"[5/{x['numbers']}] Precomputed {len(self.diff_pitch_composite_to_indices)} different pitch composite mappings...")

        # (same cls, polar, scene, diff polar) -> indices
        for key in self.composite_to_indices:
            cls_, pitch_, scene_, polar_ = key
            if key not in self.diff_polar_composite_to_indices:
                self.diff_polar_composite_to_indices[key] = []
            for polar_idx in np.unique(polar_array):
                if polar_idx != polar_:
                    diff_polar_key = (cls_, pitch_, scene_, polar_idx)
                    # Add indices from the different polar key (but the same cls, pitch, scene)
                    self.diff_polar_composite_to_indices[key].extend(self.composite_to_indices[diff_polar_key])
        LOGGER.info(
            f"[6/{x['numbers']}] Precomputed {len(self.diff_polar_composite_to_indices)} different polar composite mappings...")
        # (same cls, pitch, scene, diff polar) -> rawidx indices
        self.diff_polar_raw_idx = {}
        for key in self.diff_polar_composite_to_indices:
            if key not in self.diff_polar_raw_idx:
                self.diff_polar_raw_idx[key] = []
            for idx in self.diff_polar_composite_to_indices[key]:
                raw_idx = Path(self.im_files[idx]).stem.split("_")[3]
                self.diff_polar_raw_idx[key].append(raw_idx)

        LOGGER.info(
            f"[7/{x['numbers']}] Precomputed {len(self.diff_polar_raw_idx)} different polar raw index mappings...")
        # Convert lists to numpy arrays
        for key in self.composite_to_indices:
            self.composite_to_indices[key] = np.array(self.composite_to_indices[key])
            self.diff_scene_composite_to_indices[key] = np.array(self.diff_scene_composite_to_indices[key])
            self.diff_pitch_composite_to_indices[key] = np.array(self.diff_pitch_composite_to_indices[key])
            self.diff_polar_composite_to_indices[key] = np.array(self.diff_polar_composite_to_indices[key])
            self.diff_polar_raw_idx[key] = np.array(self.diff_polar_raw_idx[key])

        x['cls_to_indices'] = self.cls_to_indices
        x['diff_cls_to_indices'] = self.diff_cls_to_indices
        x['composite_to_indices'] = self.composite_to_indices
        x['diff_scene_composite_to_indices'] = self.diff_scene_composite_to_indices
        x['diff_pitch_composite_to_indices'] = self.diff_pitch_composite_to_indices
        x['diff_polar_composite_to_indices'] = self.diff_polar_composite_to_indices
        x['diff_polar_raw_idx'] = self.diff_polar_raw_idx

        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_img_files(self, img_txt_path, data_root):
        """
            Read image files from the specified path.

            Args:
                img_txt_path (str): Path to split image txt file.
                data_root (str): Path to the root directory containing the raw data.
            Returns:
                (List[str]): List of image file paths.
            Raises:
                FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            with open(img_txt_path, 'r', encoding="utf-8") as f:
                img_files = f.read().strip().splitlines()
                parent = str(data_root) + os.sep
                img_files = [parent + x for x in img_files]
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data paths from {img_txt_path}\n") from e

        return img_files

    def get_labels(self):
        cache_path = self.img_txt_path.replace(".txt", ".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["results"][-1] == len(self.im_files)  # matches found images
            LOGGER.info("Load existsing cache: {}".format(cache_path))
            # assert cache["hash"] == get_hash(self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nc, n = cache.pop("results")  # found, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nc} corrupt"
            tqdm(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly.")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files
        self.sar_info_dict["cls"] = [lb["cls"] for lb in labels]
        self.sar_info_dict["scene"] = [lb["scene"] for lb in labels]
        self.sar_info_dict["band"] = [lb["band"] for lb in labels]
        self.sar_info_dict["pitch"] = [lb["pitch"] for lb in labels]
        self.sar_info_dict["angle"] = [lb["angle"] for lb in labels]
        self.sar_info_dict["polar"] = [lb["polar"] for lb in labels]
        return labels

    def get_index_mappings(self):
        cache_path = self.img_txt_path.replace(".txt", "_index_mappings.cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            LOGGER.info("Load existing index mappings cache: {}".format(cache_path))
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_index_mappings(cache_path), False

        # Display cache
        n = cache.pop("numbers")  # found, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}..."
            tqdm(None, desc=self.prefix + d, total=n, initial=n)  # display results

        # Read cache
        [cache.pop(k) for k in ("version",)]  # remove items

        self.cls_to_indices = cache["cls_to_indices"]
        self.diff_cls_to_indices = cache["diff_cls_to_indices"]
        self.composite_to_indices = cache["composite_to_indices"]
        self.diff_scene_composite_to_indices = cache["diff_scene_composite_to_indices"]
        self.diff_pitch_composite_to_indices = cache["diff_pitch_composite_to_indices"]
        self.diff_polar_composite_to_indices = cache["diff_polar_composite_to_indices"]
        self.diff_polar_raw_idx = cache["diff_polar_raw_idx"]

        if not cache["cls_to_indices"]:
            LOGGER.warning(f"WARNING ⚠️ No mappings found in {cache_path}, training may not work correctly.")
        if not cache["diff_cls_to_indices"]:
            LOGGER.warning(
                f"WARNING ⚠️ No different class mappings found in {cache_path}, training may not work correctly.")
        if not cache["composite_to_indices"]:
            LOGGER.warning(f"WARNING ⚠️ No composite mappings found in {cache_path}, training may not work correctly.")
        if not cache["diff_scene_composite_to_indices"]:
            LOGGER.warning(
                f"WARNING ⚠️ No different scene composite mappings found in {cache_path}, training may not work correctly.")
        if not cache["diff_pitch_composite_to_indices"]:
            LOGGER.warning(
                f"WARNING ⚠️ No different pitch composite mappings found in {cache_path}, training may not work correctly.")
        if not cache["diff_polar_composite_to_indices"]:
            LOGGER.warning(
                f"WARNING ⚠️ No different polar composite mappings found in {cache_path}, training may not work correctly.")
        if not cache["diff_polar_raw_idx"]:
            LOGGER.warning(
                f"WARNING ⚠️ No different polar raw indices found in {cache_path}, training may not work correctly.")

    def cache_labels(self, path):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        path = Path(path)
        x = {"labels": []}
        nf, nc, msgs = 0, 0, []  # number found, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        num_threads = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
        with ThreadPool(num_threads) as pool:
            results = pool.imap(
                func=paser_image_label,
                iterable=zip(
                    self.im_files,
                    repeat(self.prefix),
                    repeat(self.class_names_dict),
                    repeat(self.sar_scene_dict),
                    repeat(self.sar_polar_dict),
                    repeat(self.sar_band_dict),
                    repeat(self.sar_pitch_range),
                    repeat(self.sar_angle_range)
                ),
            )
            pbar = tqdm(results, desc=desc, total=total)
            for im_file, cls, shape, scene, band, pitch, angle, polar, nf_f, nc_f, msg in pbar:
                nf += nf_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": cls,
                            "scene": scene,
                            "band": band,
                            "pitch": pitch,
                            "angle": angle,
                            "polar": polar,
                        }
                    )
                if msg:
                    LOGGER.info("\n".join(msgs))
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            pbar.close()
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No images found in {path}.")
        # x["hash"] = get_hash(self.im_files)
        x["results"] = nf, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def check_cache_ram(self, safety_margin=0.5):
        """
        Check if there's enough RAM for caching images.

        Args:
            safety_margin (float, optional): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files), cv2.IMREAD_GRAYSCALE)  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images ⚠️"
            )
            return False
        return True

    def check_cache_disk(self, safety_margin=0.5):
        """
        Check if there's enough disk space for caching images.

        Args:
            safety_margin (float, optional): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.info(f"{self.prefix}Skipping caching images to disk, directory not writeable ⚠️")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.info(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk ⚠️"
            )
            return False
        return True

    def load_image(self, i, rect_mode=True):
        cv2.setNumThreads(0)
        """
        Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool, optional): Whether to use rectangular resizing.

        Returns:
            (np.ndarray): Loaded image.
            (tuple): Original image dimensions (h, w).
            (tuple): Resized image dimensions (h, w).

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # BGR
            else:  # read image
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            # if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
            #     r = self.imgsz / max(h0, w0)  # ratio
            #     if r != 1:  # if sizes are not equal
            #         w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
            #         im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            # elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
            #     im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self):
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        num_threads = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
        with ThreadPool(num_threads) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = tqdm(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i], cv2.IMREAD_GRAYSCALE), allow_pickle=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """Return transformed label information for given index with optimized sample lookup."""
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        im, _, _ = self.load_image(index)

        # Extract current sample attributes
        cls_idx = label["cls"]
        polar_idx = label["polar"]
        scene_idx = label["scene"]
        pitch = label["pitch"]
        angle = label["angle"]
        img_raw_idx = Path(self.im_files[index]).stem.split("_")[3]

        # Optimized sample lookup using precomputed mappings

        # 1. Negative sample (different class)
        neg_cls_idx = np.random.choice(self.diff_cls_to_indices[cls_idx])
        im_neg_cls, _, _ = self.load_image(neg_cls_idx)

        # 2. Positive sample for scene (different scene, same polar, angle within 10 degrees)
        pos_scene_idx = self._get_pos_scene_sample(cls_idx, pitch, scene_idx, polar_idx, angle)
        im_pos_scene, _, _ = self.load_image(pos_scene_idx)

        # 3. Positive sample for pitch (different pitch, same polar, angle within 10 degrees)
        pos_pitch_idx = self._get_pos_pitch_sample(cls_idx, pitch, scene_idx, polar_idx, angle)
        im_pos_pitch, _, _ = self.load_image(pos_pitch_idx)

        # 4. Positive sample for angle (same scene and polar, different angle)
        pos_angle_idx = self._get_pos_angle_sample(cls_idx, pitch, scene_idx, polar_idx, index)
        im_pos_angle, _, _ = self.load_image(pos_angle_idx)

        # 5. Positive sample for polar (same scene and raw image, different polar)
        pos_polar_idx = self._get_pos_polar_sample(cls_idx, pitch, scene_idx, polar_idx, img_raw_idx)
        im_pos_polar, _, _ = self.load_image(pos_polar_idx)

        # Transform images
        im = Image.fromarray(im)
        label["img"] = self.transforms(im)

        im_neg_cls = Image.fromarray(im_neg_cls)
        label["img_neg_cls"] = self.transforms(im_neg_cls)
        label["neg_cls"] = self.labels[neg_cls_idx]["cls"]
        label["img_neg_cls_file"] = self.im_files[neg_cls_idx]

        im_pos_scene = Image.fromarray(im_pos_scene)
        label["img_pos_scene"] = self.transforms(im_pos_scene)
        label["pos_scene"] = self.labels[pos_scene_idx]["scene"]
        label["img_pos_scene_file"] = self.im_files[pos_scene_idx]

        im_pos_pitch = Image.fromarray(im_pos_pitch)
        label["img_pos_pitch"] = self.transforms(im_pos_pitch)
        label["pos_pitch"] = self.labels[pos_pitch_idx]["pitch"]
        label["img_pos_pitch_file"] = self.im_files[pos_pitch_idx]

        im_pos_angle = Image.fromarray(im_pos_angle)
        label["img_pos_angle"] = self.transforms(im_pos_angle)
        label["pos_angle"] = self.labels[pos_angle_idx]["angle"]
        label["img_pos_angle_file"] = self.im_files[pos_angle_idx]

        im_pos_polar = Image.fromarray(im_pos_polar)
        label["img_pos_polar"] = self.transforms(im_pos_polar)
        label["pos_polar"] = self.labels[pos_polar_idx]["polar"]
        label["img_pos_polar_file"] = self.im_files[pos_polar_idx]

        return label

    def _get_pos_scene_sample(self, cls_idx, pitch, scene_idx, polar_idx, angle):
        """Get positive sample for scene using numpy operations."""
        diff_scene_composite_indices = self.diff_scene_composite_to_indices[(cls_idx, pitch, scene_idx, polar_idx)]
        if len(diff_scene_composite_indices) > 0:
            # Filter by angle within 10 degrees
            angle_diff = np.abs(self.angle_array[diff_scene_composite_indices] - angle)
            angle_filtered = diff_scene_composite_indices[angle_diff <= 10]
            if len(angle_filtered) > 0:
                return int(np.random.choice(angle_filtered))
            else:
                return int(np.random.choice(diff_scene_composite_indices))
        else:
            diff_pitch_composite_indices = self.diff_pitch_composite_to_indices[(cls_idx, pitch, scene_idx, polar_idx)]
            return int(np.random.choice(diff_pitch_composite_indices))

    def _get_pos_pitch_sample(self, cls_idx, pitch, scene_idx, polar_idx, angle):
        """Get positive sample for pitch using numpy operations."""
        diff_pitch_composite_indices = self.diff_pitch_composite_to_indices[(cls_idx, pitch, scene_idx, polar_idx)]
        # Filter by angle within 10 degrees
        angle_diff = np.abs(self.angle_array[diff_pitch_composite_indices] - angle)
        angle_filtered = diff_pitch_composite_indices[angle_diff <= 10]

        if len(angle_filtered) > 0:
            return int(np.random.choice(angle_filtered))
        else:
            return int(np.random.choice(diff_pitch_composite_indices))

    def _get_pos_angle_sample(self, cls_idx, pitch, scene_idx, polar_idx, current_idx):
        """Get positive sample for angle using numpy operations."""
        # Use composite key for efficient lookup
        key = (cls_idx, pitch, scene_idx, polar_idx)
        composite_indices = self.composite_to_indices.get(key, np.array([], dtype=int))
        # Remove current index
        angle_indices = composite_indices[composite_indices != current_idx]
        return int(np.random.choice(angle_indices))

    def _get_pos_polar_sample(self, cls_idx, pitch, scene_idx, polar_idx, img_raw_idx):
        """Get positive sample for polar using numpy operations."""
        diff_polar_composite_indices = self.diff_polar_composite_to_indices[(cls_idx, pitch, scene_idx, polar_idx)]
        diff_polar_raw_idxs = self.diff_polar_raw_idx[(cls_idx, pitch, scene_idx, polar_idx)]
        # Filter by same raw image ID
        mask = diff_polar_raw_idxs == img_raw_idx
        diff_polar_composite_indices = diff_polar_composite_indices[mask]
        return int(np.random.choice(diff_polar_composite_indices))

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "img_neg_cls", "img_pos_scene", "img_pos_pitch", "img_pos_angle", "img_pos_polar",
                     "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"band", "cls", "scene", "pitch", "angle", "polar", "neg_cls", "pos_scene", "pos_pitch",
                     "pos_angle", "pos_polar"}:
                value = torch.tensor(value)
            new_batch[k] = value
        return new_batch


