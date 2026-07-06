"""
@time: 2026/02/10
@file: utils.py
@author: WD                     ___       __   ________
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ /
                                ____/|__/      /_____/

"""
import numpy as np
import os
import hashlib
from PIL import Image, ImageOps
from pathlib import Path
from Classification.utils import LOGGER, LOCAL_RANK, DATASET_CACHE_VERSION, is_dir_writeable


def paser_image_label(args):
    """paser one image label."""
    im_file, prefix, cls_names_dict, sar_scene_dict, sar_polar_dict, sar_band_dict, sar_pitch_range, sar_angle_range = args
    # Number (missing, found, empty, corrupt), message
    nf, nc, msg = 0, 0, ""
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = im.size  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in "bmp", f"invalid image format {im.format}. bmp required"
        # if im.format.lower() in {"jpg", "jpeg"}:
        #     with open(im_file, "rb") as f:
        #         f.seek(-2, 2)
        #         if f.read() != b"\xff\xd9":  # corrupt JPEG
        #             ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
        #             msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # Verify labels
        nf = 1  # image found
        img_name = Path(im_file).stem
        # Grass_KU_10_001_0.0_look_1_234_238_61.515_Pulse_24260_25699_HV_Bulldozer(Lift)_256
        img_name_ = img_name.split('_')
        scene = img_name_[0]
        band = img_name_[1]
        pitch = float(img_name_[2])
        angle = float(img_name_[4])
        polar = img_name_[13]
        cls_name = img_name.split('_' + polar + '_')[-1][:-4]

        assert scene in list(sar_scene_dict.keys()), f"invalid scene {scene} in image name {img_name}, required scenes are {sar_scene_dict.keys()}"
        assert band in list(sar_band_dict.keys()), f"invalid band {band} in image name {img_name}, required bands are {sar_band_dict.keys()}"
        assert sar_pitch_range[0] <= pitch <= sar_pitch_range[1], f"invalid pitch {pitch} in image name {img_name}, required pitches range is {sar_pitch_range}"
        assert sar_angle_range[0] <= angle <= sar_angle_range[1], f"invalid angle {angle} in image name {img_name}, required angle range is {sar_angle_range}"
        assert polar in list(sar_polar_dict.keys()), f"invalid polar {polar} in image name {img_name}, required polars are {sar_polar_dict.keys()}"
        assert cls_name in list(cls_names_dict.keys()), f"invalid class name {cls_name} in image name {img_name}, required class names are {cls_names_dict.keys()}"

        cls = cls_names_dict[cls_name]
        scene = sar_scene_dict[scene]
        polar = sar_polar_dict[polar]
        band = sar_band_dict[band]

        return im_file, cls, shape, scene, band, pitch, angle, polar, nf, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, None, None, None, nf, nc, msg]

def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache

def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def save_dataset_cache_file(prefix, path, x, version):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = version  # add cache version
    if is_dir_writeable(path.parent):
        # if path.exists():
        #     path.unlink()  # remove *.cache file if exists
        with open(str(path), "wb") as file:  # context manager here fixes windows async np.save bug
            np.save(file, x)
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")