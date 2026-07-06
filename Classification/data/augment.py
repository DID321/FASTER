# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
@time: 2026/02/11
@file: augment.py
@author: WD                     ___       __   ________
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ /
                                ____/|__/      /_____/

"""
import math
import torch
from Classification.utils import LOGGER, colorstr

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0


# Classification augmentations -----------------------------------------------------------------------------------------
def classify_transforms(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation="BILINEAR",
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    Creates a composition of image transforms for classification tasks.

    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (tuple): Mean values for each RGB channel used in normalization.
        std (tuple): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float): Fraction of the image to be cropped.

    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if isinstance(size, (tuple, list)):
        assert len(size) == 2, f"'size' tuples must be length 2, not length {len(size)}"
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)

    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # Resize the shortest edge to matching target dim for non-square target
        tfl = [T.Resize(scale_size)]
    tfl.extend(
        [
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )
    return T.Compose(tfl)


# Classification training augmentations --------------------------------------------------------------------------------
def classify_augmentations(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter=False,
    erasing=0.0,
    interpolation="BILINEAR",
):
    """
    Creates a composition of image augmentation transforms for classification tasks.

    This function generates a set of image transformations suitable for training classification models. It includes
    options for resizing, flipping, color jittering, auto augmentation, and random erasing.

    Args:
        size (int): Target size for the image after transformations.
        mean (tuple): Mean values for normalization, one per channel.
        std (tuple): Standard deviation values for normalization, one per channel.
        scale (tuple | None): Range of size of the origin size cropped.
        ratio (tuple | None): Range of aspect ratio of the origin aspect ratio cropped.
        hflip (float): Probability of horizontal flip.
        vflip (float): Probability of vertical flip.
        auto_augment (str | None): Auto augmentation policy. Can be 'randaugment', 'augmix', 'autoaugment' or None.
        hsv_h (float): Image HSV-Hue augmentation factor.
        hsv_s (float): Image HSV-Saturation augmentation factor.
        hsv_v (float): Image HSV-Value augmentation factor.
        force_color_jitter (bool): Whether to apply color jitter even if auto augment is enabled.
        erasing (float): Probability of random erasing.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.

    Returns:
        (torchvision.transforms.Compose): A composition of image augmentation transforms.

    Examples:
        >>> transforms = classify_augmentations(size=224, auto_augment="randaugment")
        >>> augmented_image = transforms(original_image)
    """
    # Transforms to apply if Albumentations not installed
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if not isinstance(size, int):
        raise TypeError(f"classify_transforms() size {size} must be integer, not (list, tuple)")
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str), f"Provided argument should be string, but got type {type(auto_augment)}"
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            try:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            except Exception:
                LOGGER.warning('"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.')

        elif auto_augment == "augmix":
            try:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            except Exception:
                LOGGER.warning('"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.')

        elif auto_augment == "autoaugment":
            try:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            except Exception:
                LOGGER.warning('"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.')

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment" or None'
            )

    if not disable_color_jitter:
        secondary_tfl.append(T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h))

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)

