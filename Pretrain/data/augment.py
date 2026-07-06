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
from Pretrain.utils import LOGGER, colorstr

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0


import torch
import random
import math
from torchvision.transforms import functional as F
from PIL import Image

class RandomResizedCenterCrop(torch.nn.Module):
    """
    随机中心附近裁剪并缩放到指定尺寸。

    参数：
        target_size (tuple): 目标尺寸 (height, width)
        max_offset (float): 中心点最大偏移比例(相对于图像短边),默认为0.2
        area_ratio_range (tuple): 裁剪区域面积占原图面积的比例范围,默认为(0.3, 0.8)
        aspect_ratio_range (tuple): 裁剪区域的宽高比范围,默认为(0.75, 1.33)
    """
    def __init__(self, target_size, max_offset=0.2, area_ratio_range=(0.3, 0.8), aspect_ratio_range=(0.75, 1.33), interpolation=Image.BILINEAR, antialias=True):
        super().__init__()
        self.target_size = target_size
        self.max_offset = max_offset
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.interpolation = interpolation
        self.antialias = antialias
    
    def get_params(self, img):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        # 获取原始图像尺寸
        if isinstance(img, torch.Tensor):
            # 假设Tensor形状为 (C, H, W)
            h, w = img.shape[-2:]
        else:
            w, h = img.size

        # 图像中心点
        center_x = w / 2
        center_y = h / 2

        # 在中心点附近添加随机偏移
        max_pixel_offset = self.max_offset * min(h, w)  # 最大偏移像素数
        offset_x = random.uniform(-max_pixel_offset, max_pixel_offset)
        offset_y = random.uniform(-max_pixel_offset, max_pixel_offset)
        crop_center_x = center_x + offset_x
        crop_center_y = center_y + offset_y

        
        # 随机生成裁剪区域的面积和宽高比
        area_ratio = random.uniform(*self.area_ratio_range)
        aspect_ratio = random.uniform(*self.aspect_ratio_range)

        # print(f"crop_center_x: {crop_center_x}, crop_center_y: {crop_center_y}")
        # print(f"max_pixel_offset: {max_pixel_offset}, offset_x: {offset_x}, offset_y: {offset_y}")
        # print(f"area_ratio: {area_ratio}, aspect_ratio: {aspect_ratio}")

        # 计算裁剪区域的宽度和高度
        crop_area = area_ratio * h * w
        crop_w = math.sqrt(crop_area * aspect_ratio)
        crop_h = crop_area / crop_w

        # 确保宽高不超过图像边界（以中心点为中心）
        left = crop_center_x - crop_w / 2
        top = crop_center_y - crop_h / 2
        right = crop_center_x + crop_w / 2
        bottom = crop_center_y + crop_h / 2

        # 如果裁剪框超出图像边界，则将其拉回边界内
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)

        # print(f"left: {left}, top: {top}, right: {right}, bottom: {bottom}")
        # print(f"crop_w: {crop_w}, crop_h: {crop_h}")
        # print(f"right - left: {right - left}, bottom - top: {bottom - top}")
        # 确保裁剪区域有效（宽高为正）
        if right - left <= 0 or bottom - top <= 0 or right - left < crop_w //2 or bottom - top < crop_h//2:
            # 若无效（极少情况），则回退到中心裁剪
            left = max(0, center_x - w/4)
            top = max(0, center_y - h/4)
            right = min(w, center_x + w/4)
            bottom = min(h, center_y + h/4)

        # 转换为整数坐标（四舍五入）
        left = int(round(left))
        top = int(round(top))
        right = int(round(right))
        bottom = int(round(bottom))
        return top, left, right - left, bottom - top
    
    def forward(self, img):
        """
        参数：
            img (PIL Image or Tensor): 输入图像
        返回：
            PIL Image or Tensor: 处理后的图像
        """
        # 获取随机裁剪参数
        top, left, w, h = self.get_params(img)

        return F.resized_crop(img, top, left, h, w, self.target_size, self.interpolation, antialias=self.antialias)

    def __repr__(self):
        return f"{self.__class__.__name__}(target_size={self.target_size}, max_offset={self.max_offset}, " \
               f"area_ratio_range={self.area_ratio_range}, aspect_ratio_range={self.aspect_ratio_range}, " \
               f"interpolation={self.interpolation}, antialias={self.antialias})"

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

