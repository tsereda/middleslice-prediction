# transforms.py

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
)

def get_train_transforms(image_size, spacing):
    """
    Returns the MONAI transform pipeline for training.
    This includes loading, standardizing orientation and spacing,
    intensity normalization, cropping, and resizing.
    """
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair", "label"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair", "label"]),
        Orientationd(keys=["t1", "t1ce", "t2", "flair", "label"], axcodes="RAS"),
        Spacingd(
            keys=["t1", "t1ce", "t2", "flair", "label"],
            pixdim=spacing,
            mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["t1", "t1ce", "t2", "flair"],
            a_min=0, a_max=4000, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["t1", "t1ce", "t2", "flair", "label"], source_key="t1"),
        Resized(
            keys=["t1", "t1ce", "t2", "flair", "label"],
            spatial_size=(image_size[0], image_size[1], -1)
        ),
    ])