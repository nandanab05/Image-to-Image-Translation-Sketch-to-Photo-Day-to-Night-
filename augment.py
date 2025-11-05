# src/augment.py
import albumentations as A
import numpy as np
import cv2

def get_img2img_transform(height: int, width: int) -> A.Compose:
    """
    Define augmentations for Image-to-Image tasks.
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.1),
        ],
        # 'additional_targets' lets you apply the *exact same* transform
        # to a second "target" image (e.g., the 'photo' for a 'sketch').
        additional_targets={'imageB': 'image'}
    )

def get_crowd_transform(height: int, width: int) -> A.Compose:
    """
    Define augmentations for crowd density tasks.
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.2),
            A.Rotate(limit=15, p=0.2, border_mode=cv2.BORDER_CONSTANT),
        ],
        # This tells albumentations to also transform the (x,y) coordinates
        keypoint_params=A.KeypointParams(format='xy') 
    )

def augment_image(image: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Applies a given transformation to a single image."""
    return transform(image=image)['image']

def augment_paired_images(imageA: np.ndarray, imageB: np.ndarray, transform: A.Compose):
    """Applies the same transformation to two paired images."""
    transformed = transform(image=imageA, imageB=imageB)
    return transformed['image'], transformed['imageB']