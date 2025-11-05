# src/preprocess.py
import cv2
import numpy as np
from typing import Tuple

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from a file path using OpenCV.
    Returns:
        A NumPy array in BGR format, or None if loading fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            return None
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resizes an image to the target size.
    Args:
        image: The input image as a NumPy array.
        target_size: A tuple (width, height).
    """
    # cv2.resize expects (width, height)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(image: np.ndarray, mode: str = "gan") -> np.ndarray:
    """
    Normalizes pixel values.
    - 'gan': Normalizes to [-1, 1]
    - 'standard': Normalizes to [0, 1]
    """
    # Convert to float32 first
    image = image.astype(np.float32)
    
    if mode == "gan":
        # (image / 127.5) - 1.0
        return (image / 127.5) - 1.0
    elif mode == "standard":
        # image / 255.0
        return image / 255.0
    else:
        raise ValueError("Invalid normalization mode. Choose 'gan' or 'standard'.")

def save_image(image: np.ndarray, output_path: str):
    """
    Saves a NumPy array as an image file.
    Note: If normalizing to [-1, 1] or [0, 1], you might want to 
    save as .npy files instead of .jpg to preserve float values.
    """
    # Denormalize from [-1, 1] back to [0, 255]
    if image.min() < 0: 
        image = (image + 1.0) * 127.5
    # Denormalize from [0, 1] back to [0, 255]
    elif image.max() <= 1.0:
        image = image * 255.0
        
    cv2.imwrite(output_path, image.astype(np.uint8))