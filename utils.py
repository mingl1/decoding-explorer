import os
import sys

import cv2
import numpy as np
import psutil
from numpy.typing import NDArray


def to_uint16(arr):
    """
    Converts image data array to a 16 bit unsigned integer array.

    .. warning:: Data loss may occur as data points outside of 16 bit (0-65535) are
       clipped.

    StackReg generally output float arrays. This utility function can be used to convert
    such an output array to 16 bit data by clipping to 16 bit (to remove values < 0 and
    greater 16 bit) and rounding the floats.

    :type arr: array_like (Ni..., M, Nk...)
    :param arr: Source array (usually float)

    :rtype:  ndarray(Ni..., Nj..., Nk...)
    :return: Input array clipped & rounded to unsigned 16 bit integer
    """
    return to_int_dtype(arr, np.uint16)


def to_int_dtype(arr: np.ndarray, dtype: type):
    """
    Converts image data array to an integer array of the given dtype.

    .. warning:: Data loss may occur as data points outside of the datatypes min/max are
       clipped.

    StackReg generally output float arrays. This utility function can be used to convert
    such an output array to integer data by clipping to the bounds of the integer dtype
    and rounding the floats.

    :type arr: array_like (Ni..., M, Nk...)
    :param arr: Source array (usually float)

    :type dtype: type
    :param dtype: Integer datatype (e.g. np.uint16)

    :rtype:  ndarray(Ni..., Nj..., Nk...)
    :return: Input array clipped & rounded to integer dtype
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array"

    if not np.issubdtype(dtype, np.integer):
        return arr

    ii = np.iinfo(dtype)

    return np.round(arr.clip(min=ii.min, max=ii.max)).astype(dtype)


def is_dark_mode():
    if sys.platform == "win32":
        import winreg

        try:
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(
                registry,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return value == 0  # 0 = dark mode, 1 = light mode
        except Exception:
            return False  # fallback to light mode if any error occurs
    else:
        return False


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def shading_correction(bright_layer_array):
    # Load an image
    # image = cv2.imread(filepath)
    bright_layer_array = cv2.cvtColor(
        bright_layer_array, cv2.COLOR_GRAY2BGR
    )  # convert into RGB image
    bright_layer_array = (bright_layer_array / 255).astype(
        np.uint8
    )  # convert from 16 uint to 8 bit
    image = bright_layer_array
    pixel_size = 7  # this is the size of little square to rescale the intensity
    tile_size_n = int((image.shape[0]) / pixel_size)

    def adjust_local_contrast(
        image, clip_limit=2.0, tile_size=(tile_size_n, tile_size_n)
    ):
        # Convert image to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Calculate the percentile values for the L channel
        min_val, max_val = np.percentile(l_channel, (0, 100))

        # Scale the values of the L channel to 0-255 range
        l_channel_scaled = np.array(
            255 * (l_channel - min_val) / (max_val - min_val), dtype=np.uint8
        )

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the scaled L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_channel_clahe = clahe.apply(l_channel_scaled)

        # Merge the adjusted L channel with the original A and B channels
        lab = cv2.merge((l_channel_clahe, a_channel, b_channel))

        # Convert back to BGR color space
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result

    # Adjust local contrast
    adjusted_image = adjust_local_contrast(image)

    # convert into one layer output
    converted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    converted_image = converted_image.astype(np.uint16)
    converted_image = to_uint16(converted_image * 256)

    # Display original and adjusted images
    # cv2.imshow('Original', image)
    # cv2.imshow('Adjusted', adjusted_image)
    # tifffile.imwrite('Original.tif', image)
    # tifffile.imwrite('converted_image.tif', converted_image)

    return converted_image


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size
    return mem_bytes / (1024 * 1024)  # MB

import itertools
import math

def get_std_dev(numbers):
    """Calculates the standard deviation of a list of numbers."""
    if not numbers:
        return 0
    mean = sum(numbers) / len(numbers)
    variance = sum([(x - mean) ** 2 for x in numbers]) / len(numbers)
    return math.sqrt(variance)

def find_min_std_partition(numbers):
    """
    Finds the binary grouping of numbers that minimizes the weighted
    standard deviation using a brute-force approach.
    """
    n = len(numbers)

    min_total_std = float('inf')
    best_groups = ([], [])

    # Iterate through all possible sizes for the first group (from 1 to n-1)
    for k in range(1, n // 2 + 1):
        # Generate all unique combinations of size k for the first group
        for group1_tuple in itertools.combinations(numbers, k):
            group1 = list(group1_tuple)
            
            # The second group contains all numbers not in the first group
            group2 = [num for num in numbers if num not in group1]
            
            # Special handling for duplicate numbers
            if len(group1) + len(group2) != n:
                original_copy = list(numbers)
                group2 = []
                for num in original_copy:
                    if num in group1:
                        group1.remove(num)
                    else:
                        group2.append(num)
                group1 = list(group1_tuple)

            # Calculate standard deviations for each group
            std1 = get_std_dev(group1)
            std2 = get_std_dev(group2)
            
            # Calculate the weighted total standard deviation
            total_std = (len(group1) * std1 + len(group2) * std2) / n
            
            # Check if this partition is better than the current best
            if total_std < min_total_std:
                min_total_std = total_std
                best_groups = [group1, group2]

    return best_groups, min_total_std

def calculate_ncc(img1, img2):
    """
    Calculate NCC (Normalized Cross-Correlation) between two images.

    Args:
        img1: First image (reference/target)
        img2: Second image (aligned)

    Returns:
        NCC value between -1 and 1 (1 = perfect correlation)
    """
    try:
        # Ensure images have the same shape
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]

        # Convert to float to avoid overflow
        img1_float = img1.astype(np.float64)
        img2_float = img2.astype(np.float64)

        # Flatten images
        img1_flat = img1_float.flatten()
        img2_flat = img2_float.flatten()

        # Calculate means
        mean1 = np.mean(img1_flat)
        mean2 = np.mean(img2_flat)

        # Center the data
        img1_centered = img1_flat - mean1
        img2_centered = img2_flat - mean2

        # Calculate NCC
        numerator = np.sum(img1_centered * img2_centered)
        denominator = np.sqrt(np.sum(img1_centered**2) * np.sum(img2_centered**2))

        if denominator == 0:
            return 0.0  # No correlation if one image is constant

        ncc = numerator / denominator
        return ncc

    except Exception as e:
        print(f"Error calculating NCC: {str(e)}")
        return None


def to_uint8(image):
    """Convert image to uint8 with proper scaling"""
    # Check if image is already uint8
    if image.dtype == np.uint8:
        return image

    # Convert to float and scale to 0-255
    img_float = image.astype(np.float32)
    if img_float.max() > img_float.min():  # Check to avoid division by zero
        img_norm = (img_float - img_float.min()) * (
            255.0 / (img_float.max() - img_float.min())
        )
        return img_norm.astype(np.uint8)
    else:
        print("Warning: Image has no variation, returning zeros")
        return np.zeros_like(image, dtype=np.uint8)


def adjust_contrast(
    img: NDArray[np.float32] | NDArray[np.float64], min_percentile=2, max_percentile=98
):
    """Adjust image contrast using percentile-based clipping for float images"""
    # Calculate percentiles
    minval = np.percentile(img, min_percentile)
    maxval = np.percentile(img, max_percentile)

    # Avoid division by zero
    if maxval - minval < 1e-12:
        return np.zeros_like(img)

    # Clip and rescale to [0.0, 1.0]
    img_adjusted = np.clip(img, minval, maxval)
    img_adjusted = (img_adjusted - minval) / (maxval - minval)

    return img_adjusted  # stays float64, values in [0.0, 1.0]
