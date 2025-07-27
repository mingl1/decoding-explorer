import os
import sys

import cv2
import numpy as np
import psutil
from pystackreg.util import to_uint16


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
