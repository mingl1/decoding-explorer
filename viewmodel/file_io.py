import os

import numpy as np
import tifffile
from tifffile import TiffFile

from model.file_item import FileItem
from model.status_enum import FileStatus


def list_tiff_files(folder_path):
    tiff_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                tiff_files.append(os.path.join(root, file))
        break  # depth 1 only
    return tiff_files


def get_tif_info(path):
    with TiffFile(path) as tif:
        page = tif.pages[0]
        pages = len(tif.pages)
        shape = (pages,) + page.shape
        return shape, page.dtype


def load_image(item: FileItem):
    try:
        return tifffile.memmap(item.path, shape=item.shape, mode="r")
    except ValueError:
        return tifffile.imread(item.path)


def load_and_constrain_image(file_item: FileItem, max_size: int) -> np.ndarray:
    """Load image and apply max_size constraint."""
    img = load_image(file_item)
    if len(img.shape) == 3:
        img = np.array(img)[:, :max_size, :max_size]
    else:
        img = np.array(img)[:max_size, :max_size]
    if file_item.working_image is not None:
        if len(file_item.working_image.shape) == 2:
            img[int(file_item.metadata.reference_channel)] = np.expand_dims(
                file_item.working_image, axis=0
            )
        elif len(file_item.working_image.shape) == 3:
            img = np.array(file_item.working_image)
    return img


def status_from_filename(file_path: str) -> FileStatus:
    filename_base = os.path.basename(file_path).lower()
    for status in FileStatus:
        if status.name.startswith("_"):
            continue
        assert isinstance(status.value, str)
        prefix = status.value.lower() + "_"
        if filename_base.startswith(prefix):
            return status
    return FileStatus.RAW
