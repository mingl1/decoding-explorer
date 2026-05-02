import logging
import os
import threading

import numpy as np
import utils
from PIL import Image
from tifffile import TiffFileError

from model.file_item import FileItem
from model.status_enum import FileStatus
from viewmodel.file_io import get_tif_info, list_tiff_files, load_image, status_from_filename

logger = logging.getLogger(__name__)


def folder_loading_task(folder_paths, stop: threading.Event):
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    to_be_emitted = []
    for folder_path in folder_paths:
        tiff_files = list_tiff_files(folder_path)
        total_files = len(tiff_files)

        for i, file in enumerate(tiff_files):
            if stop.is_set():
                return to_be_emitted
            yield int(i / total_files * 100) if total_files else 0, f"Loading files ({i + 1}/{total_files})"

            status = status_from_filename(file)
            try:
                shape, dtype = get_tif_info(file)
            except (OSError, ValueError, TiffFileError):
                logger.warning(f"Skipping invalid TIFF file: {file}")
                continue

            file_item = FileItem(path=file, status=status)
            file_item.shape = shape
            file_item.original_shape = shape
            file_item.dtype = str(dtype)
            file_item.metadata.max_size = (
                min(shape[-2], shape[-1]) if len(shape) >= 2 else 10000
            )
            to_be_emitted.append(file_item)

    yield 100, "Folder loaded"
    return to_be_emitted


def file_loading_task(file_paths, stop: threading.Event):
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    total_files = len(file_paths)
    loaded_files = []

    for idx, file_path in enumerate(file_paths):
        if stop.is_set():
            break

        file_progress = int(idx / total_files * 100) if total_files else 0
        yield file_progress, f"Loading file {idx + 1}/{total_files}..."

        if not os.path.isfile(file_path):
            continue

        status = status_from_filename(file_path)
        file_item = FileItem(path=file_path, status=status)

        try:
            shape, dtype = get_tif_info(file_path)
        except (OSError, ValueError, TiffFileError):
            try:
                arr = np.array(Image.open(file_path))
                shape = arr.shape
                dtype = arr.dtype
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}", exc_info=True)
                yield file_progress, f"Error loading {os.path.basename(file_path)}: {str(e)}"
                continue

        file_item.shape = shape
        file_item.original_shape = shape
        file_item.dtype = str(dtype)
        file_item.metadata.max_size = (
            min(shape[-2], shape[-1]) if len(shape) >= 2 else 10000
        )
        loaded_files.append(file_item)

    yield 100, f"Loaded {total_files} file(s)"
    return loaded_files


def shading_correction_task(selected_files, files, stop: threading.Event):
    to_be_updated = []
    total_files = len(selected_files)

    for i, f in enumerate(selected_files):
        if stop.is_set():
            break

        progress_pct = int(i / total_files * 100) if total_files else 0
        yield progress_pct, f"Reading from disk ({i + 1}/{total_files})"

        my_f = files.get(f.path)
        if not my_f:
            continue

        image = np.array(load_image(f))
        bf_channel = int(f.metadata.reference_channel)
        bright_field = image[bf_channel] if bf_channel < image.shape[0] else image[0]
        max_size = int(f.metadata.max_size)
        bright_field = bright_field[:max_size, :max_size]

        yield progress_pct, f"Applying shading correction ({i + 1}/{total_files})"
        corrected = utils.shading_correction(bright_field)
        my_f.working_image = corrected
        my_f.status = FileStatus.SHADE_CORRECTED
        my_f.metadata.prefix = FileStatus.SHADE_CORRECTED.name.lower()
        to_be_updated.append(my_f)

    yield 100, "Shading correction complete"
    return to_be_updated
