import os

import tifffile
from PyQt6.QtCore import QObject, pyqtSignal
from tifffile import TiffFile

import utils
from model.file_item import FileItem
from model.status_enum import FileStatus
from utils import get_memory_usage_mb


class FileManagerVM(QObject):
    file_list_updated = pyqtSignal(list)
    file_status_updated = pyqtSignal(str, FileStatus)

    def __init__(self):
        super().__init__()
        self.files: dict[str, FileItem] = {}
        self.emitted_files = set()

    def load_folder(self, folder_path):
        to_be_emitted = []
        for file in list_tiff_files(folder_path):
            self.files[file] = FileItem(path=file, metadata={})
            shape, dtype = get_tif_info(file)
            self.files[file].shape = shape
            self.files[file].dtype = str(dtype)
            if file not in self.emitted_files:
                self.emitted_files.add(file)
                to_be_emitted.append(self.files[file])
        self.file_list_updated.emit(to_be_emitted)

    def load_file(self, file_path):
        if os.path.isfile(file_path):
            self.files[file_path] = FileItem(path=file_path, metadata={})
            self.file_list_updated.emit(list(self.files.values()))

    def apply_shading(self, selected_files):
        for f in selected_files:
            image = load_image(f.path, f.max_size)
            corrected = utils.shading_correction(image)
            f.aligned_image = corrected
            f.status = FileStatus.SHADE_CORRECTED
            self.file_status_updated.emit(f.path, f.status)


def list_tiff_files(folder_path):
    tiff_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                tiff_files.append(os.path.join(root, file))
        break  # Stop after the first directory (depth 1)

    return tiff_files


def get_tif_info(path):
    with TiffFile(path) as tif:
        page = tif.pages[0]
        pages = len(tif.pages)
        shape = page.shape  # e.g., (height, width)
        dtype = page.dtype  # optional
        shape = (pages,) + shape
        return shape, dtype


def load_image(file_path, max_size):
    return tifffile.memmap(file_path, shape=(max_size, max_size), mode="r")
