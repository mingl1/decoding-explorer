import os
from email.mime import image
from math import e
from weakref import ref

import numpy as np
import tifffile
from pandas import DataFrame
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from tifffile import TiffFile

import image_processing
import utils
from model.file_item import FileItem
from model.status_enum import FileStatus
from utils import get_memory_usage_mb


class BeadGenerationThread(QThread):
    beads_generated = pyqtSignal(DataFrame)

    def __init__(self, ref_bf, tifs, max_size, signal_to_noise_cutoff):
        super().__init__()
        self.ref_bf = ref_bf
        self.tifs = tifs
        self.max_size = max_size
        self.signal_to_noise_cutoff = signal_to_noise_cutoff

    def run(self):
        results = image_processing.process_beads(
            self.ref_bf,
            self.tifs,
            max_size=self.max_size,
            signal_to_noise_cutoff=self.signal_to_noise_cutoff,
        )
        self.beads_generated.emit(results)


class FileManagerVM(QObject):
    file_list_updated = pyqtSignal(list)
    file_status_updated = pyqtSignal(list)
    file_metadata_updated = pyqtSignal(dict)
    align_progress = pyqtSignal(int, str)
    align_error = pyqtSignal(str)
    align_complete = pyqtSignal(list)
    export_progress = pyqtSignal(int, int)
    beads_generated = pyqtSignal(DataFrame)

    def __init__(self):
        super().__init__()
        self.files: dict[str, FileItem] = {}
        self.reference_item: FileItem | None = None
        self.emitted_files = set()
        self.register_thread = None
        self.bead_thread = None
        self.selected_files = []

    def set_reference_item(self, file_item: FileItem):
        self.reference_item = file_item
        print(f"Reference item set to: {file_item.path}")

    def load_folder(self, folder_path):
        to_be_emitted = []
        for file in list_tiff_files(folder_path):
            self.files[file] = FileItem(path=file)
            shape, dtype = get_tif_info(file)
            self.files[file].shape = shape
            self.files[file].dtype = str(dtype)
            if file not in self.emitted_files:
                self.emitted_files.add(file)
                to_be_emitted.append(self.files[file])
        self.file_list_updated.emit(to_be_emitted)

    def load_file(self, file_path):
        if os.path.isfile(file_path):
            self.files[file_path] = FileItem(path=file_path)
            shape, dtype = get_tif_info(file_path)
            self.files[file_path].shape = shape
            self.files[file_path].dtype = str(dtype)
            if file_path not in self.emitted_files:
                self.emitted_files.add(file_path)
                self.file_list_updated.emit([self.files[file_path]])

    def apply_shading(self, selected_files: list[FileItem]):
        to_be_updated = []
        for f in selected_files:
            image = load_image(f.path, int(f.metadata.max_size))
            bf_channel = int(f.metadata.reference_channel)
            bright_field = (
                image[bf_channel] if bf_channel < image.shape[0] else image[0]
            )
            my_f = self.files.get(f.path)
            if not my_f:
                continue
            corrected = utils.shading_correction(bright_field)
            my_f.working_image = corrected
            my_f.status = FileStatus.SHADE_CORRECTED
            to_be_updated.append(my_f)
        self.file_status_updated.emit(to_be_updated)

    def apply_metadata(self, metadata_changes: dict, selected_files: list[FileItem]):
        for f in selected_files:
            saved_f = self.files.get(f.path)
            if not saved_f:
                continue
            for key, value in metadata_changes.items():
                if hasattr(saved_f.metadata, key):
                    setattr(saved_f.metadata, key, value)
        self.file_metadata_updated.emit(metadata_changes)

    def align_channels(self, selected_files: list[FileItem]):
        if not self.reference_item:
            return
        from align_arrays import Register

        alignable_images = []
        for f in selected_files:
            image = np.array(load_image(f.path, int(f.metadata.max_size)))
            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=0)
            if image.shape[0] < int(f.metadata.reference_channel) + 1:
                self.align_error.emit(
                    f"File {os.path.basename(f.path)} reference channel {f.metadata.reference_channel} exceeds number of channels ({image.shape[0]}). Skipping alignment for this file."
                )
                continue
            image = np.array(image)[
                :, : int(f.metadata.max_size), : int(f.metadata.max_size)
            ]
            if f.working_image is not None:
                image[int(f.metadata.reference_channel)] = np.array(f.working_image)[
                    : int(f.metadata.max_size), : int(f.metadata.max_size)
                ]
            alignable = {
                "image": image,
                "max_size": int(f.metadata.max_size),
                "alignment_layer": int(f.metadata.reference_channel),
                "num_tiles": f.metadata.num_tiles,
                "overlap": f.metadata.overlap,
                "file_path": f.path,
            }
            alignable_images.append(alignable)

        self.register_thread = Register(
            np.array(
                load_image(
                    self.reference_item.path, int(self.reference_item.metadata.max_size)
                )
            )[
                :,
                : int(self.reference_item.metadata.max_size),
                : int(self.reference_item.metadata.max_size),
            ],
            {
                "max_size": int(self.reference_item.metadata.max_size),
                "alignment_layer": int(self.reference_item.metadata.reference_channel),
                "num_tiles": self.reference_item.metadata.num_tiles,
                "overlap": self.reference_item.metadata.overlap,
                "file_path": self.reference_item.path,
            },
            alignable_images,
        )
        self.register_thread.progress.connect(self.align_progress.emit)
        self.register_thread.error.connect(self.align_error.emit)
        self.register_thread.alignment_complete.connect(self.align_complete.emit)
        self.register_thread.alignment_complete.connect(
            lambda aligned: self._on_alignment_complete(aligned, selected_files)
        )
        self.register_thread.run_registration()

    def _on_alignment_complete(
        self, aligned_tifs: list[np.ndarray], selected: list[FileItem]
    ):
        assert self.reference_item is not None
        to_be_updated = []
        for i, f in enumerate(selected):
            my_f = self.files.get(f.path)
            if not my_f:
                continue
            my_f.working_image = aligned_tifs[i]
            my_f.status = FileStatus.ALIGNED
            to_be_updated.append(my_f)
        self.file_status_updated.emit(to_be_updated)

    def cancel_alignment(self):
        if self.register_thread:
            self.register_thread.cancel()

    def set_reference(self, file_item: FileItem):
        if self.reference_item:
            if self.reference_item.path in self.files:
                working_image = self.files[self.reference_item.path].working_image
                if working_image is None:
                    self.files[self.reference_item.path].status = FileStatus.RAW
                elif len(working_image.shape) == 2:
                    self.files[self.reference_item.path].status = (
                        FileStatus.SHADE_CORRECTED
                    )
                elif len(working_image.shape) > 2:
                    self.files[self.reference_item.path].status = FileStatus.ALIGNED
        self.reference_item = file_item
        self.files[file_item.path].status = FileStatus.REFERENCE
        self.file_status_updated.emit([self.files[file_item.path]])

    def export_files(self, folder_path: str, selected_files: list[FileItem]):
        total_files = len(selected_files)
        for i, f in enumerate(selected_files):
            file_item = self.files.get(f.path)
            if not file_item:
                continue
            export_image = load_image(file_item.path, int(file_item.metadata.max_size))
            if file_item.working_image is not None:
                if len(file_item.working_image.shape) == 2:
                    export_image = np.array(export_image)
                    bf_channel = int(file_item.metadata.reference_channel)
                    export_image[bf_channel] = file_item.working_image.squeeze()
                elif len(file_item.working_image.shape) == 3:
                    export_image = np.array(file_item.working_image)
            metadata = {
                "axes": file_item.metadata.axes,
                "unit": file_item.metadata.unit,
                "PhysicalSizeX": file_item.metadata.PhysicalSizeX,
                "PhysicalSizeY": file_item.metadata.PhysicalSizeY,
            }

            file_name = os.path.basename(file_item.path)
            if file_item.metadata.prefix:
                file_name = f"{file_item.metadata.prefix}_{file_name}"

            export_path = os.path.join(folder_path, file_name)
            tifffile.imwrite(export_path, export_image, metadata=metadata)
            self.export_progress.emit(i + 1, total_files)

    def generate_beads(self):
        to_be_updated = []
        assert (
            self.reference_item is not None
        ), "Reference item must be set before generating beads."
        tifs = []
        curr_files = self.selected_files
        curr_files.insert(0, self.reference_item)
        for f in curr_files:
            my_f = self.files.get(f.path)
            if not my_f:
                continue
            img = load_image(f.path, int(f.metadata.max_size))
            if f.working_image is not None and len(f.working_image.shape) == 2:
                img = np.array(img)
                img[int(f.metadata.reference_channel)] = np.expand_dims(
                    f.working_image, axis=0
                )
            elif f.working_image is not None and len(f.working_image.shape) == 3:
                img = np.array(f.working_image)
            tifs.append((img, f))
            my_f.status = FileStatus.BEADS_GENERATED
            to_be_updated.append(my_f)
        ref_bf_path = self.reference_item.path
        ref_bf_channel = int(self.reference_item.metadata.reference_channel)
        ref_img = self.files[self.reference_item.path].working_image
        ref_bf = None
        if ref_img is not None:
            if len(ref_img.shape) == 2:
                ref_bf = ref_img
            elif len(ref_img.shape) == 3:
                ref_bf = np.array(ref_img)
        else:
            ref_img = load_image(
                ref_bf_path, int(self.reference_item.metadata.max_size)
            )
            if len(ref_img.shape) == 3:
                ref_bf = ref_img[ref_bf_channel]
            elif len(ref_img.shape) == 2:
                ref_bf = ref_img
        if ref_bf is None:
            self.align_error.emit(
                "Reference image does not have a valid brightfield channel for bead generation."
            )
            return
        # cy0 cy1 cy2 based on order in tifs, reference is always first
        self.bead_thread = BeadGenerationThread(
            ref_bf,
            tifs,
            max_size=int(self.reference_item.metadata.max_size),
            signal_to_noise_cutoff=0.1,
        )
        self.bead_thread.beads_generated.connect(
            lambda res: self._on_beads_generated(tifs, res)
        )
        self.bead_thread.start()

    def _on_beads_generated(self, tifs, results):
        to_be_updated = []
        ref_bf_path = tifs[0][1].path
        self.files[ref_bf_path].beads = results
        file_items = [f[1] for f in tifs]
        for f in file_items:
            my_f = self.files.get(f.path)
            if not my_f:
                continue
            my_f.status = FileStatus.BEADS_GENERATED
            to_be_updated.append(my_f)
        self.beads_generated.emit(results)
        self.file_status_updated.emit(to_be_updated)


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
    try:
        return tifffile.memmap(file_path, shape=(max_size, max_size), mode="r")
    except ValueError:
        return tifffile.imread(file_path)
