import logging
import os
from functools import reduce
from typing import List, Optional

import numpy as np
import pandas as pd
import tifffile
from pandas import DataFrame, Series
from PIL import Image
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog
from tifffile import TiffFile, TiffFileError

import image_processing
import utils
from model.file_item import FileItem
from model.status_enum import FileStatus
from view.alignment_preview_dialog import AlignmentPreviewDialog

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
NOT_SENT = "..."

# Bead CSV validation constants
REQUIRED_BEAD_COLUMNS = ["x", "y"]
CYCLE_COLUMN_PREFIX = "cy"


class BeadGenerationThread(QThread):
    progress = pyqtSignal(int, str)
    bead_generated = pyqtSignal(dict)

    def __init__(
        self,
        ref_file,
        file_items: list,
        files: dict,
        signal_to_noise_cutoff,
        use_stardist=False,
        model_name="model_4_400epoch_no_aug",
    ):
        super().__init__()
        self.ref_file = ref_file
        self.file_items = file_items
        self.files = files
        self.signal_to_noise_cutoff = signal_to_noise_cutoff
        self.use_stardist = use_stardist
        self.model_name = model_name
        self._is_running = True

    def run(self):
        self._is_running = True
        try:
            ref_bf_channel = int(self.ref_file.metadata.reference_channel)
            ref_max_size = int(self.ref_file.metadata.max_size)
            ref_img = self.files[self.ref_file.path].working_image
            ref_bf = None
            if ref_img is not None:
                if len(ref_img.shape) == 2:
                    ref_bf = ref_img
                elif len(ref_img.shape) == 3:
                    ref_bf = np.array(ref_img)
            else:
                ref_img = load_image(self.files[self.ref_file.path])
                if len(ref_img.shape) == 3:
                    ref_bf = np.array(ref_img)[
                        ref_bf_channel, :ref_max_size, :ref_max_size
                    ]
                elif len(ref_img.shape) == 2:
                    ref_bf = np.array(ref_img)[:ref_max_size, :ref_max_size]

            if ref_bf is None:
                logger.error(
                    "Reference image does not have a valid brightfield channel for bead generation."
                )
                return

            tifs = []
            total_files = len(self.file_items)
            for i, f in enumerate(self.file_items):
                if not self._is_running:
                    break
                progress_pct = int((i / total_files) * 100)
                self.progress.emit(
                    progress_pct, f"Loading images ({i + 1}/{total_files})"
                )

                my_f = self.files.get(f.path)
                if not my_f:
                    continue
                max_size = int(f.metadata.max_size)
                img = load_and_constrain_image(my_f, max_size)
                tifs.append((img, f))

            if not self._is_running:
                return

            self.progress.emit(90, "Processing beads...")
            progress_offset = 90

            def scaled_progress(p, m):
                if self._is_running:
                    self.progress.emit(min(99, progress_offset + p), m)

            results = image_processing.process_beads(
                ref_bf,
                tifs,
                max_size=ref_max_size,
                signal_to_noise_cutoff=self.signal_to_noise_cutoff,
                progress_callback=scaled_progress,
                is_running_callback=self.is_running,
                use_stardist=self.use_stardist,
                model_name=self.model_name,
            )
            if self._is_running:
                self.bead_generated.emit(results)
        except Exception as e:
            logger.error(f"Error in BeadGenerationThread: {e}", exc_info=True)
        return None

    def cancel(self):
        self._is_running = False

    def is_running(self):
        return self._is_running


class ShadingCorrectionThread(QThread):
    progress = pyqtSignal(int, str)
    shading_complete = pyqtSignal(list)

    def __init__(self, selected_files: list, files: dict):
        super().__init__()
        self.selected_files = selected_files
        self.files = files
        self._is_running = True

    def run(self):
        self._is_running = True
        try:
            to_be_updated = []
            total_files = len(self.selected_files)
            for i, f in enumerate(self.selected_files):
                if not self._is_running:
                    break
                progress_pct = int((i / total_files) * 100)
                self.progress.emit(
                    progress_pct, f"Reading from disk ({i + 1}/{total_files})"
                )

                my_f = self.files.get(f.path)
                if not my_f:
                    continue

                image = np.array(load_image(f))
                bf_channel = int(f.metadata.reference_channel)
                bright_field = (
                    image[bf_channel] if bf_channel < image.shape[0] else image[0]
                )
                max_size = int(f.metadata.max_size)
                bright_field = bright_field[:max_size, :max_size]

                self.progress.emit(
                    progress_pct, f"Applying shading correction ({i + 1}/{total_files})"
                )
                corrected = utils.shading_correction(bright_field)
                my_f.working_image = corrected
                my_f.status = FileStatus.SHADE_CORRECTED
                my_f.metadata.prefix = FileStatus.SHADE_CORRECTED.name.lower()
                to_be_updated.append(my_f)

            if self._is_running:
                self.progress.emit(100, "Shading correction complete")
                self.shading_complete.emit(to_be_updated)
        except Exception as e:
            logger.error(f"Error in ShadingCorrectionThread: {e}", exc_info=True)
            self.progress.emit(-1, f"Error: {str(e)}")

    def cancel(self):
        self._is_running = False


class FolderLoadingThread(QThread):
    progress = pyqtSignal(int, str)
    folder_loaded = pyqtSignal(list)

    def __init__(self, folder_path: List[str] | str):
        super().__init__()
        if isinstance(folder_path, str):
            self.folder_paths = [folder_path]
        else:
            self.folder_paths = folder_path
        self._is_running = True

    def run(self):
        self._is_running = True
        try:
            to_be_emitted = []
            for folder_path in self.folder_paths:
                tiff_files = list_tiff_files(folder_path)
                total_files = len(tiff_files)

                for i, file in enumerate(tiff_files):
                    if not self._is_running:
                        break
                    progress_pct = int((i / total_files) * 100)
                    self.progress.emit(
                        progress_pct, f"Loading files ({i + 1}/{total_files})"
                    )

                    status = self._get_status_from_filename(file)
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

            if self._is_running:
                self.progress.emit(100, "Folder loaded")
                self.folder_loaded.emit(to_be_emitted)
        except Exception as e:
            logger.error(f"Error in FolderLoadingThread: {e}", exc_info=True)
            self.progress.emit(-1, f"Error: {str(e)}")

    def _get_status_from_filename(self, file_path: str) -> FileStatus:
        filename_base = os.path.basename(file_path).lower()
        for status in FileStatus:
            if status.name.startswith("_"):
                continue
            assert isinstance(status.value, str)
            prefix = status.value.lower() + "_"
            if filename_base.startswith(prefix):
                return status
        return FileStatus.RAW

    def cancel(self):
        self._is_running = False


class FileLoadingThread(QThread):
    progress = pyqtSignal(int, str)
    file_loaded = pyqtSignal(list)

    def __init__(self, file_paths: List[str] | str):
        super().__init__()
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths
        self._is_running = True

    def run(self):
        self._is_running = True
        total_files = len(self.file_paths)
        loaded_files = []
        try:
            for idx, file_path in enumerate(self.file_paths):
                if not self._is_running:
                    break

                # Update progress for current file
                file_progress = int((idx / total_files) * 100)
                self.progress.emit(
                    file_progress, f"Loading file {idx + 1}/{total_files}..."
                )

                if os.path.isfile(file_path):
                    status = self._get_status_from_filename(file_path)
                    file_item = FileItem(path=file_path, status=status)

                    try:
                        shape, dtype = get_tif_info(file_path)
                    except (OSError, ValueError, TiffFileError):
                        try:
                            arr = np.array(Image.open(file_path))
                            shape = arr.shape
                            dtype = arr.dtype
                        except Exception as e:
                            logger.error(
                                f"Failed to load {file_path}: {e}", exc_info=True
                            )
                            self.progress.emit(
                                file_progress,
                                f"Error loading {os.path.basename(file_path)}: {str(e)}",
                            )
                            continue  # Skip this file and continue with next

                    file_item.shape = shape
                    file_item.original_shape = shape
                    file_item.dtype = str(dtype)
                    file_item.metadata.max_size = (
                        min(shape[-2], shape[-1]) if len(shape) >= 2 else 10000
                    )

                    if self._is_running:
                        loaded_files.append(file_item)

            # Final progress update
            if self._is_running:
                self.file_loaded.emit(loaded_files)
                self.progress.emit(100, f"Loaded {total_files} file(s)")

        except Exception as e:
            logger.error(f"Error in FileLoadingThread: {e}", exc_info=True)
            self.progress.emit(-1, f"Error: {str(e)}")

    def _get_status_from_filename(self, file_path: str) -> FileStatus:
        filename_base = os.path.basename(file_path).lower()
        for status in FileStatus:
            if status.name.startswith("_"):
                continue
            assert isinstance(status.value, str)
            prefix = status.value.lower() + "_"
            if filename_base.startswith(prefix):
                return status
        return FileStatus.RAW

    def cancel(self):
        self._is_running = False


class ExportThread(QThread):
    progress = pyqtSignal(int, int)
    export_complete = pyqtSignal()
    export_error = pyqtSignal(str)

    def __init__(
        self,
        folder_path: str,
        files: dict,
        selected_files: list,
        metadata_changes: dict | None = None,
    ):
        super().__init__()
        self.folder_path = folder_path
        self.files = files
        self.selected_files = selected_files
        self.metadata_changes = metadata_changes or {}
        self._is_running = True

    def run(self):
        self._is_running = True
        try:
            total_files = len(self.selected_files)
            for i, f in enumerate(self.selected_files):
                if not self._is_running:
                    break
                self.progress.emit(i + 1, total_files)

                file_item = self.files.get(f.path)
                if not file_item:
                    continue

                export_image = load_image(file_item)
                if file_item.working_image is not None:
                    if len(file_item.working_image.shape) == 2:
                        export_image = np.array(export_image)[
                            :,
                            : file_item.metadata.max_size,
                            : file_item.metadata.max_size,
                        ]
                        bf_channel = int(file_item.metadata.reference_channel)
                        export_image[bf_channel] = file_item.working_image.squeeze()
                    elif len(file_item.working_image.shape) == 3:
                        export_image = np.array(file_item.working_image)

                meta_metadata = {
                    "axes": file_item.metadata.axes,
                    "unit": file_item.metadata.unit,
                    "PhysicalSizeX": file_item.metadata.PhysicalSizeX,
                    "PhysicalSizeY": file_item.metadata.PhysicalSizeY,
                }

                if len(export_image.shape) > 2:
                    export_image = export_image[
                        :,
                        : int(file_item.metadata.max_size),
                        : int(file_item.metadata.max_size),
                    ]
                elif len(export_image.shape) == 2:
                    export_image = export_image[
                        : int(file_item.metadata.max_size),
                        : int(file_item.metadata.max_size),
                    ]
                file_name = os.path.basename(file_item.path)

                for status in FileStatus:
                    if status.name.startswith("_"):
                        continue
                    prefix_to_check = status.value.lower() + "_"
                    if file_name.lower().startswith(prefix_to_check):
                        file_name = file_name[len(prefix_to_check) :]
                        break

                if file_item.metadata.prefix:
                    file_name = f"{file_item.metadata.prefix}_{file_name}"

                export_path = os.path.join(self.folder_path, file_name)
                tifffile.imwrite(export_path, export_image, metadata=meta_metadata)

            if self._is_running:
                self.progress.emit(total_files, total_files)
                self.export_complete.emit()
        except Exception as e:
            logger.error(f"Error in ExportThread: {e}", exc_info=True)
            self.export_error.emit(str(e))

    def cancel(self):
        self._is_running = False


class BeadUploadThread(QThread):
    progress = pyqtSignal(int, str)
    upload_complete = pyqtSignal(FileItem)

    def __init__(
        self,
        csv_path: str,
        reference_file: FileItem,
        cycle_assignments: dict,
        files: dict,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.reference_file = reference_file
        self.cycle_assignments = cycle_assignments
        self.files = files
        self._is_running = True

    def run(self):
        self._is_running = True
        try:
            print(
                f"BeadUploadThread.run started, reference_file path: {self.reference_file.path}"
            )
            self.progress.emit(0, "Loading beads data...")
            beads_df = pd.read_csv(self.csv_path)
            print(f"Loaded CSV with {len(beads_df)} rows")

            cycles = {}
            total_cycles = len(self.cycle_assignments)
            for cycle_idx, (idx, file_item) in enumerate(
                self.cycle_assignments.items()
            ):
                if not self._is_running:
                    break
                progress_pct = int((idx / total_cycles) * 100)
                self.progress.emit(progress_pct, f"Loading cycle {idx}...")

                cycle_name = f"{CYCLE_COLUMN_PREFIX}{idx}"
                max_size = int(self.reference_file.metadata.max_size)
                img = load_and_constrain_image(file_item, max_size)
                cycles[cycle_name] = img

            if not self._is_running:
                return

            self.reference_file.beads = beads_df
            self.reference_file.cycles = cycles
            self.reference_file.cycle_files = self.cycle_assignments
            self.reference_file.status = FileStatus.BEADS_GENERATED
            self.reference_file.metadata.prefix = (
                FileStatus.BEADS_GENERATED.name.lower()
            )

            print(
                f"BeadUploadThread done, beads attached: {self.reference_file.beads is not None}"
            )
            self.progress.emit(100, "Beads loaded")
            self.upload_complete.emit(self.reference_file)
        except Exception as e:
            logger.error(f"Error in BeadUploadThread: {e}", exc_info=True)
            self.progress.emit(-1, f"Error: {str(e)}")

    def cancel(self):
        self._is_running = False


class FileManagerVM(QObject):
    file_list_updated = pyqtSignal(list)
    file_information_update = pyqtSignal(list)
    file_metadata_updated = pyqtSignal(dict)
    metadata_corrected_sig = pyqtSignal(dict)
    align_progress = pyqtSignal(int, str)
    align_error = pyqtSignal(str)
    align_complete = pyqtSignal(list)
    export_progress = pyqtSignal(int, int)
    export_complete = pyqtSignal()
    export_error = pyqtSignal(str)
    beads_generated = pyqtSignal(DataFrame)
    bead_progress = pyqtSignal(int, str)
    shading_progress = pyqtSignal(int, str)
    shading_complete = pyqtSignal(list)
    folder_loaded = pyqtSignal(list)
    folder_loading_progress = pyqtSignal(int, str)
    file_loaded = pyqtSignal(FileItem)
    file_loading_progress = pyqtSignal(int, str)
    bead_upload_progress = pyqtSignal(int, str)
    bead_upload_complete = pyqtSignal(FileItem)
    inspect_beads_signal = pyqtSignal(
        dict, DataFrame, dict, Series, np.ndarray, DataFrame, int
    )

    def __init__(self):
        super().__init__()
        self.files: dict[str, FileItem] = {}
        self.reference_item: FileItem | None = None
        self.emitted_files = set()
        self.register_thread = None
        self.bead_thread = None
        self.shading_thread = None
        self.folder_thread = None
        self.file_thread = None
        self.export_thread = None
        self.upload_thread = None
        self.selected_files = []
        self._pending_files = {}

    def set_reference_item(self, file_item: FileItem):
        self.reference_item = file_item
        logger.debug(f"Reference item set to: {file_item.path}")

    def inspect_beads(self, file_item: FileItem, protein_profile: DataFrame):
        logger.debug(f"Inspecting beads for file: {file_item.path}")

        if file_item.path not in self.files:
            self.align_error.emit("File not found in the manager.")
            return
        most_updated_file = self.files[file_item.path]
        if most_updated_file.beads is None or most_updated_file.beads.empty:
            self.align_error.emit("No beads data available for this file.")
            return

        if len(protein_profile) > 0:
            logger.debug(protein_profile.head())
        # test data
        # df = pd.read_csv("./test_outputs/efficient_test.csv")
        # most_updated_file.beads = df

        # if shade corrected, this will be the shade corrected bf image
        # bf = self._get_brightfield_image(most_updated_file)
        # if bf is None:
        # return
        cycles = most_updated_file.cycles if most_updated_file.cycles else {}
        bboxs = (
            most_updated_file.bboxs
            if most_updated_file.bboxs is not None
            else pd.Series()
        )
        labeled_image = (
            most_updated_file.labeled_image
            if most_updated_file.labeled_image is not None
            else np.array([])
        )
        bright_fields = {}
        # bright_fields["cy0"] = bf
        # use first channel of cycles as brightfield for other cycles
        for cy_name, cy_image in cycles.items():
            bright_fields[cy_name] = cy_image[0]
        # set each cycle to exclude brightfield from decoding
        flour_cycles = {}
        flour_channel_start = most_updated_file.metadata.reference_channel + 1
        for cy_name in cycles.keys():
            flour_cycles[cy_name] = cycles[cy_name][flour_channel_start:]
        self.inspect_beads_signal.emit(
            bright_fields,
            most_updated_file.beads,
            flour_cycles,
            bboxs,
            labeled_image,
            protein_profile,
            most_updated_file.metadata.max_size,
        )

    def _get_brightfield_image(self, file_item: FileItem) -> Optional[np.ndarray]:
        """Retrieve the brightfield image from the file item, considering shading correction."""
        max_size = int(file_item.metadata.max_size)
        image = load_image(file_item)
        bf_channel = int(file_item.metadata.reference_channel)
        if file_item.working_image is not None:
            # If shade corrected or aligned image exists, use it
            if len(file_item.working_image.shape) == 2:
                return file_item.working_image[:max_size, :max_size]
            elif len(file_item.working_image.shape) > 2:
                if bf_channel < file_item.working_image.shape[0]:
                    return file_item.working_image[bf_channel, :max_size, :max_size]
                else:
                    self.align_error.emit(
                        f"Brightfield channel {bf_channel} exceeds number of channels in working image for file {os.path.basename(file_item.path)}."
                    )
                    return None
        # Fallback to original image
        if len(image.shape) > 2:
            if bf_channel < image.shape[0]:
                return image[bf_channel, :max_size, :max_size]
            else:
                self.align_error.emit(
                    f"Brightfield channel {bf_channel} exceeds number of channels in original image for file {os.path.basename(file_item.path)}."
                )
                return None
        elif len(image.shape) == 2:
            return image[:max_size, :max_size]
        return None

    def _get_status_from_filename(self, file_path: str) -> FileStatus:
        filename_base = os.path.basename(file_path).lower()
        for status in FileStatus:
            if status.name.startswith("_"):
                continue
            assert isinstance(status.value, str)
            prefix = status.value.lower() + "_"
            logger.debug(f"prefix: {prefix}, {status}")
            if filename_base.startswith(prefix):
                return status
        return FileStatus.RAW

    def load_folder(self, folder_path: List[str] | str):
        self._pending_files = {}
        self.folder_thread = FolderLoadingThread(folder_path)
        self.folder_thread.progress.connect(self.folder_loading_progress.emit)
        self.folder_thread.folder_loaded.connect(self._on_folder_loaded)
        self.folder_thread.start()

    def _on_folder_loaded(self, file_items: List[FileItem]):
        to_be_emitted = []
        for file_item in file_items:
            if file_item.path not in self.emitted_files:
                self.emitted_files.add(file_item.path)
                self._pending_files[file_item.path] = file_item
                to_be_emitted.append(file_item)
            else:
                self._pending_files[file_item.path] = file_item
        self.files.update(self._pending_files)
        self._pending_files = {}
        self.file_list_updated.emit(to_be_emitted)

    def load_file(self, file_paths: List[str] | str):
        self.file_thread = FileLoadingThread(file_paths)
        self.file_thread.progress.connect(self.file_loading_progress.emit)
        self.file_thread.file_loaded.connect(self._on_file_loaded)
        self.file_thread.start()

    def _on_file_loaded(self, file_items: List[FileItem]):
        to_be_emitted = []
        for file_item in file_items:
            if file_item.path in self.emitted_files:
                print(f"File {file_item.path} already loaded.")
                continue
            self.emitted_files.add(file_item.path)
            self.files[file_item.path] = file_item
            to_be_emitted.append(file_item)
        self.file_list_updated.emit(to_be_emitted)

    def apply_shading(self, selected_files: List[FileItem]):
        self.shading_thread = ShadingCorrectionThread(selected_files, self.files)
        self.shading_thread.progress.connect(self.shading_progress.emit)
        self.shading_thread.shading_complete.connect(self._on_shading_complete)
        self.shading_thread.start()

    def _on_shading_complete(self, to_be_updated: List[FileItem]):
        self.file_information_update.emit(to_be_updated)
        self.shading_complete.emit(to_be_updated)

    def cancel_shading(self):
        if self.shading_thread:
            self.shading_thread.cancel()

    def apply_crop(
        self, file_items: list[FileItem], x1: int, y1: int, x2: int, y2: int
    ):
        """
        Apply crop to selected files by updating working_image.

        Args:
            file_items: Files to crop
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
        """
        to_be_updated = []

        for f in file_items:
            my_f = self.files.get(f.path)
            if not my_f:
                continue

            # Get full image (use working_image if available, otherwise load original)
            if my_f.working_image is not None:
                full_image = my_f.working_image
            else:
                full_image = np.array(load_image(my_f))

            # Validate bounds
            if len(full_image.shape) == 3:
                num_channels, h, w = full_image.shape
            else:
                h, w = full_image.shape

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 <= x1 or y2 <= y1:
                print(
                    f"Invalid crop bounds for {my_f.path}: ({x1},{y1}) to ({x2},{y2})"
                )
                continue

            # Apply crop to all channels
            if len(full_image.shape) == 3:
                cropped = full_image[:, y1:y2, x1:x2]
            else:
                cropped = full_image[y1:y2, x1:x2]

            # Update FileItem
            my_f.working_image = cropped
            my_f.shape = cropped.shape

            # Update metadata crop_bounds for reference
            my_f.metadata.crop_bounds = (x1, y1, x2, y2)

            to_be_updated.append(my_f)

        if to_be_updated:
            self.file_information_update.emit(to_be_updated)

    # appy metadata changes to selected files
    def apply_metadata(self, metadata_changes: dict, selected_files: list[FileItem]):
        logger.debug(f"Applying metadata changes: {metadata_changes}")
        logger.debug(f"To selected files: {[f.path for f in selected_files]}")
        to_be_updated = []
        # min of width and height for all files selected
        max_viable_size = reduce(
            lambda x, y: min(x, y),
            [
                min(int(f.original_shape[-2]), int(f.original_shape[-1]))
                for f in selected_files
            ],
            int(metadata_changes.get("max_size", float("inf"))),
        )
        corrected_values = {}

        if "max_size" in metadata_changes:
            if max_viable_size != metadata_changes["max_size"]:
                corrected_values = {"max_size": max_viable_size}

        changes = [key for key, val in metadata_changes.items() if val != NOT_SENT]

        for f in selected_files:
            saved_f = self.files.get(f.path)
            if not saved_f:
                logger.warning(f"File {f.path} not found in manager.")
                continue

            use_status_as_prefix = metadata_changes.get("use_status_as_prefix", False)
            if use_status_as_prefix:
                saved_f.metadata.prefix = saved_f.status.value.lower()

            for key in changes:
                # max_size and use_status_as_prefix handled separately
                if not key.startswith("_") and key not in (
                    "max_size",
                    "use_status_as_prefix",
                ):
                    try:
                        value = metadata_changes.get(key, NOT_SENT)
                        if hasattr(saved_f.metadata, key):
                            setattr(saved_f.metadata, key, value)
                    except Exception:
                        pass
            max_size = metadata_changes.get("max_size", NOT_SENT)
            if max_size is not NOT_SENT:
                actual_max_size = (
                    max_viable_size if max_viable_size != max_size else max_size
                )
                saved_f.metadata.max_size = actual_max_size
                original_shape = saved_f.original_shape
                if len(original_shape) == 3:
                    saved_f.shape = (
                        original_shape[0],
                        max_viable_size,
                        max_viable_size,
                    )
                else:
                    saved_f.shape = (max_viable_size, max_viable_size)
            to_be_updated.append(saved_f)

        self.file_metadata_updated.emit(metadata_changes)
        if to_be_updated:
            self.file_information_update.emit(to_be_updated)
        if corrected_values:
            self.metadata_corrected_sig.emit(corrected_values)

    def align_channels(self, selected_files: list[FileItem]):
        if not self.reference_item:
            return
        from align_arrays import Register

        selected_files = [
            f for f in selected_files if f.path != self.reference_item.path
        ]

        self.register_thread = Register(
            self.reference_item,
            {
                "max_size": int(self.reference_item.metadata.max_size),
                "alignment_layer": int(self.reference_item.metadata.reference_channel),
                "num_tiles": self.reference_item.metadata.num_tiles,
                "overlap": self.reference_item.metadata.overlap,
                "file_path": self.reference_item.path,
                "threshold": self.reference_item.threshold,
            },
            selected_files,
            self.files,
        )
        self.register_thread.progress.connect(self.align_progress.emit)
        self.register_thread.error.connect(self.align_error.emit)
        self.register_thread.alignment_complete.connect(self.align_complete.emit)
        self.register_thread.alignment_complete.connect(
            lambda aligned: self._on_alignment_complete(aligned, selected_files)
        )
        self.register_thread.run_registration()

    def _on_alignment_complete(
        self,
        aligned_tifs: list[np.ndarray],
        selected: list[FileItem],
    ):
        if not self.reference_item:
            self.align_error.emit("Reference item not set for alignment completion.")
            return

        target_image = self._get_brightfield_image(self.reference_item)
        if target_image is None:
            self.align_error.emit("Could not load reference image for preview.")
            return

        moving_images = []
        for i, f_item in enumerate(selected):
            ref_channel = int(f_item.metadata.reference_channel)
            if ref_channel < aligned_tifs[i].shape[0]:
                moving_images.append(aligned_tifs[i][ref_channel])
            else:
                # Fallback to first channel if ref_channel is out of bounds
                moving_images.append(aligned_tifs[i][0])

        dialog = AlignmentPreviewDialog(target_image, moving_images, can_emit=True)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            to_be_updated = []
            for i, f in enumerate(selected):
                my_f = self.files.get(f.path)
                if not my_f:
                    continue
                my_f.working_image = aligned_tifs[i]
                my_f.status = FileStatus.ALIGNED
                my_f.metadata.prefix = FileStatus.ALIGNED.name.lower()
                to_be_updated.append(my_f)
            self.file_information_update.emit(to_be_updated)

    def delete_files(self, selected_files: list[FileItem]):
        for f in selected_files:
            print(f"Deleting file: {f.path}")
            if f.path in self.files:
                print("File found in manager, deleting.")
                del self.files[f.path]
            if f.path in self.emitted_files:
                print("File found in emitted files, removing.")
                self.emitted_files.remove(f.path)

    def cancel_alignment(self):
        if self.register_thread:
            self.register_thread.cancel()

    def cancel_bead_generation(self):
        if self.bead_thread:
            self.bead_thread.cancel()

    def set_status(self, file_item: FileItem, status: FileStatus):
        if file_item.path in self.files:
            saved_f = self.files[file_item.path]
            saved_f.status = status
            saved_f.metadata.prefix = status.value.lower()
            self.file_information_update.emit([saved_f])

    def set_reference(self, file_item: FileItem):
        to_be = []
        prev_ref = self.reference_item
        self.reference_item = file_item
        to_be.append(file_item)
        if prev_ref and prev_ref.path != file_item.path:
            to_be.append(prev_ref)
        self.file_information_update.emit(to_be)

    def clear_reference(self):
        if self.reference_item:
            old_ref = self.reference_item
            self.reference_item = None
            self.file_information_update.emit([old_ref])

    def export_files(self, folder_path: str, selected_files: list[FileItem]):
        self.export_thread = ExportThread(folder_path, self.files, selected_files)
        self.export_thread.progress.connect(self.export_progress.emit)
        self.export_thread.export_complete.connect(self._on_export_complete)
        self.export_thread.export_error.connect(self.export_error.emit)
        self.export_thread.start()

    def _on_export_complete(self):
        self.export_complete.emit()

    def cancel_export(self):
        if self.export_thread:
            self.export_thread.cancel()

    def generate_beads(
        self,
        cycle_assignments: dict[int, FileItem],
        use_stardist=False,
        model_name="model_4_400epoch_no_aug",
    ):
        assert self.reference_item is not None, (
            "Reference item must be set before generating beads."
        )

        sorted_cycles = sorted(cycle_assignments.keys())
        ordered_files = [cycle_assignments[cycle] for cycle in sorted_cycles]
        ordered_files.insert(0, self.reference_item)
        curr_ref_path = self.reference_item.path

        self.bead_thread = BeadGenerationThread(
            self.reference_item,
            ordered_files,
            self.files,
            signal_to_noise_cutoff=0.1,
            use_stardist=use_stardist,
            model_name=model_name,
        )
        self.bead_thread.bead_generated.connect(
            lambda res: self._on_beads_generated(res, curr_ref_path, cycle_assignments)
        )
        self.bead_thread.progress.connect(self.bead_progress.emit)
        self.bead_thread.start()

    def _on_beads_generated(
        self, results: dict, reference_path, cycle_assignments: dict[int, FileItem]
    ):
        cycles = results.get("cycles", {})
        bboxs = results.get("bboxs", pd.Series())
        labeled_image = results.get("labeled_image", None)
        beads = results.get("beads", pd.DataFrame())

        self.files[reference_path].cycles = cycles
        self.files[reference_path].bboxs = bboxs
        self.files[reference_path].labeled_image = labeled_image
        self.files[reference_path].beads = beads
        self.files[reference_path].status = FileStatus.BEADS_GENERATED
        self.files[
            reference_path
        ].metadata.prefix = FileStatus.BEADS_GENERATED.name.lower()
        self.files[reference_path].cycle_files = cycle_assignments
        self.beads_generated.emit(beads)
        self.bead_progress.emit(100, "Done generating beads")
        self.file_information_update.emit([self.files[reference_path]])

    def validate_bead_csv(self, csv_path: str) -> tuple[bool, str | None, int]:
        """Validate the structure of a bead CSV file.

        Returns:
            tuple: (is_valid, error_message, num_cycles)
        """
        if not os.path.exists(csv_path):
            return (False, "CSV file does not exist", 0)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return (False, f"Failed to read CSV: {str(e)}", 0)

        # Check for required columns
        missing_cols = [col for col in REQUIRED_BEAD_COLUMNS if col not in df.columns]
        if missing_cols:
            return (False, f"Missing required columns: {', '.join(missing_cols)}", 0)

        # Check for cycle columns (cy0, cy1, cy2, ...)
        cycle_cols = [
            col
            for col in df.columns
            if col.startswith(CYCLE_COLUMN_PREFIX)
            and col[len(CYCLE_COLUMN_PREFIX) :].isdigit()
        ]
        cycle_cols_sorted = sorted(
            cycle_cols, key=lambda x: int(x[len(CYCLE_COLUMN_PREFIX) :])
        )

        if not cycle_cols:
            return (
                False,
                f"No cycle columns found (expected {CYCLE_COLUMN_PREFIX}0, {CYCLE_COLUMN_PREFIX}1, etc.)",
                0,
            )

        # Validate sequential cycle numbers
        expected_cycles = [
            f"{CYCLE_COLUMN_PREFIX}{i}" for i in range(len(cycle_cols_sorted))
        ]
        if cycle_cols_sorted != expected_cycles:
            return (
                False,
                f"Cycle columns must be sequential ({CYCLE_COLUMN_PREFIX}0, {CYCLE_COLUMN_PREFIX}1, {CYCLE_COLUMN_PREFIX}2, ...)",
                0,
            )

        return (True, None, len(cycle_cols_sorted))

    def store_uploaded_beads(
        self,
        csv_path: str,
        reference_file: FileItem,
        cycle_assignments: dict[int, FileItem],
    ):
        print(
            f"store_uploaded_beads called, reference_file path: {reference_file.path}"
        )
        print(f"files dict keys before: {list(self.files.keys())[:5]}...")
        self.upload_thread = BeadUploadThread(
            csv_path, reference_file, cycle_assignments, self.files
        )
        self.upload_thread.progress.connect(self.bead_upload_progress.emit)
        self.upload_thread.upload_complete.connect(self._on_beads_uploaded)
        self.upload_thread.start()

    def _on_beads_uploaded(self, reference_file: FileItem):
        print(f"_on_beads_uploaded called, beads: {reference_file.beads is not None}")
        self.file_information_update.emit([reference_file])
        self.bead_upload_complete.emit(reference_file)


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


def load_image(item: FileItem):
    try:
        return tifffile.memmap(item.path, shape=item.shape, mode="r")
    except ValueError:
        return tifffile.imread(item.path)


def load_and_constrain_image(file_item: FileItem, max_size: int) -> np.ndarray:
    """Load image and apply max_size constraint.

    Args:
        file_item: FileItem to load image from
        max_size: Maximum size constraint for width/height

    Returns:
        Image array constrained to max_size
    """
    img = load_image(file_item)

    # Apply max_size constraint
    if len(img.shape) == 3:
        img = np.array(img)[:, :max_size, :max_size]
    else:
        img = np.array(img)[:max_size, :max_size]

    # Use working_image if available
    if file_item.working_image is not None:
        if len(file_item.working_image.shape) == 2:
            img[int(file_item.metadata.reference_channel)] = np.expand_dims(
                file_item.working_image, axis=0
            )
        elif len(file_item.working_image.shape) == 3:
            img = np.array(file_item.working_image)

    return img
