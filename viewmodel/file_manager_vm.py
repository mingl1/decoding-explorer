import logging
import os
from functools import reduce
from typing import Optional

import imageio.v3 as iio  # or PIL / cv2
import numpy as np
import pandas as pd
import tifffile
from lark import logger
from pandas import DataFrame, Series
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
        ref_bf,
        tifs,
        max_size,
        signal_to_noise_cutoff,
        use_stardist=False,
        model_name="model_4_400epoch_no_aug",
    ):
        super().__init__()
        self.ref_bf = ref_bf
        self.tifs = tifs
        self.max_size = max_size
        self.signal_to_noise_cutoff = signal_to_noise_cutoff
        self.use_stardist = use_stardist
        self.model_name = model_name
        self._is_running = True

    def run(self):
        self._is_running = True
        try:
            results = image_processing.process_beads(
                self.ref_bf,
                self.tifs,
                max_size=self.max_size,
                signal_to_noise_cutoff=self.signal_to_noise_cutoff,
                progress_callback=self.progress.emit,
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


class FileManagerVM(QObject):
    file_list_updated = pyqtSignal(list)
    file_information_update = pyqtSignal(list)
    file_metadata_updated = pyqtSignal(dict)
    metadata_corrected_sig = pyqtSignal(dict)
    align_progress = pyqtSignal(int, str)
    align_error = pyqtSignal(str)
    align_complete = pyqtSignal(list)
    export_progress = pyqtSignal(int, int)
    beads_generated = pyqtSignal(DataFrame)
    bead_progress = pyqtSignal(int, str)
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
        self.selected_files = []

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
        image = load_image(file_item)
        bf_channel = int(file_item.metadata.reference_channel)
        if file_item.working_image is not None:
            # If shade corrected or aligned image exists, use it
            if len(file_item.working_image.shape) == 2:
                return file_item.working_image
            elif len(file_item.working_image.shape) > 2:
                if bf_channel < file_item.working_image.shape[0]:
                    return file_item.working_image[bf_channel]
                else:
                    self.align_error.emit(
                        f"Brightfield channel {bf_channel} exceeds number of channels in working image for file {os.path.basename(file_item.path)}."
                    )
                    return None
        # Fallback to original image
        if len(image.shape) > 2:
            if bf_channel < image.shape[0]:
                return image[bf_channel]
            else:
                self.align_error.emit(
                    f"Brightfield channel {bf_channel} exceeds number of channels in original image for file {os.path.basename(file_item.path)}."
                )
                return None
        elif len(image.shape) == 2:
            return image
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

    def load_folder(self, folder_path):
        to_be_emitted = []
        for file in list_tiff_files(folder_path):
            status = self._get_status_from_filename(file)

            try:
                shape, dtype = get_tif_info(file)
            except (OSError, ValueError, TiffFileError):
                logger.warning(f"Skipping invalid TIFF file: {file}")
                continue

            self.files[file] = FileItem(path=file, status=status)
            self.files[file].shape = shape
            self.files[file].original_shape = shape
            self.files[file].dtype = str(dtype)
            self.files[file].metadata.max_size = (
                min(shape[-2], shape[-1]) if len(shape) >= 2 else 10000
            )
            if file not in self.emitted_files:
                self.emitted_files.add(file)
                to_be_emitted.append(self.files[file])
        self.file_list_updated.emit(to_be_emitted)

    def load_file(self, file_path):
        if os.path.isfile(file_path):
            status = self._get_status_from_filename(file_path)
            self.files[file_path] = FileItem(path=file_path, status=status)

            try:
                # Try TIFF-specific metadata extraction
                shape, dtype = get_tif_info(file_path)
            except (OSError, ValueError, TiffFileError):
                # Fallback: load as generic image array
                try:
                    arr = iio.imread(file_path)
                    shape = arr.shape
                    dtype = arr.dtype
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}", exc_info=True)
                    return  # Can't even load as generic image

            self.files[file_path].shape = shape
            self.files[file_path].original_shape = shape
            self.files[file_path].dtype = str(dtype)
            self.files[file_path].metadata.max_size = (
                min(shape[-2], shape[-1]) if len(shape) >= 2 else 10000
            )

            if file_path not in self.emitted_files:
                self.emitted_files.add(file_path)
                self.file_list_updated.emit([self.files[file_path]])

    def apply_shading(self, selected_files: list[FileItem]):
        to_be_updated = []
        for f in selected_files:
            image = np.array(load_image(f))
            bf_channel = int(f.metadata.reference_channel)
            bright_field = (
                image[bf_channel] if bf_channel < image.shape[0] else image[0]
            )
            # Apply max_size constraint before shading correction
            max_size = int(f.metadata.max_size)
            bright_field = bright_field[:max_size, :max_size]
            my_f = self.files.get(f.path)
            if not my_f:
                continue
            corrected = utils.shading_correction(bright_field)
            my_f.working_image = corrected
            my_f.status = FileStatus.SHADE_CORRECTED
            my_f.metadata.prefix = FileStatus.SHADE_CORRECTED.name.lower()
            to_be_updated.append(my_f)
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
        # import here to avoid startup overhead
        from align_arrays import Register

        selected_files = [
            f for f in selected_files if f.path != self.reference_item.path
        ]
        alignable_images = []
        for f in selected_files:
            image = np.array(load_image(f))
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
            np.array(load_image(self.reference_item))[
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
                "threshold": self.reference_item.threshold,
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
            if f.path in self.files:
                del self.files[f.path]
                if f.path in self.emitted_files:
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
        self.reference_item = file_item
        self.file_information_update.emit([file_item])

    def clear_reference(self):
        if self.reference_item:
            old_ref = self.reference_item
            self.reference_item = None
            self.file_information_update.emit([old_ref])

    def export_files(self, folder_path: str, selected_files: list[FileItem]):
        total_files = len(selected_files)
        self.export_progress.emit(0, total_files)

        for i, f in enumerate(selected_files):
            file_item = self.files.get(f.path)
            if not file_item:
                continue
            export_image = load_image(file_item)
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
            # ensure max size
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

            # Check for and remove existing status prefixes
            for status in FileStatus:
                if status.name.startswith("_"):
                    continue
                prefix_to_check = status.value.lower() + "_"
                if file_name.lower().startswith(prefix_to_check):
                    file_name = file_name[len(prefix_to_check) :]
                    break

            if file_item.metadata.prefix:
                file_name = f"{file_item.metadata.prefix}_{file_name}"

            export_path = os.path.join(folder_path, file_name)
            tifffile.imwrite(export_path, export_image, metadata=metadata)
            self.export_progress.emit(i + 1, total_files)

    def generate_beads(self, cycle_assignments: dict[int, FileItem], use_stardist=False, model_name='model_4_400epoch_no_aug'):
        assert self.reference_item is not None, (
            "Reference item must be set before generating beads."
        )
        tifs = []

        sorted_cycles = sorted(cycle_assignments.keys())
        ordered_files = [cycle_assignments[cycle] for cycle in sorted_cycles]
        ordered_files.insert(0, self.reference_item)  # reference is always first
        curr_ref_path = self.reference_item.path
        for f in ordered_files:
            my_f = self.files.get(f.path)
            if not my_f:
                continue
            max_size = int(f.metadata.max_size)
            img = load_and_constrain_image(my_f, max_size)
            tifs.append((img, f))
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
            ref_img = load_image(self.files[ref_bf_path])
            ref_max_size = int(self.reference_item.metadata.max_size)
            if len(ref_img.shape) == 3:
                ref_bf = np.array(ref_img)[ref_bf_channel, :ref_max_size, :ref_max_size]
            elif len(ref_img.shape) == 2:
                ref_bf = np.array(ref_img)[:ref_max_size, :ref_max_size]
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
        """Store uploaded bead CSV data and cycle images in reference FileItem.

        Args:
            csv_path: Path to the bead CSV file
            reference_file: FileItem to store beads and cycles
            cycle_assignments: Dict mapping cycle index to FileItem
        """
        # Load beads DataFrame
        beads_df = pd.read_csv(csv_path)
        reference_file.beads = beads_df

        # Load cycle images and store in cycles dict
        cycles = {}
        max_size = int(reference_file.metadata.max_size)
        for cycle_idx, file_item in cycle_assignments.items():
            cycle_name = f"{CYCLE_COLUMN_PREFIX}{cycle_idx}"
            img = load_and_constrain_image(file_item, max_size)
            cycles[cycle_name] = img

        reference_file.cycles = cycles
        reference_file.cycle_files = cycle_assignments
        reference_file.status = FileStatus.BEADS_GENERATED
        reference_file.metadata.prefix = FileStatus.BEADS_GENERATED.name.lower()

        # Emit signal to update UI
        self.file_information_update.emit([reference_file])


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
