import logging
import os
import threading
from functools import reduce
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QDialog

import image_processing
from model.file_item import FileItem
from model.status_enum import FileStatus
from view.alignment_preview_dialog import AlignmentPreviewDialog
from viewmodel.file_io import load_image
from viewmodel.tasks import (
    bead_generation_task,
    bead_upload_task,
    brightfield_batch_loading_task,
    export_task,
    file_loading_task,
    folder_loading_task,
    shading_correction_task,
)
from viewmodel.worker_thread import WorkerThread

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
NOT_SENT = "..."

# Bead CSV validation constants
REQUIRED_BEAD_COLUMNS = ["x", "y"]
CYCLE_COLUMN_PREFIX = "cy"


class FileManagerVM(QObject):
    file_list_updated = pyqtSignal(list)
    file_information_update = pyqtSignal(list)
    file_metadata_updated = pyqtSignal(dict)
    metadata_corrected_sig = pyqtSignal(dict)
    align_progress = pyqtSignal(int, str)
    manual_align_preview_progress = pyqtSignal(int, str)
    manual_align_preview_loaded = pyqtSignal(object)
    manual_align_preview_error = pyqtSignal(str)
    align_error = pyqtSignal(str)
    align_complete = pyqtSignal(list)
    export_progress = pyqtSignal(int, str)
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
    dataset_assignment_changed = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.files: dict[str, FileItem] = {}
        self.reference_item: FileItem | None = None
        self.dataset_cycle_assignments: dict[int, FileItem] | None = None
        self.dataset_protein_file: FileItem | None = None
        self.dataset_assignment_valid = False
        self.dataset_assignment_reason = "Assign cycles to continue."
        self.emitted_files = set()
        self.register_thread = None
        self.bead_thread = None
        self.shading_thread = None
        self.folder_thread = None
        self.file_thread = None
        self.export_thread = None
        self.upload_thread = None
        self.manual_align_preview_thread = None
        self.selected_files = []
        self._pending_files = {}
        self.dataset_assignment_changed.emit(
            self.dataset_assignment_valid, self.dataset_assignment_reason
        )

    def _emit_dataset_assignment_changed(self):
        self.dataset_assignment_changed.emit(
            self.dataset_assignment_valid, self.dataset_assignment_reason
        )

    def clear_dataset_cycle_assignments(self, reason: Optional[str] = None):
        self.dataset_cycle_assignments = None
        self.dataset_protein_file = None
        self.dataset_assignment_valid = False
        self.dataset_assignment_reason = reason or "Assign cycles to continue."
        self._emit_dataset_assignment_changed()

    def set_dataset_cycle_assignments(
        self,
        assignments: dict[int, FileItem],
        protein_file: Optional[FileItem] = None,
    ) -> bool:
        if self.reference_item is None:
            self.clear_dataset_cycle_assignments("Please set a reference image first.")
            return False
        if self.reference_item.path not in self.files:
            self.clear_dataset_cycle_assignments(
                "Reference file is not available in the current session."
            )
            return False

        normalized: dict[int, FileItem] = {0: self.reference_item}
        for cycle_num, file_item in assignments.items():
            try:
                cycle_idx = int(cycle_num)
            except (TypeError, ValueError):
                self.clear_dataset_cycle_assignments("Invalid cycle assignment index.")
                return False

            if cycle_idx == 0:
                continue
            if cycle_idx < 0:
                self.clear_dataset_cycle_assignments(
                    "Cycle indices must be non-negative."
                )
                return False
            saved_f = self.files.get(file_item.path)
            if saved_f is None:
                self.clear_dataset_cycle_assignments(
                    f"Assigned file not found: {os.path.basename(file_item.path)}"
                )
                return False
            normalized[cycle_idx] = saved_f

        unique_paths = {f.path for f in normalized.values()}
        if len(unique_paths) != len(normalized):
            self.clear_dataset_cycle_assignments(
                "Each file must be assigned to exactly one cycle."
            )
            return False

        saved_protein_file = None
        if protein_file is not None:
            saved_protein_file = self.files.get(protein_file.path)
            if saved_protein_file is None:
                self.clear_dataset_cycle_assignments(
                    f"Protein file not found: {os.path.basename(protein_file.path)}"
                )
                return False
            if saved_protein_file.path == self.reference_item.path:
                self.clear_dataset_cycle_assignments(
                    "Protein file cannot be the reference."
                )
                return False
            if saved_protein_file.path in unique_paths:
                self.clear_dataset_cycle_assignments(
                    "Protein file cannot also be assigned to a cycle."
                )
                return False

        self.dataset_cycle_assignments = {
            cycle_num: file_item
            for cycle_num, file_item in sorted(normalized.items(), key=lambda x: x[0])
        }
        self.dataset_protein_file = saved_protein_file
        self.dataset_assignment_valid = True
        protein_text = ""
        if self.dataset_protein_file is not None:
            protein_text = (
                f" + protein ({os.path.basename(self.dataset_protein_file.path)})"
            )
        self.dataset_assignment_reason = (
            f"Assigned {len(self.dataset_cycle_assignments)} cycle(s){protein_text}."
        )
        self._emit_dataset_assignment_changed()
        return True

    def get_dataset_cycle_assignments(self) -> Optional[dict[int, FileItem]]:
        if not self.is_dataset_ready():
            return None
        assert self.dataset_cycle_assignments is not None
        return dict(self.dataset_cycle_assignments)

    def get_dataset_files_ordered(self) -> list[FileItem]:
        assignments = self.get_dataset_cycle_assignments()
        if assignments is None:
            return []
        return [assignments[cycle_num] for cycle_num in sorted(assignments.keys())]

    def get_dataset_protein_file(self) -> Optional[FileItem]:
        if not self.is_dataset_ready():
            return None
        return self.dataset_protein_file

    def is_dataset_ready(self) -> bool:
        if (
            not self.dataset_assignment_valid
            or self.reference_item is None
            or self.dataset_cycle_assignments is None
        ):
            return False
        if 0 not in self.dataset_cycle_assignments:
            return False
        if self.dataset_cycle_assignments[0].path != self.reference_item.path:
            return False
        for file_item in self.dataset_cycle_assignments.values():
            if file_item.path not in self.files:
                return False
        if self.dataset_protein_file is not None:
            if self.dataset_protein_file.path not in self.files:
                return False
            if self.dataset_protein_file.path == self.reference_item.path:
                return False
        return True

    def set_reference_item(self, file_item: FileItem):
        self.reference_item = file_item
        self.clear_dataset_cycle_assignments("Reference changed. Reassign cycles.")
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
            if cy_image.ndim == 3:
                bright_fields[cy_name] = cy_image[0]
            else:
                bright_fields[cy_name] = cy_image
        # set each cycle to exclude brightfield from decoding
        flour_cycles = {}
        flour_channel_start = most_updated_file.metadata.reference_channel + 1
        for cy_name, cy_image in cycles.items():
            if cy_image.ndim == 3:
                flour_cycles[cy_name] = cy_image[flour_channel_start:]
            else:
                flour_cycles[cy_name] = cy_image
        self.inspect_beads_signal.emit(
            bright_fields,
            most_updated_file.beads,
            flour_cycles,
            bboxs,
            labeled_image,
            protein_profile,
            most_updated_file.metadata.max_size,
        )

    def _extract_brightfield_image(
        self,
        file_item: FileItem,
        materialize=False,
        use_original=False,
    ) -> tuple[Optional[np.ndarray], Optional[str]]:
        latest_file = self.files.get(file_item.path, file_item)
        # None slice == full dimension; used when caller wants the untruncated image
        sz = None if use_original else int(latest_file.metadata.max_size)
        bf_channel = int(latest_file.metadata.reference_channel)

        if latest_file.working_image is not None and not use_original:
            working = latest_file.working_image
            if len(working.shape) == 2:
                brightfield = working[:sz, :sz]
            elif len(working.shape) > 2:
                if bf_channel < working.shape[0]:
                    brightfield = working[bf_channel, :sz, :sz]
                else:
                    return (
                        None,
                        f"Brightfield channel {bf_channel} exceeds number of channels in working image for file {os.path.basename(latest_file.path)}.",
                    )
            else:
                return (
                    None,
                    f"Unsupported working image dimensions for file {os.path.basename(latest_file.path)}.",
                )
            if materialize:
                return np.array(brightfield), None
            return brightfield, None

        image = load_image(latest_file)
        if len(image.shape) > 2:
            if bf_channel < image.shape[0]:
                brightfield = image[bf_channel, :sz, :sz]
            else:
                return (
                    None,
                    f"Brightfield channel {bf_channel} exceeds number of channels in original image for file {os.path.basename(latest_file.path)}.",
                )
        elif len(image.shape) == 2:
            brightfield = image[:sz, :sz]
        else:
            return (
                None,
                f"Unsupported image dimensions for file {os.path.basename(latest_file.path)}.",
            )

        if materialize:
            return np.array(brightfield), None
        return brightfield, None

    def _get_brightfield_image(
        self, file_item: FileItem, use_original=False
    ) -> Optional[np.ndarray]:
        image, error_msg = self._extract_brightfield_image(
            file_item,
            materialize=False,
            use_original=use_original,
        )
        if error_msg:
            self.align_error.emit(error_msg)
            return None
        return image

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
        stop = threading.Event()
        self.folder_thread = WorkerThread(folder_loading_task(folder_path, stop), stop)
        self.folder_thread.progress.connect(self.folder_loading_progress.emit)
        self.folder_thread.completed.connect(self._on_folder_loaded)
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
        if len(to_be_emitted) > 0:
            self.clear_dataset_cycle_assignments("File list changed. Reassign cycles.")
        self.file_list_updated.emit(to_be_emitted)

    def load_file(self, file_paths: List[str] | str):
        stop = threading.Event()
        self.file_thread = WorkerThread(file_loading_task(file_paths, stop), stop)
        self.file_thread.progress.connect(self.file_loading_progress.emit)
        self.file_thread.completed.connect(self._on_file_loaded)
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
        if len(to_be_emitted) > 0:
            self.clear_dataset_cycle_assignments("File list changed. Reassign cycles.")
        self.file_list_updated.emit(to_be_emitted)

    def apply_shading(self, selected_files: List[FileItem]):
        stop = threading.Event()
        self.shading_thread = WorkerThread(
            shading_correction_task(selected_files, self.files, stop), stop
        )
        self.shading_thread.progress.connect(self.shading_progress.emit)
        self.shading_thread.completed.connect(self._on_shading_complete)
        self.shading_thread.start()

    def _on_shading_complete(self, to_be_updated: List[FileItem]):
        self.file_information_update.emit(to_be_updated)
        self.shading_complete.emit(to_be_updated)

    def cancel_shading(self):
        if self.shading_thread:
            self.shading_thread.cancel()

    def load_manual_align_preview_images(
        self, reference_item: FileItem, moving_items: list[FileItem]
    ):
        ordered_files = [reference_item]
        ordered_files.extend(moving_items)
        if len(ordered_files) == 0:
            self.manual_align_preview_error.emit(
                "No images selected for manual alignment preview."
            )
            return

        self.cancel_manual_align_preview_loading()
        stop = threading.Event()
        self.manual_align_preview_thread = WorkerThread(
            brightfield_batch_loading_task(
                ordered_files,
                self.files,
                self._extract_brightfield_image,
                materialize=True,
                stop=stop,
            ),
            stop,
        )
        self.manual_align_preview_thread.progress.connect(
            self.manual_align_preview_progress.emit
        )
        self.manual_align_preview_thread.completed.connect(
            self.manual_align_preview_loaded.emit
        )
        self.manual_align_preview_thread.failed.connect(
            self.manual_align_preview_error.emit
        )
        self.manual_align_preview_thread.finished.connect(
            self._on_manual_align_preview_loading_finished
        )
        self.manual_align_preview_thread.start()

    def _on_manual_align_preview_loading_finished(self):
        finished_thread = self.sender()
        if finished_thread is self.manual_align_preview_thread:
            self.manual_align_preview_thread = None

    def cancel_manual_align_preview_loading(self):
        if self.manual_align_preview_thread:
            self.manual_align_preview_thread.cancel()

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
            my_f.status = FileStatus.CROPPED
            my_f.metadata.prefix = FileStatus.CROPPED.name.lower()

            # Cap max_size to the new (smaller) cropped dimensions
            cropped_min_side = min(cropped.shape[-2], cropped.shape[-1])
            if int(my_f.metadata.max_size) > cropped_min_side:
                my_f.metadata.max_size = cropped_min_side

            # Update metadata crop_bounds for reference
            my_f.metadata.crop_bounds = (x1, y1, x2, y2)

            to_be_updated.append(my_f)

        if to_be_updated:
            self.file_information_update.emit(to_be_updated)

    def apply_crop_with_transform(
        self, file_item: FileItem, transform: np.ndarray, crop_w: int, crop_h: int
    ):
        my_f = self.files.get(file_item.path)
        if not my_f:
            return

        full_image = np.array(load_image(my_f))

        T = np.asarray(transform, dtype=np.float64).reshape(2, 3)
        A = T[:, :2]
        t = T[:, 2]
        inv_a = np.linalg.inv(A)
        M = np.hstack([inv_a, (-inv_a @ t).reshape(2, 1)]).astype(np.float32)

        crop_w = int(round(crop_w))
        crop_h = int(round(crop_h))

        if len(full_image.shape) == 2:
            warped = cv2.warpAffine(
                full_image, M, (crop_w, crop_h), flags=cv2.INTER_LINEAR
            )
        else:
            num_channels = full_image.shape[0]
            warped = np.zeros((num_channels, crop_h, crop_w), dtype=full_image.dtype)
            for ch in range(num_channels):
                warped[ch] = cv2.warpAffine(
                    full_image[ch], M, (crop_w, crop_h), flags=cv2.INTER_LINEAR
                )

        my_f.working_image = warped
        my_f.shape = warped.shape
        my_f.status = FileStatus.CROPPED
        my_f.metadata.prefix = FileStatus.CROPPED.name.lower()

        cropped_min_side = min(warped.shape[-2], warped.shape[-1])
        if int(my_f.metadata.max_size) > cropped_min_side:
            my_f.metadata.max_size = cropped_min_side

        my_f.metadata.crop_bounds = (0, 0, crop_w, crop_h)

        self.file_information_update.emit([my_f])

    # appy metadata changes to selected files
    def apply_metadata(self, metadata_changes: dict, selected_files: list[FileItem]):
        logger.debug(f"Applying metadata changes: {metadata_changes}")
        logger.debug(f"To selected files: {[f.path for f in selected_files]}")
        to_be_updated = []
        # min of width and height for all files selected
        max_viable_size = reduce(
            lambda x, y: min(x, y),
            [
                min(
                    int(
                        (
                            f.working_image.shape
                            if f.working_image is not None
                            else f.original_shape
                        )[-2]
                    ),
                    int(
                        (
                            f.working_image.shape
                            if f.working_image is not None
                            else f.original_shape
                        )[-1]
                    ),
                )
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

        layer_labels = self._build_alignment_preview_layer_labels(selected)
        dialog = AlignmentPreviewDialog(
            target_image,
            moving_images,
            can_emit=True,
            layer_labels=layer_labels,
        )

        if dialog.exec() == QDialog.DialogCode.Accepted:
            to_be_updated = []
            for i, f in enumerate(selected):
                my_f = self.files.get(f.path)
                if not my_f:
                    continue
                my_f.working_image = aligned_tifs[i]
                my_f.status = FileStatus.AUTO_ALIGNED
                my_f.metadata.prefix = FileStatus.AUTO_ALIGNED.name.lower()
                to_be_updated.append(my_f)
            self.file_information_update.emit(to_be_updated)

    def _build_alignment_preview_layer_labels(
        self, selected: list[FileItem]
    ) -> list[str]:
        cycle_path_to_label = {}
        if self.dataset_cycle_assignments is not None:
            for cycle_num, file_item in self.dataset_cycle_assignments.items():
                if cycle_num == 0:
                    continue
                cycle_path_to_label[file_item.path] = f"Cycle {cycle_num + 1}"

        protein_path = None
        if self.dataset_protein_file is not None:
            protein_path = self.dataset_protein_file.path

        labels = []
        for index, file_item in enumerate(selected):
            if protein_path is not None and file_item.path == protein_path:
                labels.append("Protein")
                continue
            label = cycle_path_to_label.get(file_item.path)
            if label is not None:
                labels.append(label)
                continue
            labels.append(f"Moving Image {index + 1}")
        return labels

    def delete_files(self, selected_files: list[FileItem]):
        deleted_any = False
        for f in selected_files:
            print(f"Deleting file: {f.path}")
            if f.path in self.files:
                print("File found in manager, deleting.")
                del self.files[f.path]
                deleted_any = True
            if f.path in self.emitted_files:
                print("File found in emitted files, removing.")
                self.emitted_files.remove(f.path)
        if deleted_any:
            if self.reference_item and self.reference_item.path not in self.files:
                self.reference_item = None
            self.clear_dataset_cycle_assignments("File list changed. Reassign cycles.")

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
        self.clear_dataset_cycle_assignments("Reference changed. Reassign cycles.")
        to_be.append(file_item)
        if prev_ref and prev_ref.path != file_item.path:
            to_be.append(prev_ref)
        self.file_information_update.emit(to_be)

    def clear_reference(self):
        if self.reference_item:
            old_ref = self.reference_item
            self.reference_item = None
            self.clear_dataset_cycle_assignments("Please set a reference image first.")
            self.file_information_update.emit([old_ref])

    def export_files(self, folder_path: str, selected_files: list[FileItem]):
        stop = threading.Event()
        self.export_thread = WorkerThread(
            export_task(folder_path, self.files, selected_files, stop), stop
        )
        self.export_thread.progress.connect(self.export_progress.emit)
        self.export_thread.completed.connect(lambda _: self._on_export_complete())
        self.export_thread.failed.connect(self.export_error.emit)
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
        model_name="model_5_400epoch",
        stardist_use_guess_tiles=True,
        stardist_n_tiles=1,
        use_stardist_bead_centers=False,
        area_multiplier=1.8,
        ensemble_ratio_start=image_processing.DEFAULT_ENSEMBLE_RATIO_START,
        ensemble_ratio_end=image_processing.DEFAULT_ENSEMBLE_RATIO_END,
        ensemble_ratio_step=image_processing.DEFAULT_ENSEMBLE_RATIO_STEP,
    ):
        assert self.reference_item is not None, (
            "Reference item must be set before generating beads."
        )

        normalized_assignments = dict(cycle_assignments)
        normalized_assignments[0] = self.reference_item
        sorted_cycles = sorted(
            [cycle_num for cycle_num in normalized_assignments.keys() if cycle_num != 0]
        )
        ordered_files = [self.reference_item]
        ordered_files.extend(
            [normalized_assignments[cycle_num] for cycle_num in sorted_cycles]
        )
        curr_ref_path = self.reference_item.path

        stop = threading.Event()
        self.bead_thread = WorkerThread(
            bead_generation_task(
                self.reference_item,
                ordered_files,
                self.files,
                signal_to_noise_cutoff=0.1,
                stop=stop,
                use_stardist=use_stardist,
                model_name=model_name,
                stardist_use_guess_tiles=stardist_use_guess_tiles,
                stardist_n_tiles=stardist_n_tiles,
                use_stardist_bead_centers=use_stardist_bead_centers,
                area_multiplier=area_multiplier,
                ensemble_ratio_start=ensemble_ratio_start,
                ensemble_ratio_end=ensemble_ratio_end,
                ensemble_ratio_step=ensemble_ratio_step,
            ),
            stop,
        )
        self.bead_thread.completed.connect(
            lambda res: self._on_beads_generated(
                res, curr_ref_path, normalized_assignments
            )
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
        pre_ensemble_beads = results.get("pre_ensemble_beads", None)
        ensemble_cache = results.get("ensemble_cache", None)
        ensemble_sweep_stats = results.get("ensemble_sweep_stats", None)
        ensemble_ratio_applied = results.get("ensemble_ratio_applied", None)

        self.files[reference_path].cycles = cycles
        self.files[reference_path].bboxs = bboxs
        self.files[reference_path].labeled_image = labeled_image
        self.files[reference_path].beads = beads
        self.files[reference_path].pre_ensemble_beads = pre_ensemble_beads
        self.files[reference_path].ensemble_cache = ensemble_cache
        self.files[reference_path].ensemble_sweep_stats = ensemble_sweep_stats
        self.files[reference_path].ensemble_ratio_applied = ensemble_ratio_applied
        self.files[reference_path].ensemble_ratio_selected = ensemble_ratio_applied
        self.files[reference_path].status = FileStatus.BEADS_GENERATED
        self.files[
            reference_path
        ].metadata.prefix = FileStatus.BEADS_GENERATED.name.lower()
        self.files[reference_path].cycle_files = cycle_assignments
        self.beads_generated.emit(beads)
        self.bead_progress.emit(100, "Done generating beads")
        self.file_information_update.emit([self.files[reference_path]])

    def recompute_ensemble_sweep(
        self, file_item: FileItem, start: float, end: float, step: float
    ) -> DataFrame:
        if file_item.path not in self.files:
            raise ValueError("File is not managed by the current session.")
        saved_f = self.files[file_item.path]
        if saved_f.pre_ensemble_beads is None or saved_f.ensemble_cache is None:
            raise ValueError("No StarDist ensemble cache found for this file.")
        sweep_df = image_processing.compute_ensemble_sweep_stats(
            pre_ensemble_beads=saved_f.pre_ensemble_beads,
            ensemble_cache=saved_f.ensemble_cache,
            start=float(start),
            end=float(end),
            step=float(step),
        )
        saved_f.ensemble_sweep_stats = sweep_df
        if sweep_df.empty:
            saved_f.ensemble_ratio_selected = None
        else:
            if saved_f.ensemble_ratio_applied is not None:
                ratio_arr = sweep_df["ratio"].to_numpy(dtype=np.float64)
                target = float(saved_f.ensemble_ratio_applied)
                idx = int(np.argmin(np.abs(ratio_arr - target)))
                saved_f.ensemble_ratio_selected = float(ratio_arr[idx])
            else:
                saved_f.ensemble_ratio_selected = float(sweep_df.iloc[0]["ratio"])
        self.file_information_update.emit([saved_f])
        return sweep_df

    def apply_ensemble_ratio(self, file_item: FileItem, ratio: float) -> DataFrame:
        if file_item.path not in self.files:
            raise ValueError("File is not managed by the current session.")
        saved_f = self.files[file_item.path]
        if saved_f.pre_ensemble_beads is None or saved_f.ensemble_cache is None:
            raise ValueError("No StarDist ensemble cache found for this file.")
        ensembled_df = image_processing.build_ensembled_beads_from_cache(
            pre_ensemble_beads=saved_f.pre_ensemble_beads,
            ensemble_cache=saved_f.ensemble_cache,
            ratio=float(ratio),
        )
        saved_f.beads = ensembled_df
        saved_f.ensemble_ratio_applied = float(ratio)
        saved_f.ensemble_ratio_selected = float(ratio)
        self.file_information_update.emit([saved_f])
        return ensembled_df

    def remove_ensemble_applied_changes(self, file_item: FileItem) -> DataFrame:
        if file_item.path not in self.files:
            raise ValueError("File is not managed by the current session.")
        saved_f = self.files[file_item.path]
        if saved_f.pre_ensemble_beads is None:
            raise ValueError("No pre-ensemble StarDist beads found for this file.")
        saved_f.beads = saved_f.pre_ensemble_beads.copy()
        saved_f.ensemble_ratio_applied = None
        if (
            saved_f.ensemble_sweep_stats is not None
            and not saved_f.ensemble_sweep_stats.empty
        ):
            saved_f.ensemble_ratio_selected = float(
                saved_f.ensemble_sweep_stats.iloc[0]["ratio"]
            )
        else:
            saved_f.ensemble_ratio_selected = None
        self.file_information_update.emit([saved_f])
        assert saved_f.beads is not None, (
            "Beads should not be None after removing ensemble applied changes."
        )
        return saved_f.beads

    def validate_bead_csv(self, csv_path: str) -> tuple[bool, str | None, int]:
        """Validate the structure of a bead CSV file.

        Returns:
            tuple: (is_valid, error_message, num_cycles)
        """
        if not os.path.exists(csv_path):
            return (False, "CSV file does not exist", 0)

        try:
            df = pd.read_csv(csv_path)
            df.columns = [str(c).lstrip("#").strip() for c in df.columns]
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
        stop = threading.Event()
        self.upload_thread = WorkerThread(
            bead_upload_task(
                csv_path, reference_file, cycle_assignments, self.files, stop
            ),
            stop,
        )
        self.upload_thread.progress.connect(self.bead_upload_progress.emit)
        self.upload_thread.completed.connect(self._on_beads_uploaded)
        self.upload_thread.start()

    def _on_beads_uploaded(self, reference_file: FileItem):
        print(f"_on_beads_uploaded called, beads: {reference_file.beads is not None}")
        self.file_information_update.emit([reference_file])
        self.bead_upload_complete.emit(reference_file)
