import os
from email.mime import image
from math import e
from typing import Optional
from weakref import ref

import imageio.v3 as iio  # or PIL / cv2
import numpy as np
import pandas as pd
import tifffile
from pandas import DataFrame, Series
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog
from tifffile import TiffFile, TiffFileError

import image_processing
import utils
from model.file_item import FileItem
from model.status_enum import FileStatus
from view.alignment_preview_dialog import AlignmentPreviewDialog


class BeadGenerationThread(QThread):
    progress = pyqtSignal(int, str)
    bead_generated = pyqtSignal(dict)

    def __init__(self, ref_bf, tifs, max_size, signal_to_noise_cutoff):
        super().__init__()
        self.ref_bf = ref_bf
        self.tifs = tifs
        self.max_size = max_size
        self.signal_to_noise_cutoff = signal_to_noise_cutoff
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
            )
            if self._is_running:
                self.bead_generated.emit(results)
        except Exception as e:
            print(e)
        return None

    def cancel(self):
        self._is_running = False

    def is_running(self):
        return self._is_running


class FileManagerVM(QObject):
    file_list_updated = pyqtSignal(list)
    file_information_update = pyqtSignal(list)
    file_metadata_updated = pyqtSignal(dict)
    align_progress = pyqtSignal(int, str)
    align_error = pyqtSignal(str)
    align_complete = pyqtSignal(list)
    export_progress = pyqtSignal(int, int)
    beads_generated = pyqtSignal(DataFrame)
    bead_progress = pyqtSignal(int, str)
    inspect_beads_signal = pyqtSignal(
        dict, DataFrame, dict, Series, np.ndarray, DataFrame
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
        print(f"Reference item set to: {file_item.path}")

    def inspect_beads(self, file_item: FileItem, protein_profile: DataFrame):
        print(f"Inspecting beads for file: {file_item.path}")

        if file_item.path not in self.files:
            self.align_error.emit("File not found in the manager.")
            return
        most_updated_file = self.files[file_item.path]
        if most_updated_file.beads is None or most_updated_file.beads.empty:
            self.align_error.emit("No beads data available for this file.")
            return

        if len(protein_profile) > 0:
            print(protein_profile.head())
        # test data
        # df = pd.read_csv("./test_outputs/efficient_test.csv")
        # print(f"Loaded {len(df)} beads from CSV for testing.")
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
        )

    def _get_brightfield_image(self, file_item: FileItem) -> Optional[np.ndarray]:
        """Retrieve the brightfield image from the file item, considering shading correction."""
        image = load_image(file_item.path, int(file_item.metadata.max_size))
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
            print(f"prefix: {prefix}, {status}")
            if filename_base.startswith(prefix):
                return status
        return FileStatus.RAW

    def load_folder(self, folder_path):
        to_be_emitted = []
        for file in list_tiff_files(folder_path):
            status = self._get_status_from_filename(file)
            self.files[file] = FileItem(path=file, status=status)
            shape, dtype = get_tif_info(file)
            self.files[file].shape = shape
            self.files[file].dtype = str(dtype)
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
                    print(f"Failed to load {file_path}: {e}")
                    return  # Can't even load as generic image

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
            print(corrected.dtype)
            my_f.working_image = corrected.copy()
            my_f.status = FileStatus.CORRECTED
            my_f.metadata.prefix = FileStatus.CORRECTED.name.lower()
            to_be_updated.append(my_f)
        self.file_information_update.emit(to_be_updated)

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
        # import here to avoid startup overhead
        from align_arrays import Register

        selected_files = [
            f for f in selected_files if f.path != self.reference_item.path
        ]
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
            print(f"ALIGN ARRYS SHAPE: {image.shape}")

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
            self.files[file_item.path].status = status
            self.file_information_update.emit([self.files[file_item.path]])

    def set_reference(self, file_item: FileItem):
        to_be_updated = []
        if self.reference_item:
            if self.reference_item.path in self.files:
                working_image = self.files[self.reference_item.path].working_image
                if working_image is None:
                    self.files[self.reference_item.path].status = FileStatus.RAW
                elif len(working_image.shape) == 2:
                    self.files[self.reference_item.path].status = FileStatus.CORRECTED
                elif len(working_image.shape) > 2:
                    self.files[self.reference_item.path].status = FileStatus.ALIGNED
                to_be_updated.append(self.files[self.reference_item.path])
        self.reference_item = file_item
        self.files[file_item.path].status = FileStatus.REFERENCE
        to_be_updated.append(self.files[file_item.path])
        self.file_information_update.emit(to_be_updated)

    def export_files(self, folder_path: str, selected_files: list[FileItem]):
        total_files = len(selected_files)
        self.export_progress.emit(0, total_files)

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
            print(export_image.shape, export_image.dtype)
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
            tifffile.imwrite(export_path, export_image, metadata=metadata, imagej=True)
            self.export_progress.emit(i + 1, total_files)

    def generate_beads(self, cycle_assignments: dict[int, FileItem]):
        assert (
            self.reference_item is not None
        ), "Reference item must be set before generating beads."
        tifs = []

        sorted_cycles = sorted(cycle_assignments.keys())
        ordered_files = [cycle_assignments[cycle] for cycle in sorted_cycles]
        ordered_files.insert(0, self.reference_item)  # reference is always first
        curr_ref_path = self.reference_item.path
        for f in ordered_files:
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
        post_process_beads = results.get("post_process_beads", pd.DataFrame())

        self.files[reference_path].cycles = cycles
        self.files[reference_path].bboxs = bboxs
        self.files[reference_path].labeled_image = labeled_image
        self.files[reference_path].beads = beads
        self.files[reference_path].post_process_beads = post_process_beads
        self.files[reference_path].status = FileStatus.BEADS_GENERATED
        self.files[reference_path].metadata.prefix = (
            FileStatus.BEADS_GENERATED.name.lower()
        )
        self.files[reference_path].cycle_files = cycle_assignments
        self.beads_generated.emit(beads)
        self.bead_progress.emit(100, "Done generating beads")
        self.file_information_update.emit([self.files[reference_path]])


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
        return tifffile.imread(file_path)
