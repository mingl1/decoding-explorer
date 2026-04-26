import logging
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
from viewmodel.bead_progress_estimator import BeadProgressEstimator
from viewmodel.stardist_runtime_eta import StarDistRuntimeEtaTracker

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
        model_name="model_5_400epoch",
        stardist_use_guess_tiles=True,
        stardist_n_tiles=1,
        use_stardist_bead_centers=False,
        area_multiplier=1.8,
        ensemble_ratio_start=image_processing.DEFAULT_ENSEMBLE_RATIO_START,
        ensemble_ratio_end=image_processing.DEFAULT_ENSEMBLE_RATIO_END,
        ensemble_ratio_step=image_processing.DEFAULT_ENSEMBLE_RATIO_STEP,
    ):
        super().__init__()
        self.ref_file = ref_file
        self.file_items = file_items
        self.files = files
        self.signal_to_noise_cutoff = signal_to_noise_cutoff
        self.use_stardist = use_stardist
        self.model_name = model_name
        self.stardist_use_guess_tiles = bool(stardist_use_guess_tiles)
        self.stardist_n_tiles = max(int(stardist_n_tiles), 1)
        self.use_stardist_bead_centers = bool(use_stardist_bead_centers)
        try:
            parsed_area_multiplier = float(area_multiplier)
            if parsed_area_multiplier <= 0:
                raise ValueError("area_multiplier must be > 0")
            self.area_multiplier = parsed_area_multiplier
        except (TypeError, ValueError):
            self.area_multiplier = 1.8
        self.ensemble_ratio_start = float(ensemble_ratio_start)
        self.ensemble_ratio_end = float(ensemble_ratio_end)
        self.ensemble_ratio_step = float(ensemble_ratio_step)
        self._is_running = True
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._estimator = None
        self._run_started_at: Optional[float] = None
        self._last_progress_emit_second: int = -1
        self._last_progress_emit_message = ""

    def run(self):
        self._is_running = True
        self._heartbeat_stop.clear()
        self._run_started_at = time.monotonic()
        self._last_progress_emit_second = -1
        self._last_progress_emit_message = ""
        if self.use_stardist:
            self._estimator = StarDistRuntimeEtaTracker()
        else:
            self._estimator = BeadProgressEstimator(mode="legacy")
        success = False
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
                (
                    progress_value,
                    progress_message,
                    total_eta_seconds,
                    step_eta_seconds,
                ) = self._estimator.update_stage_units_with_eta_details(
                    stage="load_images",
                    done=i + 1,
                    total=total_files,
                    message=f"Loading images ({i + 1}/{total_files})",
                )
                self._emit_estimated_progress(
                    progress_value,
                    progress_message,
                    total_eta_seconds,
                    step_eta_seconds,
                )

                my_f = self.files.get(f.path)
                if not my_f:
                    continue
                max_size = int(f.metadata.max_size)
                img = load_and_constrain_image(my_f, max_size)
                tifs.append((img, f))

            if not self._is_running:
                return

            self._start_heartbeat()
            stardist_bootstrap_stages: set[str] = set()

            def estimated_progress(p, m):
                if not self._is_running:
                    return
                if p < 0:
                    self.progress.emit(-1, m)
                    return
                (
                    progress_value,
                    progress_message,
                    total_eta_seconds,
                    step_eta_seconds,
                ) = self._estimator.update_from_message_with_eta_details(m)
                self._emit_estimated_progress(
                    progress_value,
                    progress_message,
                    total_eta_seconds,
                    step_eta_seconds,
                )
                if self.use_stardist and self._estimator:
                    stage = self._estimator.map_message_to_stage(m)
                    if (
                        stage == "initial_detection"
                        and stage not in stardist_bootstrap_stages
                    ):
                        stardist_bootstrap_stages.add(stage)
                        (
                            hb_progress_value,
                            hb_progress_message,
                            hb_total_eta_seconds,
                            hb_step_eta_seconds,
                        ) = self._estimator.heartbeat_with_eta_details()
                        self._emit_estimated_progress(
                            hb_progress_value,
                            hb_progress_message,
                            hb_total_eta_seconds,
                            hb_step_eta_seconds,
                        )

            def estimated_progress_units(stage, done, total):
                if not self._is_running:
                    return
                (
                    progress_value,
                    progress_message,
                    total_eta_seconds,
                    step_eta_seconds,
                ) = self._estimator.update_stage_units_with_eta_details(
                    stage=stage,
                    done=done,
                    total=total,
                )
                self._emit_estimated_progress(
                    progress_value,
                    progress_message,
                    total_eta_seconds,
                    step_eta_seconds,
                )

            results = image_processing.process_beads(
                ref_bf,
                tifs,
                max_size=ref_max_size,
                signal_to_noise_cutoff=self.signal_to_noise_cutoff,
                progress_callback=estimated_progress,
                progress_units_callback=estimated_progress_units,
                is_running_callback=self.is_running,
                use_stardist=self.use_stardist,
                model_name=self.model_name,
                stardist_use_guess_tiles=self.stardist_use_guess_tiles,
                stardist_n_tiles=self.stardist_n_tiles,
                use_stardist_bead_centers=self.use_stardist_bead_centers,
                area_multiplier=self.area_multiplier,
                ensemble_ratio_start=self.ensemble_ratio_start,
                ensemble_ratio_end=self.ensemble_ratio_end,
                ensemble_ratio_step=self.ensemble_ratio_step,
            )
            if self._is_running and results is not None:
                self.bead_generated.emit(results)
                success = True
        except Exception as e:
            logger.error(f"Error in BeadGenerationThread: {e}", exc_info=True)
        finally:
            self._stop_heartbeat()
            if self._estimator:
                self._estimator.finish(success=success)
        return None

    def _start_heartbeat(self):
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        self._heartbeat_thread = None

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(0.5):
            if not self._is_running or not self._estimator:
                return
            (
                progress_value,
                progress_message,
                total_eta_seconds,
                step_eta_seconds,
            ) = self._estimator.heartbeat_with_eta_details()
            self._emit_estimated_progress(
                progress_value,
                progress_message,
                total_eta_seconds,
                step_eta_seconds,
            )

    def _emit_estimated_progress(
        self,
        progress_value: int,
        progress_message: str,
        total_eta_seconds: Optional[float],
        step_eta_seconds: Optional[float],
    ):
        elapsed_seconds = 0.0
        if self._run_started_at is not None:
            elapsed_seconds = max(0.0, time.monotonic() - self._run_started_at)
        elapsed_second_bucket = int(elapsed_seconds)
        progress_int = int(progress_value)
        message = str(progress_message or "Processing beads...")
        if 0 <= progress_int < 100 and not self.use_stardist:
            if (
                elapsed_second_bucket == self._last_progress_emit_second
                and message == self._last_progress_emit_message
            ):
                return
            self._last_progress_emit_second = elapsed_second_bucket
            self._last_progress_emit_message = message
        if 0 <= progress_int < 100:
            status_parts = [
                f"Elapsed {BeadProgressEstimator.format_eta(elapsed_seconds)}"
            ]
            if step_eta_seconds is not None and step_eta_seconds > 0:
                status_parts.append(
                    f"Step ETA {BeadProgressEstimator.format_eta(step_eta_seconds)}"
                )
            if total_eta_seconds is not None and total_eta_seconds > 0:
                status_parts.append(
                    f"Total ETA {BeadProgressEstimator.format_eta(total_eta_seconds)}"
                )
            message = f"{message} ({', '.join(status_parts)})"
        self.progress.emit(progress_int, message)

    def cancel(self):
        self._is_running = False
        self._heartbeat_stop.set()

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


class BrightfieldBatchLoadingThread(QThread):
    progress = pyqtSignal(int, str)
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        file_items: list[FileItem],
        files: dict,
        brightfield_loader,
        materialize=True,
    ):
        super().__init__()
        self.file_items = file_items
        self.files = files
        self.brightfield_loader = brightfield_loader
        self.materialize = bool(materialize)
        self._is_running = True

    def run(self):
        self._is_running = True
        total_files = len(self.file_items)
        if total_files == 0:
            self.error.emit("No images selected for manual alignment preview.")
            return

        loaded_images: list[Optional[np.ndarray]] = [None] * total_files

        def _load_one(index: int, item: FileItem):
            latest_file = self.files.get(item.path, item)
            image, error_msg = self.brightfield_loader(
                latest_file, materialize=self.materialize
            )
            return index, latest_file.path, image, error_msg

        max_workers = max(1, min(total_files, 8))
        self.progress.emit(0, f"Loading manual alignment image 0/{total_files}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = {
                executor.submit(_load_one, i, file_item): i
                for i, file_item in enumerate(self.file_items)
            }
            completed_count = 0

            while pending:
                if not self._is_running:
                    for future in pending:
                        future.cancel()
                    return

                done, not_done = wait(
                    pending.keys(), timeout=0.1, return_when=FIRST_COMPLETED
                )
                if not done:
                    continue
                pending = {future: pending[future] for future in not_done}

                for future in done:
                    try:
                        idx, path, image, error_msg = future.result()
                    except Exception as e:
                        self.error.emit(f"Error loading manual alignment image: {e}")
                        for pending_future in pending:
                            pending_future.cancel()
                        return
                    if error_msg:
                        self.error.emit(error_msg)
                        for pending_future in pending:
                            pending_future.cancel()
                        return
                    if image is None:
                        self.error.emit(
                            f"Could not load image for {os.path.basename(path)}."
                        )
                        for pending_future in pending:
                            pending_future.cancel()
                        return
                    loaded_images[idx] = image
                    completed_count += 1
                    progress_pct = int((completed_count / total_files) * 100)
                    self.progress.emit(
                        progress_pct,
                        f"Loading manual alignment image {completed_count}/{total_files}",
                    )

        if not self._is_running:
            return
        if any(image is None for image in loaded_images):
            self.error.emit("Could not load all images for manual alignment preview.")
            return
        self.progress.emit(100, "Manual alignment preview ready")
        self.loaded.emit(loaded_images)

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
            self.reference_file.pre_ensemble_beads = None
            self.reference_file.ensemble_cache = None
            self.reference_file.ensemble_sweep_stats = None
            self.reference_file.ensemble_ratio_selected = None
            self.reference_file.ensemble_ratio_applied = None
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
    manual_align_preview_progress = pyqtSignal(int, str)
    manual_align_preview_loaded = pyqtSignal(object)
    manual_align_preview_error = pyqtSignal(str)
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
        if len(to_be_emitted) > 0:
            self.clear_dataset_cycle_assignments("File list changed. Reassign cycles.")
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
        if len(to_be_emitted) > 0:
            self.clear_dataset_cycle_assignments("File list changed. Reassign cycles.")
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
        self.manual_align_preview_thread = BrightfieldBatchLoadingThread(
            ordered_files,
            self.files,
            self._extract_brightfield_image,
            materialize=True,
        )
        self.manual_align_preview_thread.progress.connect(
            self.manual_align_preview_progress.emit
        )
        self.manual_align_preview_thread.loaded.connect(
            self.manual_align_preview_loaded.emit
        )
        self.manual_align_preview_thread.error.connect(
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

            # Cap max_size to the new (smaller) cropped dimensions
            cropped_min_side = min(cropped.shape[-2], cropped.shape[-1])
            if int(my_f.metadata.max_size) > cropped_min_side:
                my_f.metadata.max_size = cropped_min_side

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
            [min(int(f.shape[-2]), int(f.shape[-1])) for f in selected_files],
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
                my_f.status = FileStatus.ALIGNED
                my_f.metadata.prefix = FileStatus.ALIGNED.name.lower()
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

        self.bead_thread = BeadGenerationThread(
            self.reference_item,
            ordered_files,
            self.files,
            signal_to_noise_cutoff=0.1,
            use_stardist=use_stardist,
            model_name=model_name,
            stardist_use_guess_tiles=stardist_use_guess_tiles,
            stardist_n_tiles=stardist_n_tiles,
            use_stardist_bead_centers=use_stardist_bead_centers,
            area_multiplier=area_multiplier,
            ensemble_ratio_start=ensemble_ratio_start,
            ensemble_ratio_end=ensemble_ratio_end,
            ensemble_ratio_step=ensemble_ratio_step,
        )
        self.bead_thread.bead_generated.connect(
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
