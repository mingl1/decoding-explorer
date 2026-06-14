import logging
import os
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tifffile import TiffFileError

import image_processing
import utils
from model.file_item import FileItem
from model.status_enum import FileStatus
from viewmodel.bead_eta_estimator import BeadEtaEstimator
from viewmodel.file_io import (
    get_tif_info,
    list_tiff_files,
    load_and_constrain_image,
    load_image,
    status_from_filename,
)

_CYCLE_COLUMN_PREFIX = "cy"

logger = logging.getLogger(__name__)


# Yields (pct, msg) progress; returns list[FileItem] with shape/dtype populated (no pixel data loaded).
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
            yield (
                int(i / total_files * 100) if total_files else 0,
                f"Loading files ({i + 1}/{total_files})",
            )

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


# Yields (pct, msg) progress; returns list[FileItem]. Falls back to PIL for non-TIFF formats.
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
                yield (
                    file_progress,
                    f"Error loading {os.path.basename(file_path)}: {str(e)}",
                )
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


# Yields (pct, msg) progress; returns list[FileItem] with working_image set to the corrected brightfield.
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

        max_size = int(f.metadata.max_size)
        bf_channel = int(f.metadata.reference_channel)

        if my_f.working_image is not None:
            working = np.array(my_f.working_image)
            if len(working.shape) == 3:
                bright_field = (
                    working[bf_channel] if bf_channel < working.shape[0] else working[0]
                )
                bright_field = bright_field[:max_size, :max_size]

                yield (
                    progress_pct,
                    f"Applying shading correction ({i + 1}/{total_files})",
                )
                corrected = utils.shading_correction(bright_field)

                # Overwrite only the reference channel of the 3D working_image
                my_f.working_image[bf_channel, :max_size, :max_size] = corrected
            else:
                logger.error(
                    f"Shading correction: working_image for {f.path} is already 2D (shape: {working.shape}). Shading correction may have already been applied."
                )
                bright_field = working[:max_size, :max_size]

                yield (
                    progress_pct,
                    f"Applying shading correction ({i + 1}/{total_files})",
                )
                corrected = utils.shading_correction(bright_field)
                my_f.working_image = corrected
        else:
            image = np.array(load_image(f))
            bright_field = (
                image[bf_channel]
                if (len(image.shape) == 3 and bf_channel < image.shape[0])
                else (image[0] if len(image.shape) == 3 else image)
            )
            bright_field = bright_field[:max_size, :max_size]

            yield progress_pct, f"Applying shading correction ({i + 1}/{total_files})"
            corrected = utils.shading_correction(bright_field)
            my_f.working_image = corrected

        my_f.status = FileStatus.SHADE_CORRECTED
        my_f.metadata.prefix = FileStatus.SHADE_CORRECTED.name.lower()
        to_be_updated.append(my_f)

    yield 100, "Shading correction complete"
    return to_be_updated


# Yields (pct, msg) progress; writes TIFFs to folder_path, stripping old status prefix and applying working_image if set.
def export_task(folder_path, files, selected_files, stop: threading.Event):
    total_files = len(selected_files)
    for i, f in enumerate(selected_files):
        if stop.is_set():
            return
        yield (
            int(i / total_files * 100) if total_files else 0,
            f"Exporting {i + 1}/{total_files}",
        )

        file_item = files.get(f.path)
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
                : int(file_item.metadata.max_size), : int(file_item.metadata.max_size)
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

        tifffile.imwrite(
            os.path.join(folder_path, file_name), export_image, metadata=meta_metadata
        )

    yield 100, "Export complete"
    return None


# Yields (pct, msg) progress; attaches beads DataFrame and cycle images to reference_file in-place, returns it.
def bead_upload_task(
    csv_path, reference_file, cycle_assignments, files, stop: threading.Event
):
    yield 0, "Loading beads data..."
    beads_df = pd.read_csv(csv_path)
    beads_df.columns = [str(c).lstrip("#").strip() for c in beads_df.columns]

    cycles = {}
    total_cycles = len(cycle_assignments)
    for cycle_idx, (idx, file_item) in enumerate(cycle_assignments.items()):
        if stop.is_set():
            return None
        progress_pct = int(idx / total_cycles * 100) if total_cycles else 0
        yield progress_pct, f"Loading cycle {idx}..."

        cycle_name = f"{_CYCLE_COLUMN_PREFIX}{idx}"
        max_size = int(reference_file.metadata.max_size)
        img = load_and_constrain_image(file_item, max_size)
        cycles[cycle_name] = img

    if stop.is_set():
        return None

    reference_file.beads = beads_df
    reference_file.pre_ensemble_beads = None
    reference_file.ensemble_cache = None
    reference_file.ensemble_sweep_stats = None
    reference_file.ensemble_ratio_selected = None
    reference_file.ensemble_ratio_applied = None
    reference_file.cycles = cycles
    reference_file.cycle_files = cycle_assignments
    reference_file.status = FileStatus.BEADS_GENERATED
    reference_file.metadata.prefix = FileStatus.BEADS_GENERATED.name.lower()

    yield 100, "Beads loaded"
    return reference_file


# Yields (pct, msg) progress; loads brightfield images in parallel via ThreadPoolExecutor, returns list in original order.
def brightfield_batch_loading_task(
    file_items, files, brightfield_loader, materialize, stop: threading.Event
):
    total_files = len(file_items)
    if total_files == 0:
        raise RuntimeError("No images selected for manual alignment preview.")

    loaded_images = [None] * total_files

    def _load_one(index, item):
        latest_file = files.get(item.path, item)
        image, error_msg = brightfield_loader(latest_file, materialize=materialize)
        return index, latest_file.path, image, error_msg

    max_workers = max(1, min(total_files, 8))
    yield 0, f"Loading manual alignment image 0/{total_files}"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(_load_one, i, fi): i for i, fi in enumerate(file_items)
        }
        completed_count = 0

        while pending:
            if stop.is_set():
                for future in pending:
                    future.cancel()
                return None

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
                    for pf in pending:
                        pf.cancel()
                    raise RuntimeError(f"Error loading manual alignment image: {e}")

                if error_msg:
                    for pf in pending:
                        pf.cancel()
                    raise RuntimeError(error_msg)

                if image is None:
                    for pf in pending:
                        pf.cancel()
                    raise RuntimeError(
                        f"Could not load image for {os.path.basename(path)}."
                    )

                loaded_images[idx] = image
                completed_count += 1
                yield (
                    int(completed_count / total_files * 100),
                    f"Loading manual alignment image {completed_count}/{total_files}",
                )

    if any(image is None for image in loaded_images):
        raise RuntimeError("Could not load all images for manual alignment preview.")

    yield 100, "Manual alignment preview ready"
    return loaded_images


# Yields (pct, msg) progress; runs process_beads in a background thread with a heartbeat thread to keep progress live.
# Returns the beads result dict, or None if stopped or failed.
def bead_generation_task(
    ref_file,
    file_items: list,
    files: dict,
    signal_to_noise_cutoff,
    stop: threading.Event,
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
    # Coerce params: UI settings persist these as strings or mixed types.
    stardist_use_guess_tiles = bool(stardist_use_guess_tiles)
    stardist_n_tiles = max(int(stardist_n_tiles), 1)
    use_stardist_bead_centers = bool(use_stardist_bead_centers)
    try:
        area_multiplier = float(area_multiplier)
        if area_multiplier <= 0:
            area_multiplier = 1.8
    except (TypeError, ValueError):
        area_multiplier = 1.8
    ensemble_ratio_start = float(ensemble_ratio_start)
    ensemble_ratio_end = float(ensemble_ratio_end)
    ensemble_ratio_step = float(ensemble_ratio_step)

    estimator = BeadEtaEstimator(mode="stardist" if use_stardist else "legacy")
    run_started_at = time.monotonic()
    last_emit = [
        -1,
        "",
    ]  # list (not tuple) so the nested closure can mutate it; -1 = never emitted

    def _format_and_put(
        progress_value, progress_message, eta_range=None, progress_queue=None
    ):
        pct = int(progress_value)
        msg = str(progress_message or "Processing beads...")
        if 0 <= pct < 100 and not use_stardist:
            elapsed_bucket = int(max(0.0, time.monotonic() - run_started_at))
            if elapsed_bucket == last_emit[0] and msg == last_emit[1]:
                return
            last_emit[0] = elapsed_bucket
            last_emit[1] = msg
        if 0 < pct < 100 and eta_range is not None:
            msg = f"{msg} ({BeadEtaEstimator.format_eta_range(eta_range)})"
        if progress_queue is not None:
            progress_queue.put((pct, msg))
        return pct, msg

    # Phase 1: load reference image
    ref_bf_channel = int(ref_file.metadata.reference_channel)
    ref_max_size = int(ref_file.metadata.max_size)
    ref_img = files[ref_file.path].working_image
    ref_bf = None
    if ref_img is not None:
        if len(ref_img.shape) == 2:
            ref_bf = ref_img
        elif len(ref_img.shape) == 3:
            ref_bf = np.array(ref_img)
    else:
        ref_img = load_image(files[ref_file.path])
        if len(ref_img.shape) == 3:
            ref_bf = np.array(ref_img)[ref_bf_channel, :ref_max_size, :ref_max_size]
        elif len(ref_img.shape) == 2:
            ref_bf = np.array(ref_img)[:ref_max_size, :ref_max_size]

    if ref_bf is None:
        logger.error(
            "Reference image does not have a valid brightfield channel for bead generation."
        )
        return None

    # Phase 2: load cycle images
    tifs = []
    total_files = len(file_items)
    for i, f in enumerate(file_items):
        if stop.is_set():
            return None
        pct, msg, eta, rate = estimator.update_stage_units(
            stage="load_images", done=i + 1, total=total_files, message="Loading images"
        )
        result = _format_and_put(pct, msg, eta)
        if result is not None:
            yield result

        my_f = files.get(f.path)
        if not my_f:
            continue
        img = load_and_constrain_image(my_f, int(f.metadata.max_size))
        tifs.append((img, f))

    if stop.is_set():
        return None

    # Phase 3: set workload for stardist ETA
    if use_stardist:
        total_channels = 0
        for img, file_item in tifs:
            metadata = getattr(file_item, "metadata", None)
            reference_channel = int(
                getattr(metadata, "reference_channel", 0) if metadata else 0
            )
            flors_layers = getattr(metadata, "flors_layers", None)
            if flors_layers is not None:
                channel_count = len(flors_layers)
            else:
                channel_dim = int(img.shape[0]) if getattr(img, "ndim", 0) == 3 else 1
                channel_count = sum(
                    1 for j in range(channel_dim) if j > reference_channel
                )
            total_channels += max(int(channel_count), 0)
        estimator.set_workload(
            total_channels=total_channels, max_size_pixels=ref_max_size
        )

    # Phase 4: run process_beads in a thread with heartbeat
    progress_queue = queue.SimpleQueue()

    def _cb(p, m):
        if stop.is_set():
            return
        if p < 0:
            progress_queue.put((-1, str(m)))
            return
        pct, msg, eta, rate = estimator.update_from_message(m)
        _format_and_put(pct, msg, eta, progress_queue)

    def _units_cb(stage, done, total):
        if stop.is_set():
            return
        stage_message = None
        if use_stardist:
            if stage == "init_det_tiles":
                stage_message = "Initial bead detection"
            elif stage == "activation_regions":
                stage_message = "Processing fluorescence channels"
        pct, msg, eta, rate = estimator.update_stage_units(
            stage=stage, done=done, total=total, message=stage_message
        )
        _format_and_put(pct, msg, eta, progress_queue)

    heartbeat_stop = threading.Event()

    def _heartbeat():
        while not heartbeat_stop.wait(0.5):
            if stop.is_set():
                return
            pct, msg, eta, rate = estimator.heartbeat()
            _format_and_put(pct, msg, eta, progress_queue)

    # Heartbeat keeps progress ticking when process_beads goes silent between callbacks.
    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()

    result_holder = [
        None
    ]  # single-element lists so _run() can write back to the outer scope
    error_holder = [None]
    done_event = threading.Event()

    def _run():
        try:
            result_holder[0] = image_processing.process_beads(
                ref_bf,
                tifs,
                max_size=ref_max_size,
                signal_to_noise_cutoff=signal_to_noise_cutoff,
                progress_callback=_cb,
                progress_units_callback=_units_cb,
                is_running_callback=lambda: not stop.is_set(),
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
        except Exception as e:
            error_holder[0] = e
        finally:
            done_event.set()

    process_thread = threading.Thread(target=_run, daemon=True)
    process_thread.start()

    while not done_event.wait(0.05):
        if stop.is_set():
            break
        while not progress_queue.empty():
            yield progress_queue.get_nowait()

    heartbeat_stop.set()
    hb_thread.join(timeout=1.0)

    while not progress_queue.empty():
        yield progress_queue.get_nowait()

    if error_holder[0] is not None:
        raise error_holder[0]

    results = result_holder[0]
    if stop.is_set() or results is None:
        return None

    estimator.finish(success=True)
    return results
