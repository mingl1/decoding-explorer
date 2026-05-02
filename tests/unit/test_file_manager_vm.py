import os
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest
import tifffile
from model.file_item import FileItem
from model.status_enum import FileStatus
from PyQt6.QtWidgets import QDialog
from viewmodel.bead_eta_estimator import BeadEtaEstimator
from viewmodel.tasks import bead_generation_task, brightfield_batch_loading_task, export_task


class TestFileManagerVM:
    """Unit tests for the FileManagerVM ViewModel."""

    def test_initial_state(self, mock_file_manager_vm):
        """VM should start with empty files dict and no reference."""
        vm = mock_file_manager_vm
        assert vm.files == {}
        assert vm.reference_item is None
        assert vm.dataset_cycle_assignments is None
        assert vm.dataset_protein_file is None
        assert vm.dataset_assignment_valid is False
        assert vm.emitted_files == set()
        assert vm.register_thread is None
        assert vm.bead_thread is None
        assert vm.selected_files == []

    def test_set_dataset_cycle_assignments_requires_reference(
        self, mock_file_manager_vm, mock_file_item
    ):
        vm = mock_file_manager_vm
        vm.files[mock_file_item.path] = mock_file_item

        is_valid = vm.set_dataset_cycle_assignments({1: mock_file_item})

        assert is_valid is False
        assert vm.dataset_assignment_valid is False
        assert vm.dataset_cycle_assignments is None

    def test_set_dataset_cycle_assignments_success_and_order(
        self, mock_file_manager_vm, mock_file_items
    ):
        vm = mock_file_manager_vm
        ref_item, cy1_item, cy2_item = mock_file_items[:3]
        vm.files[ref_item.path] = ref_item
        vm.files[cy1_item.path] = cy1_item
        vm.files[cy2_item.path] = cy2_item
        vm.reference_item = ref_item

        is_valid = vm.set_dataset_cycle_assignments(
            {0: ref_item, 1: cy1_item, 2: cy2_item}
        )

        assert is_valid is True
        assert vm.dataset_assignment_valid is True
        ordered_files = vm.get_dataset_files_ordered()
        assert [f.path for f in ordered_files] == [
            ref_item.path,
            cy1_item.path,
            cy2_item.path,
        ]

    def test_set_reference_invalidates_dataset_assignments(
        self, mock_file_manager_vm, mock_file_items
    ):
        vm = mock_file_manager_vm
        ref_item, cy1_item, new_ref_item = mock_file_items[:3]
        vm.files[ref_item.path] = ref_item
        vm.files[cy1_item.path] = cy1_item
        vm.files[new_ref_item.path] = new_ref_item
        vm.reference_item = ref_item
        vm.set_dataset_cycle_assignments({0: ref_item, 1: cy1_item, 2: new_ref_item})
        assert vm.dataset_assignment_valid is True

        vm.set_reference(new_ref_item)

        assert vm.dataset_assignment_valid is False
        assert vm.dataset_cycle_assignments is None

    def test_set_dataset_cycle_assignments_with_optional_protein_file(
        self, mock_file_manager_vm, mock_file_items
    ):
        vm = mock_file_manager_vm
        ref_item, cy1_item, protein_item = mock_file_items[:3]
        vm.files[ref_item.path] = ref_item
        vm.files[cy1_item.path] = cy1_item
        vm.files[protein_item.path] = protein_item
        vm.reference_item = ref_item

        is_valid = vm.set_dataset_cycle_assignments(
            {0: ref_item, 1: cy1_item},
            protein_file=protein_item,
        )

        assert is_valid is True
        assert vm.dataset_assignment_valid is True
        assert vm.dataset_protein_file == protein_item
        assigned = vm.get_dataset_cycle_assignments()
        assert assigned is not None
        assert [f.path for f in vm.get_dataset_files_ordered()] == [
            ref_item.path,
            cy1_item.path,
        ]

    def test_set_dataset_cycle_assignments_allows_unassigned_files(
        self, mock_file_manager_vm, mock_file_items
    ):
        vm = mock_file_manager_vm
        ref_item, cy1_item, cy2_item = mock_file_items[:3]
        vm.files[ref_item.path] = ref_item
        vm.files[cy1_item.path] = cy1_item
        vm.files[cy2_item.path] = cy2_item
        vm.reference_item = ref_item

        is_valid = vm.set_dataset_cycle_assignments(
            {0: ref_item, 1: cy1_item}
        )

        assert is_valid is True
        assigned = vm.get_dataset_cycle_assignments()
        assert assigned is not None
        assert assigned[0] == ref_item
        assert assigned[1] == cy1_item

    def test_load_single_file(self, mock_file_manager_vm, tmp_tiff_path, mocker):
        """Loading a file should add it to files dict."""
        vm = mock_file_manager_vm
        path = tmp_tiff_path()

        with patch('viewmodel.file_manager_vm.get_tif_info') as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)

            vm.load_file(path)

            assert path in vm.files
            assert isinstance(vm.files[path], FileItem)
            assert vm.files[path].status == FileStatus.RAW

    def test_load_file_with_prefixed_name(self, mock_file_manager_vm, mocker):
        """File with aligned_ prefix should get ALIGNED status."""
        vm = mock_file_manager_vm

        with patch('os.path.isfile', return_value=True), \
             patch('viewmodel.file_manager_vm.get_tif_info') as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)

            vm.load_file("/test/aligned_image.tif")
            assert vm.files["/test/aligned_image.tif"].status == FileStatus.ALIGNED

    def test_load_folder(self, mock_file_manager_vm, tmp_tiff_folder, mocker):
        """Loading a folder should add all TIFFs to files dict."""
        vm = mock_file_manager_vm
        folder_path = tmp_tiff_folder(3)

        with patch('viewmodel.file_manager_vm.get_tif_info') as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)

            vm.load_folder(folder_path)

            assert len(vm.files) == 3

    def test_load_folder_non_existent(self, mock_file_manager_vm, mocker):
        """Loading non-existent folder should result in empty list."""
        vm = mock_file_manager_vm

        with patch('viewmodel.file_manager_vm.list_tiff_files', return_value=[]):
            vm.load_folder("/non/existent/folder")

            assert len(vm.files) == 0

    def test_load_folder_with_invalid_tiff_file(self, mock_file_manager_vm, tmp_path, mocker):
        """Loading folder with invalid TIFF file should skip it gracefully."""
        vm = mock_file_manager_vm

        folder = tmp_path / "test_folder"
        folder.mkdir()

        valid_tiff = folder / "valid.tif"
        arr = np.random.randint(0, 65535, (1, 512, 512), dtype=np.uint16)
        tifffile.imwrite(str(valid_tiff), arr)

        invalid_tif = folder / "invalid.tif"
        invalid_tif.write_text("This is not a valid TIFF file")

        from viewmodel import file_manager_vm
        original_get_tif_info = file_manager_vm.get_tif_info
        call_count = [0]

        def mock_tif_info(path):
            call_count[0] += 1
            if 'invalid.tif' in path:
                from tifffile import TiffFileError
                raise TiffFileError(f'not a TIFF file')
            return ((1, 512, 512), np.uint16)

        file_manager_vm.get_tif_info = mock_tif_info
        try:
            vm.load_folder(str(folder))

            assert str(valid_tiff) in vm.files
            assert str(invalid_tif) not in vm.files
            assert call_count[0] == 2
        finally:
            file_manager_vm.get_tif_info = original_get_tif_info

    def test_set_reference(self, mock_file_manager_vm, mock_file_item, mocker):
        """Setting reference should update reference_item but NOT change status."""
        vm = mock_file_manager_vm
        item = mock_file_item
        vm.files[item.path] = item
        original_status = item.status

        vm.set_reference(item)

        assert vm.reference_item == item
        assert vm.files[item.path].status == original_status

    def test_set_reference_only_updates_reference_item(
        self, mock_file_manager_vm, mock_file_items, mocker
    ):
        """Changing reference should only update reference_item, not status."""
        vm = mock_file_manager_vm

        item1, item2 = mock_file_items[0], mock_file_items[1]
        item1.status = FileStatus.RAW
        item2.status = FileStatus.SHADE_CORRECTED
        vm.files[item1.path] = item1
        vm.files[item2.path] = item2

        vm.set_reference(item1)
        vm.set_reference(item2)

        assert vm.reference_item == item2
        assert vm.files[item1.path].status == FileStatus.RAW
        assert vm.files[item2.path].status == FileStatus.SHADE_CORRECTED

    def test_set_status(self, mock_file_manager_vm, mock_file_item, mocker):
        """Setting status should update file status."""
        vm = mock_file_manager_vm
        item = mock_file_item
        vm.files[item.path] = item

        vm.set_status(item, FileStatus.ALIGNED)

        assert vm.files[item.path].status == FileStatus.ALIGNED

    def test_delete_files(self, mock_file_manager_vm, mock_file_items, mocker):
        """Deleting files should remove them from files dict."""
        vm = mock_file_manager_vm

        for item in mock_file_items:
            vm.files[item.path] = item
            vm.emitted_files.add(item.path)

        files_to_delete = [mock_file_items[0], mock_file_items[1]]
        vm.delete_files(files_to_delete)

        assert len(vm.files) == 1

    def test_delete_files_removes_from_emitted(self, mock_file_manager_vm, mock_file_item):
        """Deleting file should also remove from emitted_files set."""
        vm = mock_file_manager_vm
        item = mock_file_item
        vm.files[item.path] = item
        vm.emitted_files.add(item.path)

        vm.delete_files([item])

        assert item.path not in vm.emitted_files

    def test_apply_shading(self, mock_file_manager_vm, mock_file_item, mocker):
        """Applying shading should set working_image and update status."""
        vm = mock_file_manager_vm
        item = mock_file_item
        vm.files[item.path] = item

        mock_image = np.zeros((1, 512, 512), dtype=np.uint16)
        with patch('viewmodel.file_manager_vm.load_image', return_value=mock_image), \
             patch('utils.shading_correction') as mock_shade:
            mock_shade.return_value = np.ones((512, 512), dtype=np.uint16)
            vm.apply_shading([item])

        assert vm.files[item.path].status == FileStatus.SHADE_CORRECTED
        assert vm.files[item.path].working_image is not None

    def test_apply_shading_file_not_found(self, mock_file_manager_vm, mocker):
        """apply_shading with missing file should skip gracefully."""
        vm = mock_file_manager_vm

        with patch('viewmodel.file_manager_vm.load_image') as mock_load:
            mock_load.side_effect = KeyError("File not found")
            # Should not raise, just continue
            try:
                vm.apply_shading([])
            except KeyError:
                pass  # Expected if file not found

    def test_brightfield_batch_loading_task_returns_ordered_materialized_images(
        self, mock_file_manager_vm, tmp_path
    ):
        vm = mock_file_manager_vm

        ref_array = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8)
        moving_array = np.full((2, 8, 8), 3, dtype=np.uint16)
        ref_path = tmp_path / "ref.tif"
        moving_path = tmp_path / "moving.tif"
        tifffile.imwrite(str(ref_path), ref_array)
        tifffile.imwrite(str(moving_path), moving_array)

        ref_item = FileItem(path=str(ref_path), status=FileStatus.RAW)
        ref_item.shape = ref_array.shape
        ref_item.original_shape = ref_array.shape
        ref_item.metadata.reference_channel = 1
        ref_item.metadata.max_size = 6

        moving_item = FileItem(path=str(moving_path), status=FileStatus.RAW)
        moving_item.shape = moving_array.shape
        moving_item.original_shape = moving_array.shape
        moving_item.metadata.reference_channel = 0
        moving_item.metadata.max_size = 6
        moving_item.working_image = np.full((8, 8), 77, dtype=np.uint16)

        vm.files[ref_item.path] = ref_item
        vm.files[moving_item.path] = moving_item

        stop = threading.Event()
        gen = brightfield_batch_loading_task(
            [ref_item, moving_item], vm.files, vm._extract_brightfield_image,
            True, stop,
        )
        try:
            while True:
                next(gen)
        except StopIteration as e:
            loaded_images = e.value

        assert loaded_images is not None
        assert len(loaded_images) == 2
        np.testing.assert_array_equal(loaded_images[0], ref_array[1, :6, :6])
        np.testing.assert_array_equal(
            loaded_images[1], moving_item.working_image[:6, :6]
        )
        assert isinstance(loaded_images[0], np.ndarray)
        assert isinstance(loaded_images[1], np.ndarray)
        assert not isinstance(loaded_images[0], np.memmap)
        assert not isinstance(loaded_images[1], np.memmap)

    def test_brightfield_batch_loading_task_raises_for_invalid_channel(
        self, mock_file_manager_vm, tmp_path
    ):
        vm = mock_file_manager_vm

        img = np.zeros((1, 8, 8), dtype=np.uint16)
        bad_path = tmp_path / "bad_channel.tif"
        tifffile.imwrite(str(bad_path), img)

        bad_item = FileItem(path=str(bad_path), status=FileStatus.RAW)
        bad_item.shape = img.shape
        bad_item.original_shape = img.shape
        bad_item.metadata.reference_channel = 2
        bad_item.metadata.max_size = 8
        vm.files[bad_item.path] = bad_item

        stop = threading.Event()
        gen = brightfield_batch_loading_task(
            [bad_item], vm.files, vm._extract_brightfield_image, True, stop,
        )
        with pytest.raises(RuntimeError) as exc_info:
            for _ in gen:
                pass
        assert "exceeds number of channels" in str(exc_info.value)

    def test_brightfield_batch_loading_task_loads_images_concurrently(self):
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_loader(file_item, materialize=False):
            nonlocal active
            nonlocal max_active
            with lock:
                active += 1
                if active > max_active:
                    max_active = active
            time.sleep(0.05)
            with lock:
                active -= 1
            return np.zeros((4, 4), dtype=np.uint16), None

        file_items = [
            FileItem(path=f"/tmp/manual_align_{i}.tif", status=FileStatus.RAW)
            for i in range(4)
        ]
        stop = threading.Event()
        gen = brightfield_batch_loading_task(file_items, {}, fake_loader, True, stop)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            loaded_images = e.value

        assert loaded_images is not None
        assert len(loaded_images) == 4
        assert max_active > 1

    def test_align_channels_no_reference(self, mock_file_manager_vm, mock_file_items, mocker):
        """align_channels with no reference should return early."""
        vm = mock_file_manager_vm
        # Don't set reference
        # Should return early without doing anything
        result = vm.align_channels(mock_file_items)
        assert result is None  # Returns None when no reference

    def test_alignment_complete_updates_two_cycles_and_protein(
        self, mock_file_manager_vm, mock_file_items
    ):
        vm = mock_file_manager_vm
        ref_item, cy1_item, cy2_item = mock_file_items[:3]
        protein_item = FileItem(path="/tmp/protein_assigned.tif", status=FileStatus.RAW)

        vm.files[ref_item.path] = ref_item
        vm.files[cy1_item.path] = cy1_item
        vm.files[cy2_item.path] = cy2_item
        vm.files[protein_item.path] = protein_item
        vm.reference_item = ref_item
        vm.dataset_cycle_assignments = {0: ref_item, 1: cy1_item, 2: cy2_item}
        vm.dataset_protein_file = protein_item

        aligned_images = [
            np.full((2, 32, 32), 11, dtype=np.uint16),
            np.full((2, 32, 32), 22, dtype=np.uint16),
            np.full((2, 32, 32), 33, dtype=np.uint16),
        ]

        dialog = MagicMock()
        dialog.exec.return_value = QDialog.DialogCode.Accepted

        with patch.object(vm, "_get_brightfield_image", return_value=np.zeros((32, 32), dtype=np.uint16)), \
             patch("viewmodel.file_manager_vm.AlignmentPreviewDialog", return_value=dialog) as mock_dialog:
            vm._on_alignment_complete(aligned_images, [cy1_item, cy2_item, protein_item])

        assert mock_dialog.call_args.kwargs["layer_labels"] == [
            "Cycle 2",
            "Cycle 3",
            "Protein",
        ]

        for expected, file_item in zip(aligned_images, [cy1_item, cy2_item, protein_item]):
            saved = vm.files[file_item.path]
            assert saved.status == FileStatus.ALIGNED
            assert saved.metadata.prefix == FileStatus.ALIGNED.name.lower()
            np.testing.assert_array_equal(saved.working_image, expected)

    def test_cancel_alignment(self, mock_file_manager_vm, mock_register_thread):
        """cancel_alignment should call cancel on register thread."""
        vm = mock_file_manager_vm
        vm.register_thread = mock_register_thread

        vm.cancel_alignment()

        mock_register_thread.cancel.assert_called_once()

    def test_cancel_bead_generation(self, mock_file_manager_vm, mock_bead_thread):
        """cancel_bead_generation should call cancel on bead thread."""
        vm = mock_file_manager_vm
        vm.bead_thread = mock_bead_thread

        vm.cancel_bead_generation()

        mock_bead_thread.cancel.assert_called_once()

    def test_export_files(self, mock_file_manager_vm, tmp_path, mocker):
        """export_files should be callable without error."""
        vm = mock_file_manager_vm

        # Create a simple file item
        from model.file_item import FileItem
        path = tmp_path / "test.tif"
        path.write_text("test content")
        item = FileItem(path=str(path))
        vm.files[str(path)] = item

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        mock_image = np.zeros((1, 512, 512), dtype=np.uint16)

        with patch('viewmodel.file_manager_vm.load_image', return_value=mock_image):
            # Should run without error
            vm.export_files(str(export_folder), [item])

    def test_export_task_exports_aligned_two_cycles_and_protein(
        self, tmp_path
    ):
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        def make_item(filename: str, fill_value: int) -> tuple[FileItem, np.ndarray]:
            image_path = tmp_path / filename
            tifffile.imwrite(str(image_path), np.zeros((2, 16, 16), dtype=np.uint16))
            item = FileItem(path=str(image_path), status=FileStatus.ALIGNED)
            item.metadata.max_size = 16
            item.metadata.prefix = FileStatus.ALIGNED.name.lower()
            aligned_image = np.full((2, 16, 16), fill_value, dtype=np.uint16)
            item.working_image = aligned_image
            return item, aligned_image

        cy1_item, cy1_aligned = make_item("cycle1.tif", 101)
        cy2_item, cy2_aligned = make_item("cycle2.tif", 202)
        protein_item, protein_aligned = make_item("protein.tif", 303)

        files = {
            cy1_item.path: cy1_item,
            cy2_item.path: cy2_item,
            protein_item.path: protein_item,
        }
        selected = [cy1_item, cy2_item, protein_item]

        stop = threading.Event()
        for _ in export_task(str(export_folder), files, selected, stop):
            pass

        exported = sorted(f.name for f in export_folder.glob("*.tif"))
        assert exported == [
            "aligned_cycle1.tif",
            "aligned_cycle2.tif",
            "aligned_protein.tif",
        ]

        np.testing.assert_array_equal(
            tifffile.imread(str(export_folder / "aligned_cycle1.tif")),
            cy1_aligned,
        )
        np.testing.assert_array_equal(
            tifffile.imread(str(export_folder / "aligned_cycle2.tif")),
            cy2_aligned,
        )
        np.testing.assert_array_equal(
            tifffile.imread(str(export_folder / "aligned_protein.tif")),
            protein_aligned,
        )

    def test_apply_metadata_shape_uses_min_when_max_size_larger_than_original(self, mock_file_manager_vm, mock_file_item):
        """When max_size is larger than original shape, shape should use min (original dimensions)."""
        vm = mock_file_manager_vm
        item = mock_file_item
        vm.files[item.path] = item

        original_shape = item.shape
        original_height = original_shape[1]
        original_width = original_shape[2]
        new_max_size = 10000

        assert original_height == 512
        assert original_width == 512

        vm.apply_metadata({"max_size": new_max_size}, [item])

        expected_height = min(original_height, new_max_size)
        expected_width = min(original_width, new_max_size)
        expected_shape = (original_shape[0], expected_height, expected_width)

        assert vm.files[item.path].shape == expected_shape, \
            f"Shape should be {expected_shape} but got {vm.files[item.path].shape}"

    def test_apply_metadata_shape_updates_to_max_size_when_smaller(self, mock_file_manager_vm, mock_file_item):
        """When max_size is smaller than original shape, shape should update to max_size."""
        vm = mock_file_manager_vm
        item = mock_file_item
        vm.files[item.path] = item

        original_shape = item.shape
        original_height = original_shape[1]
        original_width = original_shape[2]
        new_max_size = 256

        assert original_height == 512
        assert original_width == 512

        vm.apply_metadata({"max_size": new_max_size}, [item])

        expected_shape = (original_shape[0], min(original_height, new_max_size), min(original_width, new_max_size))
        assert vm.files[item.path].shape == expected_shape

    def test_apply_metadata_emits_corrected_signal_when_max_size_too_large(self, mock_file_manager_vm, mock_file_item, signal_recorder):
        """When max_size is larger than original dimensions, corrected signal should be emitted."""
        vm = mock_file_manager_vm
        item = mock_file_item
        vm.files[item.path] = item

        original_shape = item.shape
        original_height = original_shape[1]
        new_max_size = 10000

        signal_recorder.connect(vm.metadata_corrected_sig)

        vm.apply_metadata({"max_size": new_max_size}, [item])

        assert signal_recorder.get_call_count() == 1
        call_args = signal_recorder.get_last_args()
        assert call_args is not None
        args, kwargs = call_args
        corrected_values = args[0]
        assert "max_size" in corrected_values
        assert corrected_values["max_size"] == min(original_height, original_shape[2])

    def test_apply_metadata_no_corrected_signal_when_max_size_valid(self, mock_file_manager_vm, mock_file_item, signal_recorder):
        """When max_size is valid (smaller than or equal to original), no corrected signal should be emitted."""
        vm = mock_file_manager_vm
        item = mock_file_item
        vm.files[item.path] = item

        new_max_size = 256

        signal_recorder.connect(vm.metadata_corrected_sig)

        vm.apply_metadata({"max_size": new_max_size}, [item])

        assert signal_recorder.get_call_count() == 0

    def test_load_file_duplicate_detection(self, mock_file_manager_vm, tmp_tiff_path):
        """Loading the same file twice should emit error and skip second load."""
        from unittest.mock import patch
        vm = mock_file_manager_vm
        path = tmp_tiff_path()

        with patch('viewmodel.file_manager_vm.get_tif_info') as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)
            
            # First load - should succeed
            vm.load_file(path)
            assert path in vm.files
            assert len(vm.files) == 1
            
            # Mock the align_error signal to track calls
            error_calls = []
            def capture_error(msg):
                error_calls.append(msg)
            vm.align_error.connect(capture_error)
            
            # Second load - should be skipped
            vm.load_file(path)
            
            # File should still exist only once
            assert len(vm.files) == 1
            assert path in vm.files
            
            # Error should be emitted
            assert len(error_calls) == 1
            assert "already loaded" in error_calls[0]

    def test_load_folder_duplicate_detection(self, mock_file_manager_vm, tmp_tiff_folder):
        """Loading folder with already loaded files should skip duplicates and emit error."""
        from unittest.mock import patch
        vm = mock_file_manager_vm
        folder = tmp_tiff_folder()
        
        with patch('viewmodel.file_manager_vm.list_tiff_files') as mock_list_files, \
             patch('viewmodel.file_manager_vm.get_tif_info') as mock_info:
            
            mock_list_files.return_value = [f"{folder}/file1.tif", f"{folder}/file2.tif"]
            mock_info.return_value = ((1, 512, 512), np.uint16)
            
            # First load - should load both files
            vm.load_folder(folder)
            assert len(vm.files) == 2
            
            # Mock the align_error signal to track calls
            error_calls = []
            def capture_error(msg):
                error_calls.append(msg)
            vm.align_error.connect(capture_error)
            
            # Second load - should skip both files
            vm.load_folder(folder)
            
            # Files should still exist only once each
            assert len(vm.files) == 2
            
            # Error should be emitted about skipped files
            assert len(error_calls) == 1
            assert "Skipped 2" in error_calls[0]
            assert "already loaded" in error_calls[0]

    def test_recompute_ensemble_sweep_updates_file_item(self, mock_file_manager_vm, mock_file_item):
        vm = mock_file_manager_vm
        vm.files[mock_file_item.path] = mock_file_item
        mock_file_item.pre_ensemble_beads = pd.DataFrame(
            {"x": [1.0], "y": [1.0], "cy0": [255], "cy1": [255]}
        )
        mock_file_item.ensemble_cache = {"cached": True}
        mock_file_item.ensemble_ratio_applied = 1.0

        expected_sweep = pd.DataFrame(
            [
                {"ratio": 1.0, "valid_pct": 20.0, "invalid_pct": 10.0, "filtered_pct": 70.0},
                {"ratio": 1.05, "valid_pct": 21.0, "invalid_pct": 9.0, "filtered_pct": 70.0},
            ]
        )

        with patch("image_processing.compute_ensemble_sweep_stats", return_value=expected_sweep):
            out = vm.recompute_ensemble_sweep(mock_file_item, 1.0, 1.05, 0.05)

        assert out.equals(expected_sweep)
        assert vm.files[mock_file_item.path].ensemble_sweep_stats.equals(expected_sweep)
        assert vm.files[mock_file_item.path].ensemble_ratio_selected == 1.0

    def test_apply_ensemble_ratio_updates_beads(self, mock_file_manager_vm, mock_file_item):
        vm = mock_file_manager_vm
        vm.files[mock_file_item.path] = mock_file_item
        mock_file_item.pre_ensemble_beads = pd.DataFrame(
            {"x": [1.0], "y": [1.0], "cy0": [255], "cy1": [255]}
        )
        mock_file_item.ensemble_cache = {"cached": True}
        new_beads = pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [1], "cy1": [2]})

        with patch("image_processing.build_ensembled_beads_from_cache", return_value=new_beads):
            out = vm.apply_ensemble_ratio(mock_file_item, 1.1)

        assert out.equals(new_beads)
        assert vm.files[mock_file_item.path].beads.equals(new_beads)
        assert vm.files[mock_file_item.path].ensemble_ratio_applied == 1.1
        assert vm.files[mock_file_item.path].ensemble_ratio_selected == 1.1

    def test_remove_ensemble_applied_changes_reverts_to_pre_ensemble(self, mock_file_manager_vm, mock_file_item):
        vm = mock_file_manager_vm
        vm.files[mock_file_item.path] = mock_file_item
        pre = pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [255], "cy1": [255]})
        mock_file_item.pre_ensemble_beads = pre.copy()
        mock_file_item.beads = pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [1], "cy1": [2]})
        mock_file_item.ensemble_ratio_applied = 1.1
        mock_file_item.ensemble_ratio_selected = 1.1
        mock_file_item.ensemble_sweep_stats = pd.DataFrame(
            [{"ratio": 1.0}, {"ratio": 1.05}]
        )

        out = vm.remove_ensemble_applied_changes(mock_file_item)

        assert out.equals(pre)
        assert vm.files[mock_file_item.path].beads.equals(pre)
        assert vm.files[mock_file_item.path].ensemble_ratio_applied is None
        assert vm.files[mock_file_item.path].ensemble_ratio_selected == 1.0

    def _run_bead_task(self, gen):
        emitted = []
        result = None
        try:
            while True:
                emitted.append(next(gen))
        except StopIteration as e:
            result = e.value
        return emitted, result

    def test_bead_generation_task_uses_time_calibrated_progress(self, qapp, tmp_tiff_path):
        ref_path = tmp_tiff_path("ref.tif")
        cy1_path = tmp_tiff_path("cy1.tif")
        ref_item = FileItem(path=ref_path, status=FileStatus.RAW)
        ref_item.metadata.max_size = 32
        ref_item.metadata.reference_channel = 0
        cy1_item = FileItem(path=cy1_path, status=FileStatus.RAW)
        cy1_item.metadata.max_size = 32
        cy1_item.metadata.reference_channel = 0

        ref_saved = FileItem(path=ref_path, status=FileStatus.RAW)
        ref_saved.metadata.max_size = 32
        ref_saved.metadata.reference_channel = 0
        ref_saved.working_image = np.zeros((32, 32), dtype=np.uint16)
        cy1_saved = FileItem(path=cy1_path, status=FileStatus.RAW)
        cy1_saved.metadata.max_size = 32
        cy1_saved.metadata.reference_channel = 0
        cy1_saved.working_image = np.zeros((32, 32), dtype=np.uint16)

        files = {ref_path: ref_saved, cy1_path: cy1_saved}

        def fake_process_beads(
            brightfield, tifs, max_size, signal_to_noise_cutoff,
            progress_callback=None, progress_units_callback=None, **kwargs,
        ):
            if progress_callback:
                progress_callback(0, "Preprocessing brightfield image...")
                progress_callback(10, "Initial bead detection...")
                progress_callback(30, "Getting activation regions from cycles")
            if progress_units_callback:
                progress_units_callback("activation_regions", 1, 4)
                progress_units_callback("activation_regions", 2, 4)
                progress_units_callback("activation_regions", 3, 4)
                progress_units_callback("activation_regions", 4, 4)
            if progress_callback:
                progress_callback(60, "Assigning beads labels")
                progress_callback(95, "Bead generation complete.")
            return {
                "beads": pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}),
                "post_resolution_beads": pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}),
                "cycles": {"cy0": np.zeros((2, 32, 32), dtype=np.uint16)},
                "labeled_image": np.zeros((32, 32), dtype=np.uint16),
            }

        stop = threading.Event()
        gen = bead_generation_task(
            ref_item, [ref_item, cy1_item], files, 0.1, stop, use_stardist=True
        )

        with patch("viewmodel.tasks.load_and_constrain_image", return_value=np.zeros((2, 32, 32), dtype=np.uint16)), \
             patch("image_processing.process_beads", side_effect=fake_process_beads):
            emitted, result = self._run_bead_task(gen)

        assert result is not None
        assert emitted
        pre_values = [v for v, m in emitted if "Preprocessing brightfield image..." in m]
        assert pre_values
        assert pre_values[0] < 30
        assert any("Processing fluorescence channels" in m for _, m in emitted)
        assert any(
            ("/s" in m or "(~" in m or " · ~" in m)
            for v, m in emitted if v > 0 and v < 100
        )
        assert not any("Elapsed" in m for v, m in emitted)
        assert max(v for v, _ in emitted if v >= 0) <= 99

    def test_bead_generation_task_stardist_sets_workload_before_processing(
        self, qapp, tmp_tiff_path
    ):
        ref_path = tmp_tiff_path("ref_workload.tif")
        cy1_path = tmp_tiff_path("cy_workload.tif")
        ref_item = FileItem(path=ref_path, status=FileStatus.RAW)
        ref_item.metadata.max_size = 32
        ref_item.metadata.reference_channel = 0
        cy1_item = FileItem(path=cy1_path, status=FileStatus.RAW)
        cy1_item.metadata.max_size = 32
        cy1_item.metadata.reference_channel = 0

        ref_saved = FileItem(path=ref_path, status=FileStatus.RAW)
        ref_saved.metadata.max_size = 32
        ref_saved.metadata.reference_channel = 0
        ref_saved.working_image = np.zeros((3, 32, 32), dtype=np.uint16)
        cy1_saved = FileItem(path=cy1_path, status=FileStatus.RAW)
        cy1_saved.metadata.max_size = 32
        cy1_saved.metadata.reference_channel = 0
        cy1_saved.working_image = np.zeros((3, 32, 32), dtype=np.uint16)

        files = {ref_path: ref_saved, cy1_path: cy1_saved}
        set_workload_calls = []

        original_set_workload = BeadEtaEstimator.set_workload

        def capturing_set_workload(self, total_channels, max_size_pixels):
            set_workload_calls.append({"total_channels": total_channels, "max_size_pixels": max_size_pixels})
            return original_set_workload(self, total_channels, max_size_pixels)

        def fake_process_beads(*args, **kwargs):
            return {
                "beads": pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}),
                "post_resolution_beads": pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}),
                "cycles": {"cy0": np.zeros((3, 32, 32), dtype=np.uint16)},
                "labeled_image": np.zeros((32, 32), dtype=np.uint16),
            }

        stop = threading.Event()
        gen = bead_generation_task(
            ref_item, [ref_item, cy1_item], files, 0.1, stop, use_stardist=True
        )

        with patch.object(BeadEtaEstimator, "set_workload", capturing_set_workload), \
             patch("viewmodel.tasks.load_and_constrain_image", return_value=np.zeros((3, 32, 32), dtype=np.uint16)), \
             patch("image_processing.process_beads", side_effect=fake_process_beads):
            self._run_bead_task(gen)

        assert len(set_workload_calls) == 1
        assert set_workload_calls[0]["total_channels"] == 4
        assert set_workload_calls[0]["max_size_pixels"] == 32

    def test_bead_generation_task_heartbeat_emits_between_sparse_callbacks(self, qapp, tmp_tiff_path):
        ref_path = tmp_tiff_path("ref2.tif")
        cy1_path = tmp_tiff_path("cy2.tif")
        ref_item = FileItem(path=ref_path, status=FileStatus.RAW)
        ref_item.metadata.max_size = 32
        ref_item.metadata.reference_channel = 0
        cy1_item = FileItem(path=cy1_path, status=FileStatus.RAW)
        cy1_item.metadata.max_size = 32
        cy1_item.metadata.reference_channel = 0

        ref_saved = FileItem(path=ref_path, status=FileStatus.RAW)
        ref_saved.metadata.max_size = 32
        ref_saved.metadata.reference_channel = 0
        ref_saved.working_image = np.zeros((32, 32), dtype=np.uint16)
        cy1_saved = FileItem(path=cy1_path, status=FileStatus.RAW)
        cy1_saved.metadata.max_size = 32
        cy1_saved.metadata.reference_channel = 0
        cy1_saved.working_image = np.zeros((32, 32), dtype=np.uint16)

        files = {ref_path: ref_saved, cy1_path: cy1_saved}

        def fake_process_beads(
            brightfield, tifs, max_size, signal_to_noise_cutoff,
            progress_callback=None, progress_units_callback=None, **kwargs,
        ):
            if progress_callback:
                progress_callback(10, "Initial bead detection...")
            time.sleep(1.1)
            if progress_callback:
                progress_callback(95, "Bead generation complete.")
            return {
                "beads": pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}),
                "post_resolution_beads": pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}),
                "cycles": {"cy0": np.zeros((2, 32, 32), dtype=np.uint16)},
                "labeled_image": np.zeros((32, 32), dtype=np.uint16),
            }

        stop = threading.Event()
        gen = bead_generation_task(
            ref_item, [ref_item, cy1_item], files, 0.1, stop, use_stardist=True
        )

        with patch("viewmodel.tasks.load_and_constrain_image", return_value=np.zeros((2, 32, 32), dtype=np.uint16)), \
             patch("image_processing.process_beads", side_effect=fake_process_beads):
            emitted, _ = self._run_bead_task(gen)

        detection_updates = [x for x in emitted if "Initial bead detection..." in x[1]]
        assert len(detection_updates) >= 1
        assert len(detection_updates) <= 3
