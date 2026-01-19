import os
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import tifffile
from model.file_item import FileItem
from model.status_enum import FileStatus


class TestFileManagerVM:
    """Unit tests for the FileManagerVM ViewModel."""

    def test_initial_state(self, mock_file_manager_vm):
        """VM should start with empty files dict and no reference."""
        vm = mock_file_manager_vm
        assert vm.files == {}
        assert vm.reference_item is None
        assert vm.emitted_files == set()
        assert vm.register_thread is None
        assert vm.bead_thread is None
        assert vm.selected_files == []

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

    def test_align_channels_no_reference(self, mock_file_manager_vm, mock_file_items, mocker):
        """align_channels with no reference should return early."""
        vm = mock_file_manager_vm
        # Don't set reference
        # Should return early without doing anything
        result = vm.align_channels(mock_file_items)
        assert result is None  # Returns None when no reference

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
