"""
End-to-end tests for multi-selection workflow.
"""

from unittest.mock import patch

import numpy as np

from model.file_item import FileItem
from model.status_enum import FileStatus


class TestMultiSelectionE2E:
    """End-to-end tests for multi-selection."""

    def test_select_single_file(self, mock_main_window, mock_file_items, qtbot):
        """Selecting one file should work."""
        window = mock_main_window

        window.update_file_list(mock_file_items)
        window.file_table_widget.selectRow(0)
        window.handle_selection_change()

        selected = window.get_selected_files()

        assert len(selected) >= 1

    def test_get_selected_files_returns_correct_items(
        self, mock_main_window, mock_file_items, qtbot
    ):
        """get_selected_files should return FileItem objects."""
        window = mock_main_window

        window.update_file_list(mock_file_items)
        window.file_table_widget.selectRow(0)

        selected = window.get_selected_files()

        for item in selected:
            assert isinstance(item, FileItem)


class TestReferenceSettingE2E:
    """End-to-end tests for reference setting."""

    def test_set_reference_updates_vm(self, mock_file_manager_vm, mock_file_items):
        """set_reference should update vm.reference_item."""
        vm = mock_file_manager_vm

        for item in mock_file_items:
            vm.files[item.path] = item

        vm.set_reference(mock_file_items[0])

        assert vm.reference_item == mock_file_items[0]

    def test_set_reference_updates_reference_item(
        self, mock_file_manager_vm, mock_file_items
    ):
        """set_reference should update reference_item but NOT change status."""
        vm = mock_file_manager_vm

        for item in mock_file_items:
            vm.files[item.path] = item

        vm.set_reference(mock_file_items[1])

        assert vm.reference_item == mock_file_items[1]
        assert vm.files[mock_file_items[1].path].status == FileStatus.RAW

    def test_change_reference_keeps_status(self, mock_file_manager_vm, mock_file_items):
        """Changing reference should NOT change any file's status."""
        vm = mock_file_manager_vm

        for item in mock_file_items:
            item.status = FileStatus.SHADE_CORRECTED
            vm.files[item.path] = item

        vm.set_reference(mock_file_items[0])
        vm.set_reference(mock_file_items[1])

        assert vm.files[mock_file_items[0].path].status == FileStatus.SHADE_CORRECTED
        assert vm.files[mock_file_items[1].path].status == FileStatus.SHADE_CORRECTED

    def test_only_one_reference_at_a_time(self, mock_file_manager_vm, mock_file_items):
        """Only the last set file should be the reference."""
        vm = mock_file_manager_vm

        for item in mock_file_items:
            vm.files[item.path] = item

        for item in mock_file_items:
            vm.set_reference(item)

        assert vm.reference_item == mock_file_items[-1]


class TestFullWorkflowE2E:
    """End-to-end tests for complete workflows."""

    def test_load_and_reference_workflow(
        self, mock_main_window, tmp_tiff_folder, qtbot
    ):
        """Complete workflow: Load folder and set reference."""
        window = mock_main_window

        folder_path = tmp_tiff_folder(3)
        with patch("viewmodel.file_manager_vm.get_tif_info") as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)
            window.vm.load_folder(folder_path)

        assert len(window.vm.files) == 3

        # Set reference
        files = list(window.vm.files.values())
        original_status = files[0].status
        window.vm.set_reference(files[0])

        assert window.vm.reference_item == files[0]
        assert window.vm.files[files[0].path].status == original_status

    def test_table_row_count_matches_loaded_files(
        self, mock_main_window, tmp_tiff_folder, qtbot
    ):
        """Table row count should match loaded files."""
        window = mock_main_window

        folder_path = tmp_tiff_folder(5)
        with patch("viewmodel.file_manager_vm.get_tif_info") as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)
            window.vm.load_folder(folder_path)

        assert window.file_table_widget.rowCount() == len(window.vm.files)
