"""
End-to-end tests for file loading workflow.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from model.file_item import FileItem
from model.status_enum import FileStatus


class TestFileLoadingE2E:
    """End-to-end tests for file loading."""

    def test_load_single_file_via_menu(
        self, mock_main_window, mock_file_manager_vm, tmp_tiff_path, qtbot
    ):
        """Loading a single file should add it to VM."""
        window = mock_main_window
        path = tmp_tiff_path()

        with patch(
            "PyQt6.QtWidgets.QFileDialog.getOpenFileName", return_value=(path, "")
        ):
            window.handle_dropped_paths([path])

        assert path in window.vm.files
        assert isinstance(window.vm.files[path], FileItem)

    def test_load_folder_via_menu(
        self, mock_main_window, mock_file_manager_vm, tmp_tiff_folder, qtbot
    ):
        """Loading a folder should add all TIFFs to VM."""
        window = mock_main_window
        folder_path = tmp_tiff_folder(3)

        with patch(
            "PyQt6.QtWidgets.QFileDialog.getExistingDirectory", return_value=folder_path
        ):
            window.handle_dropped_paths([folder_path])

        assert len(window.vm.files) == 3

    def test_drag_drop_single_file(
        self, mock_main_window, mock_file_manager_vm, tmp_tiff_path, qtbot
    ):
        """Drag-dropping a file should load it."""
        window = mock_main_window
        path = tmp_tiff_path()

        window.handle_dropped_paths([path])

        assert path in window.vm.files

    def test_drop_file_on_table_widget(
        self, mock_main_window, mock_file_manager_vm, tmp_tiff_path, qtbot
    ):
        """Dropping file on table should trigger callback."""
        widget = mock_main_window.file_table_widget
        path = tmp_tiff_path()

        mock_mime = MagicMock()
        mock_mime.hasUrls.return_value = True
        mock_url = MagicMock()
        mock_url.toLocalFile.return_value = path
        mock_mime.urls.return_value = [mock_url]

        with patch.object(widget, "file_dropped_callback") as callback:
            event = MagicMock()
            event.mimeData.return_value = mock_mime
            widget.dropEvent(event)
            callback.assert_called_once_with([path])

    def test_multiple_files_appear_in_table(
        self, mock_main_window, tmp_tiff_folder, qtbot
    ):
        """Multiple loaded files should appear in the table."""
        window = mock_main_window
        folder_path = tmp_tiff_folder(3)

        with patch("viewmodel.file_manager_vm.get_tif_info") as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)
            window.vm.load_folder(folder_path)

        assert len(window.vm.files) == 3

    def test_table_shows_correct_initial_status(
        self, mock_main_window, tmp_tiff_path, qtbot
    ):
        """Loaded file should show RAW status initially."""
        window = mock_main_window
        path = tmp_tiff_path()

        with patch("viewmodel.file_manager_vm.get_tif_info") as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)
            window.vm.load_file(path)

        assert window.vm.files[path].status == FileStatus.RAW

    def test_table_sorting_enabled(self, mock_main_window, tmp_tiff_folder, qtbot):
        """Table should have sorting enabled."""
        window = mock_main_window
        assert window.file_table_widget.isSortingEnabled()

    def test_table_columns_configured(self, mock_main_window, qtbot):
        """Table should have 4 columns with correct headers."""
        widget = mock_main_window.file_table_widget

        assert widget.columnCount() == 4
        assert widget.horizontalHeaderItem(0).text() == "Filename"
        assert widget.horizontalHeaderItem(1).text() == "Status"


class TestFileLoadingErrors:
    """Tests for error handling during file loading."""

    def test_invalid_file_path(self, mock_main_window, mock_file_manager_vm, qtbot):
        """Invalid file path should not add file to VM."""
        window = mock_main_window
        invalid_path = "/non/existent/file.tif"

        window.handle_dropped_paths([invalid_path])

        assert invalid_path not in window.vm.files

    def test_empty_folder(
        self, mock_main_window, mock_file_manager_vm, tmp_path, qtbot
    ):
        """Empty folder should result in empty VM."""
        window = mock_main_window
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        with patch("viewmodel.file_manager_vm.list_tiff_files", return_value=[]):
            window.handle_dropped_paths([str(empty_folder)])

        assert len(window.vm.files) == 0


class TestFileLoadingSignals:
    """Tests for signal handling during file loading."""

    def test_emitted_files_tracked(self, mock_main_window, tmp_tiff_path, mocker):
        """emitted_files should track which files were emitted."""
        window = mock_main_window
        path = tmp_tiff_path()

        with patch("viewmodel.file_manager_vm.get_tif_info") as mock_info:
            mock_info.return_value = ((1, 512, 512), np.uint16)
            window.vm.load_file(path)

        assert path in window.vm.emitted_files
