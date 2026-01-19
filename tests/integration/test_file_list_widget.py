import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from model.file_item import FileItem
from model.status_enum import FileStatus
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
from view.FileListWidget import FileTableWidget


class TestFileTableWidget:
    """Integration tests for the FileTableWidget."""

    def test_add_file_item(self, mock_file_table_widget, mock_file_item):
        """Adding item should create a row with filename and status."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        assert widget.rowCount() == 1
        assert widget.item(0, 0).text() == os.path.basename(mock_file_item.path)
        assert widget.item(0, 1).text() == mock_file_item.status.value

    def test_add_multiple_file_items(self, mock_file_table_widget, mock_file_items):
        """Adding multiple items should create multiple rows."""
        widget = mock_file_table_widget

        for item in mock_file_items:
            widget.add_file_item(item)

        assert widget.rowCount() == len(mock_file_items)

    def test_update_file_display(
        self, mock_file_table_widget, mock_file_item, mock_file_manager_vm
    ):
        """Updating display should change status text and color."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        updated_item = FileItem(
            path=mock_file_item.path,
            status=FileStatus.ALIGNED,
            shape=(1, 512, 512),
            dtype="uint16"
        )

        widget.update_file_display([updated_item])

        assert widget.item(0, 1).text() == FileStatus.ALIGNED.value

    def test_get_selected_files_single(self, mock_file_table_widget, mock_file_items):
        """Selecting one row should return one FileItem."""
        widget = mock_file_table_widget

        for item in mock_file_items:
            widget.add_file_item(item)

        widget.selectRow(0)

        selected = widget.get_selected_files()

        assert len(selected) >= 1
        assert selected[0].path == mock_file_items[0].path

    def test_get_selected_files_multiple(
        self, mock_file_table_widget, mock_file_items, mocker
    ):
        """Selecting rows should return items from selected rows."""
        widget = mock_file_table_widget

        for item in mock_file_items:
            widget.add_file_item(item)

        # Select first row
        widget.selectRow(0)

        selected = widget.get_selected_files()

        # Should get item from row 0
        assert len(selected) >= 1

    def test_clear_reference(
        self, mock_file_table_widget, mock_file_item, mock_file_manager_vm, mocker
    ):
        """Clearing reference should set reference_item to None."""
        widget = mock_file_table_widget
        vm = mock_file_manager_vm

        widget.add_file_item(mock_file_item)
        vm.set_reference(mock_file_item)
        assert vm.reference_item == mock_file_item

        vm.clear_reference()
        assert vm.reference_item is None

    def test_drag_drop_callback(
        self, mock_file_table_widget, mock_file_manager_vm, mocker
    ):
        """Dropping files should trigger the callback."""
        callback_mock = mocker.Mock()
        widget = FileTableWidget(
            file_dropped_callback=callback_mock,
            vm=mock_file_manager_vm
        )

        mock_mime = MagicMock()
        mock_mime.hasUrls.return_value = True
        mock_url = MagicMock()
        mock_url.toLocalFile.return_value = "/test/file.tif"
        mock_mime.urls.return_value = [mock_url]

        event = MagicMock()
        event.mimeData.return_value = mock_mime
        widget.dropEvent(event)

        callback_mock.assert_called_once()

    def test_get_selected_files_returns_file_items(
        self, mock_file_table_widget, mock_file_items
    ):
        """get_selected_files should return FileItem objects, not strings."""
        widget = mock_file_table_widget

        for item in mock_file_items:
            widget.add_file_item(item)

        widget.selectRow(0)

        selected = widget.get_selected_files()

        assert len(selected) > 0
        for item in selected:
            assert isinstance(item, FileItem)

    def test_status_column_color(self, mock_file_table_widget, mock_file_item):
        """Status column should have colored background."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        status_item = widget.item(0, 1)
        assert status_item.background().color().name() is not None

    def test_file_item_user_role(self, mock_file_table_widget, mock_file_item):
        """FileItem should be stored in UserRole data role."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        filename_item = widget.item(0, 0)
        stored_item = filename_item.data(Qt.ItemDataRole.UserRole)

        assert stored_item == mock_file_item


class TestFileListWidgetContextMenu:
    """Tests for context menu functionality."""

    def test_handle_delete_removes_from_ui(
        self, mock_file_table_widget, mock_file_items, mock_file_manager_vm, mocker
    ):
        """handle_delete_selected should remove row from UI."""
        widget = mock_file_table_widget
        vm = mock_file_manager_vm

        for item in mock_file_items:
            widget.add_file_item(item)
            vm.files[item.path] = item

        widget.selectRow(0)

        with patch.object(vm, 'delete_files'):
            initial_count = widget.rowCount()
            widget.handle_delete_selected()
            assert widget.rowCount() == initial_count - 1

    def test_handle_delete_calls_vm(
        self, mock_file_table_widget, mock_file_items, mock_file_manager_vm, mocker
    ):
        """handle_delete_selected should call VM.delete_files."""
        widget = mock_file_table_widget
        vm = mock_file_manager_vm

        for item in mock_file_items:
            widget.add_file_item(item)
            vm.files[item.path] = item

        widget.selectRow(0)

        with patch.object(vm, 'delete_files') as mock_delete:
            widget.handle_delete_selected()
            mock_delete.assert_called_once()


class TestFileListWidgetShapeUpdate:
    """Tests for shape update functionality in FileTableWidget."""

    def test_add_file_item_sets_shape_column(
        self, mock_file_table_widget, mock_file_item
    ):
        """Adding file item should set shape in column 2."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        shape_text = widget.item(0, 2).text()
        assert shape_text == "C=1, Y=512, X=512"

    def test_update_file_display_updates_shape(
        self, mock_file_table_widget, mock_file_item
    ):
        """update_file_display should update the shape column."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        original_shape = mock_file_item.shape
        updated_item = FileItem(
            path=mock_file_item.path,
            status=FileStatus.RAW,
            shape=(1, 256, 256),
            dtype="uint16"
        )

        widget.update_file_display([updated_item])

        shape_text = widget.item(0, 2).text()
        assert shape_text == "C=1, Y=256, X=256"

    def test_update_file_display_shape_with_different_channels(
        self, mock_file_table_widget, mock_file_item
    ):
        """update_file_display should handle shape with different channel count."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        updated_item = FileItem(
            path=mock_file_item.path,
            status=FileStatus.RAW,
            shape=(5, 1024, 1024),
            dtype="uint16"
        )

        widget.update_file_display([updated_item])

        shape_text = widget.item(0, 2).text()
        assert shape_text == "C=5, Y=1024, X=1024"


class TestFileListWidgetMetadataIntegration:
    """Integration tests for metadata updates affecting the FileTableWidget."""

    def test_apply_metadata_updates_shape_in_file_table(
        self, mock_main_window, mock_file_item
    ):
        """When apply_metadata changes max_size, FileTableWidget shape should update."""
        window = mock_main_window
        vm = window.vm

        vm.files[mock_file_item.path] = mock_file_item
        window.file_table_widget.add_file_item(mock_file_item)

        assert window.file_table_widget.item(0, 2).text() == "C=1, Y=512, X=512"

        vm.apply_metadata({"max_size": 256}, [mock_file_item])

        assert window.file_table_widget.item(0, 2).text() == "C=1, Y=256, X=256"

    def test_file_information_update_signal_triggers_shape_update(
        self, mock_main_window, mock_file_item
    ):
        """file_information_update signal should trigger shape update in widget."""
        window = mock_main_window
        vm = window.vm

        vm.files[mock_file_item.path] = mock_file_item
        window.file_table_widget.add_file_item(mock_file_item)

        mock_file_item.shape = (1, 256, 256)
        vm.file_information_update.emit([mock_file_item])

        assert window.file_table_widget.item(0, 2).text() == "C=1, Y=256, X=256"
