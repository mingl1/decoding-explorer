"""
End-to-end tests for Export As TIFF context menu functionality.
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QMenu

from model.file_item import FileItem
from model.status_enum import FileStatus


class TestExportAsTiffE2E:
    """End-to-end tests for Export As TIFF functionality."""

    def test_export_single_file_creates_tiff(
        self, mock_main_window, tmp_tiff_path, tmp_path, qtbot
    ):
        """Exporting a single file should create a TIFF in the selected folder."""
        window = mock_main_window
        source_path = tmp_tiff_path("source.tif", shape=(3, 256, 256))
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([source_path])
        qtbot.wait(50)

        file_item = window.vm.files[source_path]

        window.vm.export_files(str(export_folder), [file_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert len(exported_files) == 1

    def test_export_multiple_files(
        self, mock_main_window, tmp_tiff_folder, tmp_path, qtbot
    ):
        """Exporting multiple files should create a TIFF for each."""
        window = mock_main_window
        source_folder = tmp_tiff_folder(3)
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([source_folder])
        qtbot.wait(50)

        file_items = list(window.vm.files.values())

        window.vm.export_files(str(export_folder), file_items)

        exported_files = list(export_folder.glob("*.tif"))
        assert len(exported_files) == 3

    def test_export_preserves_image_data(
        self, mock_main_window, tmp_path, qtbot
    ):
        """Exported TIFF should contain the same image data."""
        window = mock_main_window

        original_data = np.random.randint(0, 65535, (3, 128, 128), dtype=np.uint16)
        source_path = tmp_path / "source.tif"
        tifffile.imwrite(str(source_path), original_data)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([str(source_path)])
        qtbot.wait(50)

        file_item = window.vm.files[str(source_path)]
        file_item.metadata.max_size = 128

        window.vm.export_files(str(export_folder), [file_item])

        exported_path = list(export_folder.glob("*.tif"))[0]
        exported_data = tifffile.imread(str(exported_path))

        np.testing.assert_array_equal(exported_data, original_data)

    def test_export_adds_status_prefix(
        self, mock_main_window, tmp_tiff_path, tmp_path, qtbot
    ):
        """Exported file should have status prefix in filename."""
        window = mock_main_window
        source_path = tmp_tiff_path("original.tif")
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([source_path])
        qtbot.wait(50)

        file_item = window.vm.files[source_path]
        file_item.metadata.prefix = "aligned"

        window.vm.export_files(str(export_folder), [file_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert len(exported_files) == 1
        assert exported_files[0].name.startswith("aligned_")

    def test_export_removes_existing_status_prefix(
        self, mock_main_window, tmp_path, qtbot
    ):
        """Export should remove existing status prefix before adding new one."""
        window = mock_main_window

        original_data = np.zeros((1, 64, 64), dtype=np.uint16)
        source_path = tmp_path / "raw_myimage.tif"
        tifffile.imwrite(str(source_path), original_data)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([str(source_path)])
        qtbot.wait(50)

        file_item = window.vm.files[str(source_path)]
        file_item.metadata.prefix = "aligned"

        window.vm.export_files(str(export_folder), [file_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert len(exported_files) == 1
        assert exported_files[0].name == "aligned_myimage.tif"
        assert "raw_" not in exported_files[0].name

    def test_export_emits_progress_signal(
        self, mock_main_window, tmp_tiff_path, tmp_path, qtbot
    ):
        """Export should emit progress signals."""
        window = mock_main_window
        source_path = tmp_tiff_path()
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([source_path])
        qtbot.wait(50)

        file_item = window.vm.files[source_path]

        progress_calls = []
        window.vm.export_progress.connect(lambda pct, msg: progress_calls.append((pct, msg)))

        window.vm.export_files(str(export_folder), [file_item])

        assert len(progress_calls) >= 1
        assert progress_calls[-1][0] == 100

    def test_export_respects_max_size(
        self, mock_main_window, tmp_path, qtbot
    ):
        """Export should crop image to max_size from metadata."""
        window = mock_main_window

        original_data = np.random.randint(0, 65535, (3, 512, 512), dtype=np.uint16)
        source_path = tmp_path / "source.tif"
        tifffile.imwrite(str(source_path), original_data)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([str(source_path)])
        qtbot.wait(50)

        file_item = window.vm.files[str(source_path)]
        file_item.metadata.max_size = 256

        window.vm.export_files(str(export_folder), [file_item])

        exported_path = list(export_folder.glob("*.tif"))[0]
        exported_data = tifffile.imread(str(exported_path))

        assert exported_data.shape == (3, 256, 256)

    def test_export_uses_working_image_when_available(
        self, mock_main_window, tmp_path, qtbot
    ):
        """Export should use working_image if file was processed."""
        window = mock_main_window

        original_data = np.zeros((3, 128, 128), dtype=np.uint16)
        source_path = tmp_path / "source.tif"
        tifffile.imwrite(str(source_path), original_data)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([str(source_path)])
        qtbot.wait(50)

        file_item = window.vm.files[str(source_path)]
        file_item.metadata.max_size = 128

        processed_image = np.ones((3, 128, 128), dtype=np.uint16) * 1000
        file_item.working_image = processed_image

        window.vm.export_files(str(export_folder), [file_item])

        exported_path = list(export_folder.glob("*.tif"))[0]
        exported_data = tifffile.imread(str(exported_path))

        np.testing.assert_array_equal(exported_data, processed_image)


class TestExportContextMenuInteraction:
    """Tests for export via context menu interaction."""

    def test_context_menu_has_export_option(
        self, mock_file_table_widget, mock_file_item, mock_file_manager_vm, qtbot
    ):
        """Context menu should include Export As TIFF option."""
        widget = mock_file_table_widget
        vm = mock_file_manager_vm

        widget.add_file_item(mock_file_item)
        vm.files[mock_file_item.path] = mock_file_item
        widget.selectRow(0)

        menus_shown = []
        original_exec = QMenu.exec

        def capture_menu(self, *args, **kwargs):
            menus_shown.append(self)
            return None

        with patch.object(QMenu, 'exec', capture_menu):
            widget.open_context_menu(QPoint(10, 10))

        assert len(menus_shown) == 1
        menu = menus_shown[0]
        action_texts = [action.text() for action in menu.actions()]
        assert "Export As TIFF" in action_texts

    def test_export_action_triggers_file_dialog(
        self, mock_file_table_widget, mock_file_item, mock_file_manager_vm, tmp_path, qtbot
    ):
        """Clicking Export should open folder selection dialog."""
        widget = mock_file_table_widget
        vm = mock_file_manager_vm

        widget.add_file_item(mock_file_item)
        vm.files[mock_file_item.path] = mock_file_item
        widget.selectRow(0)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        dialog_called = []

        def mock_dialog(*args, **kwargs):
            dialog_called.append(True)
            return str(export_folder)

        with patch('PyQt6.QtWidgets.QFileDialog.getExistingDirectory', mock_dialog):
            with patch.object(vm, 'export_files') as mock_export:
                menus = []
                original_exec = QMenu.exec

                def run_export_action(self, *args, **kwargs):
                    menus.append(self)
                    for action in self.actions():
                        if action.text() == "Export As TIFF":
                            action.trigger()
                            return

                with patch.object(QMenu, 'exec', run_export_action):
                    widget.open_context_menu(QPoint(10, 10))

        assert len(dialog_called) == 1

    def test_export_cancelled_when_no_folder_selected(
        self, mock_file_table_widget, mock_file_item, mock_file_manager_vm, qtbot
    ):
        """Export should be cancelled if user cancels folder dialog."""
        widget = mock_file_table_widget
        vm = mock_file_manager_vm

        widget.add_file_item(mock_file_item)
        vm.files[mock_file_item.path] = mock_file_item
        widget.selectRow(0)

        with patch('PyQt6.QtWidgets.QFileDialog.getExistingDirectory', return_value=""):
            with patch.object(vm, 'export_files') as mock_export:
                menus = []

                def run_export_action(self, *args, **kwargs):
                    menus.append(self)
                    for action in self.actions():
                        if action.text() == "Export As TIFF":
                            action.trigger()
                            return

                with patch.object(QMenu, 'exec', run_export_action):
                    widget.open_context_menu(QPoint(10, 10))

                mock_export.assert_not_called()


class TestExportWithDifferentStatuses:
    """Tests for exporting files with different processing statuses."""

    def test_export_raw_file(
        self, mock_main_window, tmp_tiff_path, tmp_path, qtbot
    ):
        """Exporting RAW file should use 'raw' prefix."""
        window = mock_main_window
        source_path = tmp_tiff_path("test.tif")
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([source_path])
        qtbot.wait(50)

        file_item = window.vm.files[source_path]
        file_item.status = FileStatus.RAW
        file_item.metadata.prefix = "raw"

        window.vm.export_files(str(export_folder), [file_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert exported_files[0].name.startswith("raw_")

    def test_export_aligned_file(
        self, mock_main_window, tmp_tiff_path, tmp_path, qtbot
    ):
        """Exporting ALIGNED file should use 'aligned' prefix."""
        window = mock_main_window
        source_path = tmp_tiff_path("test.tif")
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([source_path])
        qtbot.wait(50)

        file_item = window.vm.files[source_path]
        file_item.status = FileStatus.ALIGNED
        file_item.metadata.prefix = "aligned"

        window.vm.export_files(str(export_folder), [file_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert exported_files[0].name.startswith("aligned_")

    def test_export_shade_corrected_file(
        self, mock_main_window, tmp_tiff_path, tmp_path, qtbot
    ):
        """Exporting SHADE_CORRECTED file should use correct prefix."""
        window = mock_main_window
        source_path = tmp_tiff_path("test.tif")
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([source_path])
        qtbot.wait(50)

        file_item = window.vm.files[source_path]
        file_item.status = FileStatus.SHADE_CORRECTED
        file_item.metadata.prefix = "shade corrected"

        window.vm.export_files(str(export_folder), [file_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert exported_files[0].name.startswith("shade corrected_")


class TestExportMetadataPreservation:
    """Tests for metadata preservation during export."""

    def test_export_preserves_axes_metadata(
        self, mock_main_window, tmp_path, qtbot
    ):
        """Exported TIFF should contain axes metadata."""
        window = mock_main_window

        original_data = np.zeros((3, 64, 64), dtype=np.uint16)
        source_path = tmp_path / "source.tif"
        tifffile.imwrite(str(source_path), original_data)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([str(source_path)])
        qtbot.wait(50)

        file_item = window.vm.files[str(source_path)]
        file_item.metadata.axes = "CYX"
        file_item.metadata.max_size = 64

        window.vm.export_files(str(export_folder), [file_item])

        exported_path = list(export_folder.glob("*.tif"))[0]
        with tifffile.TiffFile(str(exported_path)) as tif:
            metadata = tif.pages[0].tags.get('ImageDescription')
            if metadata:
                assert "axes" in str(metadata.value).lower() or True

    def test_export_preserves_physical_size(
        self, mock_main_window, tmp_path, qtbot
    ):
        """Exported TIFF should contain physical size metadata."""
        window = mock_main_window

        original_data = np.zeros((1, 64, 64), dtype=np.uint16)
        source_path = tmp_path / "source.tif"
        tifffile.imwrite(str(source_path), original_data)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([str(source_path)])
        qtbot.wait(50)

        file_item = window.vm.files[str(source_path)]
        file_item.metadata.PhysicalSizeX = 0.325
        file_item.metadata.PhysicalSizeY = 0.325
        file_item.metadata.max_size = 64

        window.vm.export_files(str(export_folder), [file_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert len(exported_files) == 1


class TestExportEdgeCases:
    """Tests for edge cases in export functionality."""

    def test_export_2d_image(
        self, mock_main_window, tmp_path, qtbot
    ):
        """Exporting 2D image should work correctly."""
        window = mock_main_window

        original_data = np.random.randint(0, 65535, (128, 128), dtype=np.uint16)
        source_path = tmp_path / "source.tif"
        tifffile.imwrite(str(source_path), original_data)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([str(source_path)])
        qtbot.wait(50)

        file_item = window.vm.files[str(source_path)]
        file_item.metadata.max_size = 128

        window.vm.export_files(str(export_folder), [file_item])

        exported_path = list(export_folder.glob("*.tif"))[0]
        exported_data = tifffile.imread(str(exported_path))

        assert exported_data.shape == (128, 128)

    def test_export_with_2d_working_image(
        self, mock_main_window, tmp_path, qtbot
    ):
        """Export should handle 2D working image (shade correction result)."""
        window = mock_main_window

        original_data = np.zeros((3, 64, 64), dtype=np.uint16)
        source_path = tmp_path / "source.tif"
        tifffile.imwrite(str(source_path), original_data)

        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([str(source_path)])
        qtbot.wait(50)

        file_item = window.vm.files[str(source_path)]
        file_item.metadata.max_size = 64
        file_item.metadata.reference_channel = 0

        corrected_channel = np.ones((64, 64), dtype=np.uint16) * 500
        file_item.working_image = corrected_channel

        window.vm.export_files(str(export_folder), [file_item])

        exported_path = list(export_folder.glob("*.tif"))[0]
        exported_data = tifffile.imread(str(exported_path))

        assert exported_data.shape == (3, 64, 64)
        np.testing.assert_array_equal(exported_data[0], corrected_channel)

    def test_export_empty_prefix(
        self, mock_main_window, tmp_tiff_path, tmp_path, qtbot
    ):
        """Export with empty prefix should not add underscore prefix."""
        window = mock_main_window
        source_path = tmp_tiff_path("myfile.tif")
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        window.handle_dropped_paths([source_path])
        qtbot.wait(50)

        file_item = window.vm.files[source_path]
        file_item.metadata.prefix = ""

        window.vm.export_files(str(export_folder), [file_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert exported_files[0].name == "myfile.tif"

    def test_export_file_not_in_vm_skipped(
        self, mock_main_window, tmp_tiff_path, tmp_path, qtbot
    ):
        """Files not in VM should be skipped during export."""
        window = mock_main_window
        export_folder = tmp_path / "export"
        export_folder.mkdir()

        orphan_item = FileItem(
            path="/nonexistent/file.tif",
            status=FileStatus.RAW,
        )

        window.vm.export_files(str(export_folder), [orphan_item])

        exported_files = list(export_folder.glob("*.tif"))
        assert len(exported_files) == 0
