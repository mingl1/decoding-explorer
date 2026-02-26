import os
import shutil
import numpy as np
import pytest
import tifffile
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import Qt
from unittest.mock import patch
from view.main_window import MainWindow


@pytest.fixture
def tiff_folder(tmp_path):
    """Generates a folder with larger dummy TIFF files (100x100)."""
    folder = tmp_path / "test_images"
    folder.mkdir()

    for i in range(3):
        data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        file_path = folder / f"image_{i}.tif"
        tifffile.imwrite(file_path, data)

    return folder


def test_metadata_update_comprehensive(qtbot, tiff_folder):
    """E2E Test: Update multiple metadata fields and verify propagation."""
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    base_max_size = int(window.metadata_view.max_size_input.text())
    base_num_tiles = int(window.metadata_view.num_tiles_input.text())
    base_overlap = int(window.metadata_view.overlap_input.text())
    base_channel = int(window.metadata_view.channel_input.text())

    new_values = {
        "axes": "CYX",
        "unit": "um",
        "size_y": "0.789",
        "channel": str(base_channel + 1 if base_channel < 5 else 1),
        "max_size": "50",
        "num_tiles": str(base_num_tiles + 1 if base_num_tiles < 10 else 3),
        "overlap": str(base_overlap + 5 if base_overlap < 20 else 8),
    }

    window.metadata_view.axes_input.clear()
    window.metadata_view.unit_input.clear()
    window.metadata_view.size_y_input.clear()
    window.metadata_view.channel_input.clear()
    window.metadata_view.max_size_input.clear()
    window.metadata_view.num_tiles_input.clear()
    window.metadata_view.overlap_input.clear()

    qtbot.keyClicks(window.metadata_view.axes_input, new_values["axes"])
    qtbot.keyClicks(window.metadata_view.unit_input, new_values["unit"])
    qtbot.keyClicks(window.metadata_view.size_y_input, new_values["size_y"])
    qtbot.keyClicks(window.metadata_view.channel_input, new_values["channel"])
    qtbot.keyClicks(window.metadata_view.max_size_input, new_values["max_size"])
    qtbot.keyClicks(window.metadata_view.num_tiles_input, new_values["num_tiles"])
    qtbot.keyClicks(window.metadata_view.overlap_input, new_values["overlap"])

    qtbot.mouseClick(window.metadata_view.apply_btn, Qt.MouseButton.LeftButton)
    qtbot.wait(500)

    for row in range(window.file_table_widget.rowCount()):
        item = window.file_table_widget.item(row, 0)
        file_item = item.data(Qt.ItemDataRole.UserRole)
        updated_item = window.vm.files.get(file_item.path)
        if updated_item:
            item.setData(Qt.ItemDataRole.UserRole, updated_item)

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    assert window.metadata_view.axes_input.text() == new_values["axes"]
    assert window.metadata_view.unit_input.text() == new_values["unit"]
    assert float(window.metadata_view.size_y_input.text()) == float(new_values["size_y"])
    assert int(window.metadata_view.channel_input.text()) == int(new_values["channel"])
    assert int(window.metadata_view.max_size_input.text()) == int(new_values["max_size"])
    assert int(window.metadata_view.num_tiles_input.text()) == int(new_values["num_tiles"])
    assert int(window.metadata_view.overlap_input.text()) == int(new_values["overlap"])

    window.file_table_widget.selectRow(1)
    qtbot.wait(200)
    assert window.metadata_view.axes_input.text() == new_values["axes"]
