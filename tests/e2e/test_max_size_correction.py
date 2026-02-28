import os
import numpy as np
import pytest
import tifffile
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import Qt
from unittest.mock import patch
from view.main_window import MainWindow


@pytest.fixture
def small_tiff_folder(tmp_path):
    """Generates a folder with small dummy TIFF files (10x10)."""
    folder = tmp_path / "small_images"
    folder.mkdir()

    for i in range(2):
        data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        file_path = folder / f"small_{i}.tif"
        tifffile.imwrite(file_path, data)

    return folder


def test_max_size_correction_persists_on_reselection(qtbot, small_tiff_folder):
    """
    Bug Test: When user enters max_size too high, the metadata view should show
    the corrected value even after reselecting the row.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(small_tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 2
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    base_max_size = int(window.metadata_view.max_size_input.text())
    print(f"Base max_size from loaded files: {base_max_size}")

    window.metadata_view.max_size_input.clear()
    qtbot.keyClicks(window.metadata_view.max_size_input, "2048")

    qtbot.wait(500)

    corrected_max = int(window.metadata_view.max_size_input.text())
    print(f"After update - metadata_view shows: {corrected_max}")
    assert corrected_max < 2048, "max_size should be corrected to fit image dimensions"

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    after_reselect = int(window.metadata_view.max_size_input.text())
    print(f"After reselect - metadata_view shows: {after_reselect}")

    assert after_reselect == corrected_max, (
        f"After reselection, metadata_view should show corrected value {corrected_max}, "
        f"but got {after_reselect}"
    )
