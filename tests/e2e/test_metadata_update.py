
import os
import shutil
import numpy as np
import pytest
import tifffile
from PyQt6.QtWidgets import QFileDialog, QAbstractItemView
from PyQt6.QtCore import Qt
from unittest.mock import patch
from view.main_window import MainWindow

# --- TDD Phase 1: Red (The Test) ---

@pytest.fixture
def tiff_folder(tmp_path):
    """Generates a folder with multiple dummy TIFF files."""
    folder = tmp_path / "test_images"
    folder.mkdir()
    
    # Create 3 dummy tiffs
    for i in range(3):
        # Create a simple 10x10 image
        data = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        file_path = folder / f"image_{i}.tif"
        tifffile.imwrite(file_path, data)
        
    return folder

def test_metadata_update_propagates_to_files(qtbot, tiff_folder):
    """
    E2E Test:
    1. Upload folder with tiffs.
    2. Select all files.
    3. Update metadata (prefix and PhysicalSizeX).
    4. Metadata auto-updates.
    5. Verify metadata updates are reflected when selecting individual files.
    """
    # Initialize the application
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)
    
    # 1. Simulate User Uploading Folder
    # We mock the dialog to return our test folder path, but trigger the actual UI button
    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)
    
    # Wait for files to be loaded into the table
    # We expect 3 rows
    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)
    
    # 2. Select all uploaded tiffs
    # We can programmatically set selection to simulate user selecting all
    window.file_table_widget.selectAll()
    
    # Ensure metadata view is visible and populated (wait for potential signals)
    qtbot.wait(500) 
    
    # Uncheck "use status as prefix" to allow manual prefix editing
    window.metadata_view.prefix_checkbox.setChecked(False)
    qtbot.wait(100)
    
    # 3. Change Metadata in Sideview
    # Change Prefix
    new_prefix = "TEST_PREFIX"
    window.metadata_view.prefix_input.clear()
    qtbot.keyClicks(window.metadata_view.prefix_input, new_prefix)
    
    # Change PhysicalSizeX
    new_size_x = "0.123"
    window.metadata_view.size_x_input.clear()
    qtbot.keyClicks(window.metadata_view.size_x_input, new_size_x)
    
    # Wait for metadata propagation
    qtbot.wait(500)
    
    # 5. Verify Metadata Update
    # Deselect all first
    window.file_table_widget.clearSelection()
    
    # Select the first file
    window.file_table_widget.selectRow(0)
    
    # Wait for metadata view to update from selection
    qtbot.wait(200)
    
    # Assertion: The metadata view should show the new values
    assert window.metadata_view.prefix_input.text() == new_prefix
    assert float(window.metadata_view.size_x_input.text()) == float(new_size_x)
    
    # Select the second file to ensure it applied to all
    window.file_table_widget.selectRow(1)
    qtbot.wait(200)
    assert window.metadata_view.prefix_input.text() == new_prefix
    assert float(window.metadata_view.size_x_input.text()) == float(new_size_x)
