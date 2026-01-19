import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile
from PyQt6.QtWidgets import QApplication

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def tmp_tiff_path(tmp_path):
    """Create a temporary TIFF file for testing."""
    def _create_tiff(filename="test.tif", shape=(1, 512, 512), dtype=np.uint16):
        arr = np.random.randint(0, 65535, shape, dtype=dtype)
        path = tmp_path / filename
        tifffile.imwrite(str(path), arr)
        return str(path)
    return _create_tiff


@pytest.fixture
def tmp_tiff_folder(tmp_path):
    """Create a temporary folder with multiple TIFF files."""
    def _create_folder(num_files=3, filename_prefix="test"):
        folder = tmp_path / "tiffs"
        folder.mkdir()
        for i in range(num_files):
            arr = np.random.randint(0, 65535, (1, 512, 512), dtype=np.uint16)
            tifffile.imwrite(str(folder / f"{filename_prefix}_{i}.tif"), arr)
        return str(folder)
    return _create_folder


@pytest.fixture
def mock_file_item(tmp_tiff_path):
    """Create a FileItem with a temporary TIFF file."""
    from model.file_item import FileItem
    from model.status_enum import FileStatus

    path = tmp_tiff_path()
    item = FileItem(path=path, status=FileStatus.RAW)
    item.shape = (1, 512, 512)
    item.original_shape = (1, 512, 512)
    item.dtype = "uint16"
    return item


@pytest.fixture
def mock_file_items(tmp_tiff_folder):
    """Create multiple FileItems from a temporary folder."""
    from model.file_item import FileItem
    from model.status_enum import FileStatus

    folder_path = tmp_tiff_folder(3)
    items = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".tif", ".tiff")):
            path = os.path.join(folder_path, filename)
            item = FileItem(path=path, status=FileStatus.RAW)
            item.shape = (1, 512, 512)
            item.dtype = "uint16"
            items.append(item)
    return items


@pytest.fixture
def mock_file_manager_vm(qapp):
    """Create a FileManagerVM instance for testing."""
    from viewmodel.file_manager_vm import FileManagerVM
    return FileManagerVM()


@pytest.fixture
def mock_main_window(qapp):
    """Create a MainWindow instance for testing."""
    from view.MainWindow import MainWindow
    window = MainWindow()
    yield window
    window.close()


@pytest.fixture
def mock_file_table_widget(qapp, mock_file_manager_vm):
    """Create a FileTableWidget instance for testing."""
    from view.FileListWidget import FileTableWidget

    def noop_callback(paths):
        pass

    widget = FileTableWidget(
        file_dropped_callback=noop_callback,
        vm=mock_file_manager_vm
    )
    widget.show()
    yield widget
    widget.close()


@pytest.fixture
def mock_metadata_vm(qapp):
    """Create a MetadataVM instance for testing."""
    from viewmodel.metadata_vm import MetadataVM
    return MetadataVM()


@pytest.fixture
def mock_alignment_preview_dialog(qapp):
    """Create a mocked AlignmentPreviewDialog."""
    from unittest.mock import MagicMock, patch

    with patch('viewmodel.file_manager_vm.AlignmentPreviewDialog') as mock_dialog:
        mock_instance = MagicMock()
        mock_instance.exec.return_value = True  # Simulate accepted dialog
        mock_dialog.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_register_thread(qapp):
    """Create a mocked Register thread for alignment tests."""
    from unittest.mock import MagicMock, patch
    from PyQt6.QtCore import QThread, pyqtSignal

    mock_thread = MagicMock()
    mock_thread.progress = MagicMock(spec=QThread)
    mock_thread.error = MagicMock(spec=QThread)
    mock_thread.alignment_complete = MagicMock(spec=QThread)
    mock_thread.run_registration = MagicMock()

    return mock_thread


@pytest.fixture
def mock_bead_thread(qapp):
    """Create a mocked BeadGenerationThread for bead tests."""
    from unittest.mock import MagicMock, patch
    from PyQt6.QtCore import QThread, pyqtSignal

    mock_thread = MagicMock()
    mock_thread.bead_generated = MagicMock(spec=QThread)
    mock_thread.progress = MagicMock(spec=QThread)
    mock_thread.start = MagicMock()

    return mock_thread


@pytest.fixture
def mock_tifffile():
    """Mock tifffile functions for unit tests."""
    mock_shape = (1, 512, 512)
    mock_dtype = np.uint16
    mock_array = np.zeros(mock_shape, dtype=mock_dtype)

    with patch('tifffile.TiffFile') as mock_tiff, \
         patch('tifffile.imread', return_value=mock_array), \
         patch('tifffile.memmap', return_value=mock_array):

        mock_tiff.return_value.__enter__.return_value.pages[0].shape = mock_shape[1:]
        mock_tiff.return_value.__enter__.return_value.dtype = mock_dtype
        mock_tiff.return_value.__enter__.return_value.pages.__len__ = lambda self: mock_shape[0]

        yield {
            'shape': mock_shape,
            'dtype': mock_dtype,
            'array': mock_array
        }


@pytest.fixture
def mock_qfiledialog(qapp):
    """Mock QFileDialog for testing file loading."""
    from unittest.mock import patch

    with patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName') as mock_open, \
         patch('PyQt6.QtWidgets.QFileDialog.getExistingDirectory') as mock_dir, \
         patch('PyQt6.QtWidgets.QFileDialog.getSaveFileName') as mock_save:

        mock_open.return_value = ("", "")  # Default: no file selected
        mock_dir.return_value = ""  # Default: no directory selected
        mock_save.return_value = ("", "")  # Default: no file selected

        yield {
            'getOpenFileName': mock_open,
            'getExistingDirectory': mock_dir,
            'getSaveFileName': mock_save
        }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        'max_size': 512,
        'reference_channel': 0,
        'num_tiles': 1,
        'overlap': 0.1,
        'prefix': '',
    }


class SignalRecorder:
    """Helper class to record Qt signal emissions."""

    def __init__(self):
        self.signals = []

    def connect(self, signal):
        signal.connect(self._record)

    def disconnect(self, signal):
        signal.disconnect(self._record)

    def _record(self, *args, **kwargs):
        self.signals.append((args, kwargs))

    def clear(self):
        self.signals = []

    def get_last_args(self):
        if self.signals:
            return self.signals[-1]
        return None

    def get_call_count(self):
        return len(self.signals)


@pytest.fixture
def signal_recorder():
    """Create a SignalRecorder for tracking signal emissions."""
    return SignalRecorder()
