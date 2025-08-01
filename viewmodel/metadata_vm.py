from os import error

from PyQt6.QtCore import QObject, pyqtSignal
from pytools import F

from model.file_item import FileItem, MetaData
from utils import get_memory_usage_mb


class MetadataVM(QObject):
    metadata_applied_sig = pyqtSignal(dict)
    update_metadata_view_sig = pyqtSignal(list)
    shading_correction_sig = pyqtSignal(bool)
    align_channels_sig = pyqtSignal(bool)
    inspect_beads_sig = pyqtSignal(FileItem)
    error_sig = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.selected_files = []

    def update_selected_items(self, metadata_list: list[FileItem]):
        """Display metadata from selected items."""
        self.selected_files = metadata_list
        self.update_metadata_view_sig.emit(metadata_list)

    def apply_metadata(self, metadata_changes: dict):
        res = {}
        for f in self.selected_files:
            res[f.path] = MetaData(**metadata_changes)
        self.metadata_applied_sig.emit(res)

    def apply_shading_correction(self):
        self.shading_correction_sig.emit(True)

    def align_channels(self):
        self.align_channels_sig.emit(True)

    def inspect_beads(self):
        print(f"Inspecting beads for {len(self.selected_files)} items")
        if len(self.selected_files) == 0:
            self.error_sig.emit("No files selected.")
            return
        elif len(self.selected_files) > 1:
            self.error_sig.emit("You should select the reference file only.")
        self.inspect_beads_sig.emit(self.selected_files[0])
