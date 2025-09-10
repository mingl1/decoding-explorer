from os import error
from typing import Optional

import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal
from pytools import F

from model.file_item import FileItem, MetaData
from utils import get_memory_usage_mb


class MetadataVM(QObject):
    metadata_applied_sig = pyqtSignal(dict)
    update_metadata_view_sig = pyqtSignal(list)
    shading_correction_sig = pyqtSignal(bool)
    align_channels_sig = pyqtSignal(bool)
    inspect_beads_sig = pyqtSignal(FileItem, pd.DataFrame)
    error_sig = pyqtSignal(str)
    statistics_updated = pyqtSignal(dict)
    update_overview_sig = pyqtSignal(pd.DataFrame)

    def __init__(self):
        super().__init__()
        self.selected_files = []
        self.protein_df = pd.DataFrame()

    def update_selected_items(self, metadata_list: list[FileItem]):
        """Display metadata from selected items."""
        self.selected_files = metadata_list
        self.update_metadata_view_sig.emit(metadata_list)

    def apply_metadata(self, metadata_changes: dict):
        res = {}
        use_status_as_prefix = metadata_changes.pop("use_status_as_prefix", False)

        for f in self.selected_files:
            current_changes = metadata_changes.copy()
            if use_status_as_prefix:
                current_changes["prefix"] = f.status.value.lower()

            # Filter out keys that are not in MetaData
            valid_keys = {key for key in current_changes if hasattr(MetaData, key)}
            filtered_changes = {k: current_changes[k] for k in valid_keys}

            res[f.path] = MetaData(**filtered_changes)
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
        self.inspect_beads_sig.emit(self.selected_files[0], self.protein_df)

    def set_protein_files(self, files: list[str]):
        new_protein_df = (
            pd.concat(
                [pd.read_csv(f) if f.endswith(".csv") else pd.read_excel(f) for f in files]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

        cols = new_protein_df.columns.tolist()
        renamed = {}
        for i in range(1, len(cols)):
            renamed[cols[i]] = "cy" + str(i - 1)
        new_protein_df.rename(columns=renamed, inplace=True)

        
        self.protein_df = new_protein_df
        self.update_overview_sig.emit(new_protein_df)