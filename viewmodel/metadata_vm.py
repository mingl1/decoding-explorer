import logging

import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal

from model.file_item import FileItem


class MetadataVM(QObject):
    metadata_applied_sig = pyqtSignal(dict)
    metadata_corrected_sig = pyqtSignal(dict)
    update_metadata_view_sig = pyqtSignal(list)
    align_channels_sig = pyqtSignal(bool)
    inspect_beads_sig = pyqtSignal(FileItem, pd.DataFrame)
    error_sig = pyqtSignal(str)
    statistics_updated = pyqtSignal(dict)
    update_overview_sig = pyqtSignal(pd.DataFrame)
    file_shape_update_sig = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.selected_files = []
        self.protein_df = pd.DataFrame()

    def update_selected_items(self, metadata_list: list[FileItem]):
        """Display metadata from selected items."""
        self.selected_files = metadata_list
        self.update_metadata_view_sig.emit(metadata_list)

    def apply_metadata(self, metadata_changes: dict):
        logging.info(f"Applying metadata changes: {metadata_changes}")
        res = metadata_changes
        logging.info(f"Emitting metadata_applied_sig with: {res}")
        self.metadata_applied_sig.emit(res)

    def update_corrected_metadata(self, key: str, value):
        self.metadata_corrected_sig.emit({key: value})

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
                [
                    pd.read_csv(f) if f.endswith(".csv") else pd.read_excel(f)
                    for f in files
                ]
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
