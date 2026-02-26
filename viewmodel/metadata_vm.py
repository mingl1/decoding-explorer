import logging
from typing import Optional

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

    def _read_csv_with_fallback(self, file_path: str) -> pd.DataFrame:
        encodings = [
            "utf-8",
            "utf-8-sig",
            "utf-16",
            "utf-16le",
            "utf-16be",
            "latin1",
        ]
        last_error: Optional[Exception] = None

        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except (UnicodeError, pd.errors.ParserError) as exc:
                last_error = exc

        try:
            return pd.read_csv(
                file_path, encoding="latin1", sep=None, engine="python"
            )
        except Exception as exc:
            if last_error is None:
                last_error = exc

        try:
            return pd.read_excel(file_path)
        except Exception as exc:
            message = f"Unable to read protein key file '{file_path}': {exc}"
            if last_error is not None:
                message = f"{message} (last CSV error: {last_error})"
            raise ValueError(message) from exc

    def _read_protein_file(self, file_path: str) -> pd.DataFrame:
        if file_path.lower().endswith(".csv"):
            return self._read_csv_with_fallback(file_path)
        return pd.read_excel(file_path)

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
        # elif len(self.selected_files) > 1:
        #     self.error_sig.emit("You should select the reference file only.")
        for i in range(len(self.selected_files)):
            if self.selected_files[i].beads is not None:
                self.inspect_beads_sig.emit(self.selected_files[i], self.protein_df)
                return
        self.error_sig.emit("No bead data found in the selected files.")

    def set_protein_files(self, files: list[str]):
        if len(files) == 0:
            self.protein_df = pd.DataFrame()
            self.update_overview_sig.emit(self.protein_df)
            return

        try:
            new_protein_df = (
                pd.concat([self._read_protein_file(f) for f in files])
                .drop_duplicates()
                .reset_index(drop=True)
            )
        except Exception as exc:
            logging.exception("Failed to load protein key files")
            self.error_sig.emit(f"Failed to load protein key files: {exc}")
            return

        cols = new_protein_df.columns.tolist()
        renamed = {}
        for i in range(1, len(cols)):
            renamed[cols[i]] = "cy" + str(i - 1)
        new_protein_df.rename(columns=renamed, inplace=True)

        self.protein_df = new_protein_df
        self.update_overview_sig.emit(new_protein_df)
