import logging
import re
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
        sample = b""
        try:
            with open(file_path, "rb") as handle:
                sample = handle.read(4096)
        except OSError:
            sample = b""

        if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
            encodings = ["utf-16", "utf-16le", "utf-16be", "utf-8", "utf-8-sig", "latin1"]
        elif b"\x00" in sample:
            encodings = ["utf-16le", "utf-16be", "utf-16", "utf-8", "utf-8-sig", "latin1"]

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

    def _protein_column_score(self, column_name: str, series: pd.Series) -> int:
        normalized = str(column_name).strip().lower()
        normalized = normalized.replace("_", " ").replace("-", " ")
        score = 0
        exact_aliases = {
            "protein name",
            "protein",
            "gene",
            "gene name",
            "target",
            "marker",
            "name",
        }
        if normalized in exact_aliases:
            score += 6
        elif (
            "protein" in normalized
            or "gene" in normalized
            or "target" in normalized
            or "marker" in normalized
            or normalized.endswith("name")
        ):
            score += 4
        non_empty = series.dropna()
        if len(non_empty) == 0:
            return score
        numeric_values = pd.to_numeric(non_empty, errors="coerce")
        numeric_ratio = float(numeric_values.notna().sum()) / float(len(non_empty))
        if numeric_ratio <= 0.2:
            score += 2
        return score

    def _is_numeric_cycle_column(self, series: pd.Series) -> bool:
        non_empty = series.dropna()
        if len(non_empty) == 0:
            return False
        numeric_values = pd.to_numeric(non_empty, errors="coerce")
        numeric_ratio = float(numeric_values.notna().sum()) / float(len(non_empty))
        return numeric_ratio >= 0.8

    def _cycle_column_sort_key(self, column_name: str, index: int):
        match = re.search(r"(\d+)", str(column_name))
        if match is not None:
            return (0, int(match.group(1)), index)
        return (1, index, index)

    def _normalize_protein_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        working_df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
        if working_df.empty:
            return pd.DataFrame(columns=["Protein name"])

        columns = working_df.columns.tolist()
        protein_column = max(
            columns,
            key=lambda col: self._protein_column_score(str(col), working_df[col]),
        )

        cycle_candidates = []
        fallback_text_columns = []
        for idx, col in enumerate(columns):
            if col == protein_column:
                continue
            if self._is_numeric_cycle_column(working_df[col]):
                cycle_candidates.append((idx, col))
            else:
                fallback_text_columns.append(col)

        sorted_cycle_columns = [
            col
            for idx, col in sorted(
                cycle_candidates,
                key=lambda item: self._cycle_column_sort_key(str(item[1]), item[0]),
            )
        ]

        protein_series = working_df[protein_column].copy()
        for col in fallback_text_columns:
            protein_missing = protein_series.isna() | (
                protein_series.astype(str).str.strip() == ""
            )
            candidate_series = working_df[col]
            candidate_present = candidate_series.notna() & (
                candidate_series.astype(str).str.strip() != ""
            )
            fill_mask = protein_missing & candidate_present
            protein_series.loc[fill_mask] = candidate_series.loc[fill_mask]

        normalized_df = pd.DataFrame({"Protein name": protein_series})
        for i, col in enumerate(sorted_cycle_columns):
            numeric_values = pd.to_numeric(working_df[col], errors="coerce")
            if numeric_values.notna().all():
                normalized_df[f"cy{i}"] = numeric_values.astype(int)
            else:
                normalized_df[f"cy{i}"] = numeric_values.astype("Int64")

        normalized_df["Protein name"] = normalized_df["Protein name"].astype("string")
        normalized_df["Protein name"] = normalized_df["Protein name"].str.strip()
        normalized_df["Protein name"] = normalized_df["Protein name"].replace("", pd.NA)

        cycle_cols = [col for col in normalized_df.columns if col.startswith("cy")]
        protein_present = normalized_df["Protein name"].notna()
        if len(cycle_cols) > 0:
            any_cycle_present = normalized_df[cycle_cols].notna().any(axis=1)
            keep_mask = protein_present | any_cycle_present
        else:
            keep_mask = protein_present

        return normalized_df.loc[keep_mask].reset_index(drop=True)

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
                pd.concat(
                    [
                        self._normalize_protein_dataframe(self._read_protein_file(f))
                        for f in files
                    ],
                    ignore_index=True,
                )
                .drop_duplicates()
                .reset_index(drop=True)
            )
        except Exception as exc:
            logging.exception("Failed to load protein key files")
            self.error_sig.emit(f"Failed to load protein key files: {exc}")
            return

        print("Uploaded protein dataframe:")
        print(new_protein_df)

        self.protein_df = new_protein_df
        self.update_overview_sig.emit(new_protein_df)
