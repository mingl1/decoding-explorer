from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from model.file_item import FileItem


class ExportSelectionDialog(QDialog):
    def __init__(
        self,
        tiff_options: list[tuple[str, FileItem, bool]],
        beads_enabled: bool,
        beads_checked: bool,
        beads_format: str = "csv",
        excluded_beads_count: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self._tiff_checkboxes: list[tuple[QCheckBox, FileItem]] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select data to export:"))

        for label, file_item, checked in tiff_options:
            checkbox = QCheckBox(label)
            checkbox.setChecked(bool(checked))
            self._tiff_checkboxes.append((checkbox, file_item))
            layout.addWidget(checkbox)

        self.export_beads_checkbox = QCheckBox("Export beads (result)")
        self.export_beads_checkbox.setEnabled(bool(beads_enabled))
        self.export_beads_checkbox.setChecked(bool(beads_checked and beads_enabled))
        layout.addWidget(self.export_beads_checkbox)

        if beads_enabled:
            self.excluded_label = QLabel(f"Excluded by crop: {excluded_beads_count} beads")
            self.excluded_label.setStyleSheet("color: gray; margin-left: 20px;")
            layout.addWidget(self.excluded_label)


        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Beads format:"))
        self.beads_format_combo = QComboBox()
        self.beads_format_combo.addItem("CSV", "csv")
        self.beads_format_combo.addItem("XLSX", "xlsx")
        normalized_format = str(beads_format).strip().lower()
        if normalized_format == "xlsx":
            self.beads_format_combo.setCurrentIndex(1)
        else:
            self.beads_format_combo.setCurrentIndex(0)
        format_row.addWidget(self.beads_format_combo)
        format_row.addStretch(1)
        layout.addLayout(format_row)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.accept)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(export_btn)
        layout.addLayout(button_row)

        self.export_beads_checkbox.stateChanged.connect(self._sync_beads_format_state)
        self._sync_beads_format_state()

    def _sync_beads_format_state(self):
        self.beads_format_combo.setEnabled(self.export_beads_checkbox.isChecked())

    def get_selected_tiff_files(self) -> list[FileItem]:
        selected: list[FileItem] = []
        for checkbox, file_item in self._tiff_checkboxes:
            if checkbox.isChecked():
                selected.append(file_item)
        return selected

    def should_export_beads(self) -> bool:
        return bool(self.export_beads_checkbox.isChecked())

    def get_beads_format(self) -> str:
        value: Optional[str] = self.beads_format_combo.currentData()
        if value not in ("csv", "xlsx"):
            return "csv"
        return value
