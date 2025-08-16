import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from model.file_item import FileItem
from viewmodel.metadata_vm import MetadataVM


class MetadataView(QWidget):
    export_all_sig = pyqtSignal(str, list)
    generate_beads_sig = pyqtSignal()
    protein_files_uploaded = pyqtSignal(list)

    def __init__(self, parent, vm: MetadataVM):
        super().__init__(parent)
        self.form_layout = QFormLayout()
        self.vm = vm
        # Example fields
        self.prefix_input = QLineEdit()
        self.channel_input = QLineEdit()
        self.axes_input = QLineEdit("")
        self.unit_input = QLineEdit("")
        self.size_x_input = QLineEdit()
        self.size_y_input = QLineEdit()
        self.max_size_input = QLineEdit()
        self.num_tiles_input = QLineEdit()
        self.overlap_input = QLineEdit()
        metadata_title = QLabel("Metadata")
        metadata_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(1)
        align_arrays_title = QLabel("Align Arrays")
        align_arrays_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        align_arrays_separator = QFrame()
        align_arrays_separator.setFrameShape(QFrame.Shape.HLine)
        align_arrays_separator.setFrameShadow(QFrame.Shadow.Sunken)
        align_arrays_separator.setLineWidth(1)
        self.apply_btn = QPushButton("Update Metadata")
        self.apply_btn.clicked.connect(
            lambda: self.vm.apply_metadata(self.get_metadata_changes())
        )
        self.apply_shading_correction_btn = QPushButton("Apply Shading Correction")
        self.apply_shading_correction_btn.clicked.connect(
            self.vm.apply_shading_correction
        )
        self.align_channels_btn = QPushButton("Align to Reference")
        self.align_channels_btn.clicked.connect(self.vm.align_channels)

        self.upload_protein_key_btn = QPushButton("Upload Protein/Gene Key Files")
        self.upload_protein_key_btn.clicked.connect(self.upload_protein_key_files)
        self.protein_key_files_label = QLabel("No files uploaded.")
        self.generate_beads_btn = QPushButton("Generate and Export Beads")
        self.generate_beads_btn.clicked.connect(self.generate_beads_sig.emit)
        self.inspect_beads_btn = QPushButton("Inspect Beads")
        self.inspect_beads_btn.clicked.connect(self.vm.inspect_beads)
        bead_generation_title = QLabel("Bead Generation")
        bead_generation_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        bead_generation_separator = QFrame()
        bead_generation_separator.setFrameShape(QFrame.Shape.HLine)
        bead_generation_separator.setFrameShadow(QFrame.Shadow.Sunken)
        bead_generation_separator.setLineWidth(1)
        self.form_layout.addRow(metadata_title)
        self.form_layout.addRow(separator)
        self.form_layout.addRow("File Prefix:", self.prefix_input)
        self.form_layout.addRow("Axes (e.g. CYX):", self.axes_input)
        self.form_layout.addRow("Unit (e.g. um):", self.unit_input)
        self.form_layout.addRow("PhysicalSizeX:", self.size_x_input)
        self.form_layout.addRow("PhysicalSizeY:", self.size_y_input)
        self.form_layout.addRow("Alignment Channel:", self.channel_input)
        self.form_layout.addRow("Max Size:", self.max_size_input)
        self.form_layout.addRow(self.apply_btn)
        self.form_layout.addRow(align_arrays_title)
        self.form_layout.addRow(align_arrays_separator)
        self.form_layout.addRow("Num Tiles:", self.num_tiles_input)
        self.form_layout.addRow("Overlap:", self.overlap_input)
        self.form_layout.addRow(self.apply_shading_correction_btn)

        self.form_layout.addRow(self.align_channels_btn)
        self.form_layout.addRow(bead_generation_title)
        self.form_layout.addRow(bead_generation_separator)
        self.form_layout.addRow(self.upload_protein_key_btn)
        self.form_layout.addRow(self.protein_key_files_label)
        self.form_layout.addRow(self.generate_beads_btn)
        self.form_layout.addRow(self.inspect_beads_btn)
        self.stats_title = QLabel("Statistics")
        self.stats_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.stats_separator = QFrame()
        self.stats_separator.setFrameShape(QFrame.Shape.HLine)
        self.stats_separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.total_beads_label = QLabel("Total Beads: N/A")
        self.mean_rows_label = QLabel("Mean Rows per Protein: N/A")
        self.error_label = QLabel("Error: N/A")
        self.counts_table = QTableWidget()
        self.counts_table.setColumnCount(2)
        self.counts_table.setHorizontalHeaderLabels(["Protein Name", "Count"])

        self.form_layout.addRow(self.stats_title)
        self.form_layout.addRow(self.stats_separator)
        self.form_layout.addRow(self.total_beads_label)
        self.form_layout.addRow(self.mean_rows_label)
        self.form_layout.addRow(self.error_label)
        self.form_layout.addRow(self.counts_table)
        self.vm.update_metadata_view_sig.connect(self.update_metadata)

        layout = QVBoxLayout()
        layout.addLayout(self.form_layout)
        self.setLayout(layout)

    def upload_protein_key_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Protein/Gene Key files",
            "",
            "Excel and CSV files (*.csv *.xlsx)",
        )
        if files:
            self.protein_key_files_label.setText(
                "\n".join(os.path.basename(f) for f in files)
            )
            self.protein_files_uploaded.emit(files)

    def update_metadata(self, metadata_list: list[FileItem]):
        """Display metadata from selected items."""
        print(f"Setting metadata for {len(metadata_list)} items")
        if self.all_same_metadata(metadata_list):
            self.prefix_input.setText(metadata_list[0].metadata.prefix)
            self.axes_input.setText(metadata_list[0].metadata.axes)
            self.unit_input.setText(metadata_list[0].metadata.unit)
            self.size_x_input.setText(str(metadata_list[0].metadata.PhysicalSizeX))
            self.size_y_input.setText(str(metadata_list[0].metadata.PhysicalSizeY))
            self.channel_input.setText(str(metadata_list[0].metadata.reference_channel))
            self.max_size_input.setText(str(metadata_list[0].metadata.max_size))
            self.num_tiles_input.setText(str(metadata_list[0].metadata.num_tiles))
            self.overlap_input.setText(str(metadata_list[0].metadata.overlap))
        else:
            self.prefix_input.setText("")
            self.axes_input.setText("")
            self.unit_input.setText("")
            self.size_x_input.setText("")
            self.size_y_input.setText("")
            self.channel_input.setText("")
            self.max_size_input.setText("")
            self.num_tiles_input.setText("")
            self.overlap_input.setText("")

    def all_same_metadata(self, metadata_list: list[FileItem]) -> bool:
        if not metadata_list:
            return False
        first = metadata_list[0].metadata
        for item in metadata_list[1:]:
            if item.metadata != first:
                return False
        return True

    def get_metadata_changes(self):
        return {
            "prefix": self.prefix_input.text(),
            "axes": self.axes_input.text(),
            "unit": self.unit_input.text(),
            "PhysicalSizeX": float(self.size_x_input.text()),
            "PhysicalSizeY": float(self.size_y_input.text()),
            "reference_channel": int(self.channel_input.text()),
            "max_size": int(self.max_size_input.text()),
            "num_tiles": int(self.num_tiles_input.text()),
            "overlap": int(self.overlap_input.text()),
        }

    def update_statistics(self, stats: dict):
        self.total_beads_label.setText(
            f"Total Beads: {stats.get('total_beads', 'N/A')}"
        )
        self.mean_rows_label.setText(
            f"Mean Rows per Protein: {stats.get('mean_rows', 'N/A'):.2f}"
        )
        self.error_label.setText(f"Error: {stats.get('error', 'N/A'):.4f}%")

        counts_table_data = stats.get("counts_table")
        if counts_table_data is not None:
            self.counts_table.setRowCount(len(counts_table_data))
            for i, row in counts_table_data.iterrows():
                self.counts_table.setItem(
                    i, 0, QTableWidgetItem(str(row["Protein name"]))
                )
                self.counts_table.setItem(i, 1, QTableWidgetItem(str(row["row_count"])))

    def export_all(self):
        folder = QFileDialog.getExistingDirectory()
        if folder:
            self.export_all_sig.emit(folder, self.vm.selected_files)
