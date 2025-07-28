from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from model.file_item import FileItem
from viewmodel.metadata_vm import MetadataVM


class MetadataView(QWidget):
    export_all_sig = pyqtSignal(str, list)
    generate_beads_sig = pyqtSignal()

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
        metadata_title = QLabel("Metadata")
        metadata_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(1)
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
        self.export_all_btn = QPushButton("Export All")
        self.export_all_btn.clicked.connect(self.export_all)
        self.generate_beads_btn = QPushButton("Generate Beads")
        self.generate_beads_btn.clicked.connect(self.generate_beads_sig.emit)

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
        self.form_layout.addRow(self.apply_shading_correction_btn)
        self.form_layout.addRow(self.align_channels_btn)
        self.form_layout.addRow(self.export_all_btn)
        self.form_layout.addRow(self.generate_beads_btn)

        self.vm.update_metadata_view_sig.connect(self.update_metadata)

        layout = QVBoxLayout()
        layout.addLayout(self.form_layout)
        self.setLayout(layout)

    def update_metadata(self, metadata_list: list[FileItem]):
        """Display metadata from selected items."""
        print(f"Setting metadata for {len(metadata_list)} items")
        print(metadata_list)
        if self.all_same_metadata(metadata_list):
            self.prefix_input.setText(metadata_list[0].metadata.prefix)
            self.axes_input.setText(metadata_list[0].metadata.axes)
            self.unit_input.setText(metadata_list[0].metadata.unit)
            self.size_x_input.setText(str(metadata_list[0].metadata.PhysicalSizeX))
            self.size_y_input.setText(str(metadata_list[0].metadata.PhysicalSizeY))
            self.channel_input.setText(str(metadata_list[0].metadata.reference_channel))
            self.max_size_input.setText(str(metadata_list[0].metadata.max_size))
        else:
            self.prefix_input.setText("")
            self.axes_input.setText("")
            self.unit_input.setText("")
            self.size_x_input.setText("")
            self.size_y_input.setText("")
            self.channel_input.setText("")
            self.max_size_input.setText("")

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
            "reference_channel": self.channel_input.text(),
            "max_size": self.max_size_input.text(),
        }

    def export_all(self):
        folder = QFileDialog.getExistingDirectory()
        if folder:
            self.export_all_sig.emit(folder, self.vm.selected_files)
