from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class MetadataView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.form_layout = QFormLayout()

        # Example fields
        self.channel_input = QLineEdit()
        self.axes_input = QLineEdit("CYX")
        self.unit_input = QLineEdit("um")
        self.size_x_input = QLineEdit()
        self.size_y_input = QLineEdit()

        self.form_layout.addRow("Axes (e.g. CYX):", self.axes_input)
        self.form_layout.addRow("Unit (e.g. um):", self.unit_input)
        self.form_layout.addRow("PhysicalSizeX:", self.size_x_input)
        self.form_layout.addRow("PhysicalSizeY:", self.size_y_input)

        self.form_layout.addRow("Reference Channel:", self.channel_input)
        self.apply_btn = QPushButton("Apply to Selected")
        self.form_layout.addRow(self.apply_btn)

        layout = QVBoxLayout()
        layout.addLayout(self.form_layout)
        self.setLayout(layout)

    def set_metadata(self, metadata_list):
        """Display metadata from selected items."""

        def mixed_or_common(key):
            values = [md.get(key, "") for md in metadata_list]
            return (
                values[0]
                if all(v == values[0] for v in values)
                else "— multiple values —"
            )

        self.channel_input.setText(mixed_or_common("reference_channel"))

    def get_metadata_changes(self):
        return {
            "axes": self.axes_input.text(),
            "unit": self.unit_input.text(),
            "PhysicalSizeX": float(self.size_x_input.text()),
            "PhysicalSizeY": float(self.size_y_input.text()),
            "reference_channel": self.channel_input.text(),
        }
