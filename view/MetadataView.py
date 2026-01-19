import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
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
    upload_beads_sig = pyqtSignal()

    def __init__(self, parent, vm: MetadataVM):
        super().__init__(parent)
        self.form_layout = QFormLayout()
        self.form_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        self.form_layout.setHorizontalSpacing(10)
        self.form_layout.setVerticalSpacing(5)
        self.vm = vm
        # Example fields
        self.prefix_input = QLineEdit()
        self.prefix_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.prefix_input.textChanged.connect(self._on_input_changed)
        self.prefix_checkbox = QCheckBox("Use status as prefix")
        self.prefix_checkbox.stateChanged.connect(self.on_prefix_checkbox_changed)

        self.channel_input = QLineEdit()
        self.channel_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.channel_input.textChanged.connect(self._on_input_changed)
        self.axes_input = QLineEdit("")
        self.axes_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.axes_input.textChanged.connect(self._on_input_changed)
        self.unit_input = QLineEdit("")
        self.unit_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.unit_input.textChanged.connect(self._on_input_changed)
        self.size_x_input = QLineEdit()
        self.size_x_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.size_x_input.textChanged.connect(self._on_input_changed)
        self.size_y_input = QLineEdit()
        self.size_y_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.size_y_input.textChanged.connect(self._on_input_changed)
        self.max_size_input = QLineEdit()
        self.max_size_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.max_size_input.textChanged.connect(self._on_input_changed)
        self.num_tiles_input = QLineEdit()
        self.num_tiles_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.num_tiles_input.textChanged.connect(self._on_input_changed)
        self.overlap_input = QLineEdit()
        self.overlap_input.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.overlap_input.textChanged.connect(self._on_input_changed)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(70, 100)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.threshold_slider.setFixedWidth(80)
        self.threshold_value_label = QLabel("0.70")

        self.threshold_container = QWidget()
        threshold_layout = QHBoxLayout()
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.setSpacing(5)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        threshold_layout.addStretch(1)
        self.threshold_container.setLayout(threshold_layout)

        self._section_widgets = {}
        self._section_collapsed = {}

        self._create_section_widgets()
        self._create_ui_elements()

    def _create_section_widgets(self):
        self._section_widgets = {
            "metadata": [],
            "align_arrays": [],
            "bead_generation": [],
            "statistics": [],
        }
        self._section_collapsed = {
            "metadata": False,
            "align_arrays": False,
            "bead_generation": False,
            "statistics": False,
        }

    def _toggle_section(self, section_name):
        if section_name not in self._section_widgets:
            return
        self._section_collapsed[section_name] = not self._section_collapsed[
            section_name
        ]
        collapsed = self._section_collapsed[section_name]
        for widget in self._section_widgets[section_name]:
            widget.setVisible(not collapsed)

    def collapse_processing_sections(self):
        for section in ["metadata", "align_arrays", "bead_generation"]:
            self._section_collapsed[section] = True
            for widget in self._section_widgets[section]:
                widget.setVisible(False)

    def _on_threshold_changed(self, value):
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        self._on_input_changed()

    def _create_ui_elements(self):
        def make_section_title(text, section_name):
            label = QLabel(text)
            label.setStyleSheet("font-weight: bold; font-size: 16px;")
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            label.mousePressEvent = lambda event, s=section_name: self._toggle_section(
                s
            )
            return label

        def make_separator():
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFrameShadow(QFrame.Shadow.Sunken)
            sep.setLineWidth(1)
            sep.setFixedHeight(1)
            return sep

        metadata_title = make_section_title("Metadata", "metadata")
        metadata_sep = make_separator()
        align_arrays_title = make_section_title("Align Arrays", "align_arrays")
        align_arrays_sep = make_separator()
        bead_generation_title = make_section_title("Bead Generation", "bead_generation")
        bead_generation_sep = make_separator()
        self.apply_btn = QPushButton("Update Metadata")
        self.apply_btn.setVisible(False)
        self.apply_btn.clicked.connect(
            lambda: self.vm.apply_metadata(self.get_metadata_changes())
        )
        self.align_channels_btn = QPushButton("Align to Reference")
        self.align_channels_btn.clicked.connect(self.vm.align_channels)

        # StarDist controls
        self.use_stardist_checkbox = QCheckBox("Use StarDist(Recommended)")
        self.use_stardist_checkbox.setChecked(True)
        self.use_stardist_checkbox.stateChanged.connect(
            self.on_stardist_checkbox_changed
        )
        self.stardist_model_path_input = QLineEdit("model_4_400epoch_no_aug")
        self.stardist_model_path_input.setDisabled(True)  # Disabled by default

        self.upload_protein_key_btn = QPushButton("Upload Protein/Gene Key Files")
        self.upload_protein_key_btn.clicked.connect(self.upload_protein_key_files)
        self.protein_key_files_label = QLabel("No files uploaded.")
        self.generate_beads_btn = QPushButton("Generate and Export Beads")
        self.generate_beads_btn.clicked.connect(self.generate_beads_sig.emit)
        self.upload_beads_btn = QPushButton("Upload Bead CSV")
        self.upload_beads_btn.clicked.connect(self.upload_beads_sig.emit)
        self.inspect_beads_btn = QPushButton("Inspect Beads")
        self.inspect_beads_btn.clicked.connect(self.vm.inspect_beads)
        stats_title = make_section_title("Statistics", "statistics")
        stats_sep = make_separator()
        self.total_beads_label = QLabel("Total Beads: N/A")
        self.filtered_beads_label = QLabel("Filtered Beads: N/A")
        self.mean_rows_label = QLabel("Mean Rows per Protein: N/A")
        self.error_label = QLabel("Error Rate: N/A")
        self.counts_table = QTableWidget()
        self.counts_table.setColumnCount(2)
        self.counts_table.setHorizontalHeaderLabels(["Protein Name", "Count"])
        self.counts_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.counts_table_container = QWidget()
        self.counts_table_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        counts_table_layout = QVBoxLayout()
        counts_table_layout.setContentsMargins(0, 0, 0, 0)
        counts_table_layout.addWidget(self.counts_table)
        self.counts_table_container.setLayout(counts_table_layout)

        self.form_layout.addRow(metadata_title)
        self.form_layout.addRow(metadata_sep)

        self.prefix_label = QLabel("File Prefix:")
        self.axes_label = QLabel("Axes:")
        self.unit_label = QLabel("Unit:")
        self.phys_size_x_label = QLabel("PhysSizeX:")
        self.phys_size_y_label = QLabel("PhysSizeY:")
        self.align_ch_label = QLabel("Align Ch:")
        self.max_size_label = QLabel("Max Size:")

        self.form_layout.addRow(self.prefix_label, self.prefix_input)
        self.form_layout.addRow(self.prefix_checkbox)
        self.form_layout.addRow(self.axes_label, self.axes_input)
        self.form_layout.addRow(self.unit_label, self.unit_input)
        self.form_layout.addRow(self.phys_size_x_label, self.size_x_input)
        self.form_layout.addRow(self.phys_size_y_label, self.size_y_input)
        self.form_layout.addRow(self.align_ch_label, self.channel_input)
        self.form_layout.addRow(self.max_size_label, self.max_size_input)
        self.form_layout.addRow(self.apply_btn)
        self._section_widgets["metadata"] = [
            self.prefix_label,
            self.prefix_input,
            self.prefix_checkbox,
            self.axes_label,
            self.axes_input,
            self.unit_label,
            self.unit_input,
            self.phys_size_x_label,
            self.size_x_input,
            self.phys_size_y_label,
            self.size_y_input,
            self.align_ch_label,
            self.channel_input,
            self.max_size_label,
            self.max_size_input,
            self.apply_btn,
            metadata_sep,
        ]

        self.form_layout.addRow(align_arrays_title)
        self.form_layout.addRow(align_arrays_sep)

        self.num_tiles_label = QLabel("Num Tiles:")
        self.overlap_label = QLabel("Overlap:")
        self.ncc_thresh_label = QLabel("NCC Thresh:")

        self.form_layout.addRow(self.num_tiles_label, self.num_tiles_input)
        self.form_layout.addRow(self.overlap_label, self.overlap_input)
        self.form_layout.addRow(self.ncc_thresh_label, self.threshold_container)

        self.apply_shading_checkbox = QCheckBox("Apply shading correction")
        self.apply_shading_checkbox.setChecked(True)
        self.form_layout.addRow(self.align_channels_btn)
        self.form_layout.addRow(self.apply_shading_checkbox)
        self._section_widgets["align_arrays"] = [
            self.num_tiles_label,
            self.num_tiles_input,
            self.overlap_label,
            self.overlap_input,
            self.ncc_thresh_label,
            self.threshold_container,
            self.align_channels_btn,
            self.apply_shading_checkbox,
            align_arrays_sep,
        ]

        self.form_layout.addRow(bead_generation_title)
        self.form_layout.addRow(bead_generation_sep)
        self.form_layout.addRow(self.use_stardist_checkbox)

        self.stardist_model_label = QLabel("Model Path:")
        self.form_layout.addRow(
            self.stardist_model_label, self.stardist_model_path_input
        )

        self.form_layout.addRow(self.upload_protein_key_btn)
        self.form_layout.addRow(self.protein_key_files_label)
        self.form_layout.addRow(self.generate_beads_btn)
        self.form_layout.addRow(self.upload_beads_btn)
        self._section_widgets["bead_generation"] = [
            self.use_stardist_checkbox,
            self.stardist_model_label,
            self.stardist_model_path_input,
            self.upload_protein_key_btn,
            self.protein_key_files_label,
            self.generate_beads_btn,
            self.upload_beads_btn,
            # self.inspect_beads_btn,
            bead_generation_sep,
        ]

        self.form_layout.addRow(stats_title)
        self.form_layout.addRow(stats_sep)
        self.form_layout.addRow(self.total_beads_label)
        self.form_layout.addRow(self.mean_rows_label)

        self.form_layout.addRow(self.filtered_beads_label)
        self.form_layout.addRow(self.error_label)
        self.form_layout.addRow(self.inspect_beads_btn)
        self.form_layout.addRow(self.counts_table_container)
        self._section_widgets["statistics"] = [
            self.total_beads_label,
            self.mean_rows_label,
            self.filtered_beads_label,
            self.error_label,
            self.inspect_beads_btn,
            self.counts_table_container,
            stats_sep,
        ]

        self.vm.update_metadata_view_sig.connect(self.update_metadata)
        self.vm.metadata_corrected_sig.connect(self.on_metadata_corrected)

        self.content_widget = QWidget()
        self.content_widget.setLayout(self.form_layout)
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout()
        layout.addWidget(self.content_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.update_metadata([])

    def on_prefix_checkbox_changed(self, state):
        is_checked = state == Qt.CheckState.Checked
        self.prefix_input.setDisabled(is_checked)
        if self.vm.selected_files:
            for file_item in self.vm.selected_files:
                file_item.metadata.use_status_as_prefix = is_checked

    def on_stardist_checkbox_changed(self, state):
        """Enable/disable StarDist model path input based on checkbox state."""
        is_checked = state == Qt.CheckState.Checked
        self.stardist_model_path_input.setDisabled(not is_checked)

    def _on_input_changed(self):
        """Save all input values to selected FileItems immediately."""
        if not self.vm.selected_files:
            return
        metadata_changes = self.get_metadata_changes()
        files_to_update = []

        for file_item in self.vm.selected_files:
            max_size_changed = "max_size" in metadata_changes

            if "prefix" in metadata_changes:
                file_item.metadata.prefix = metadata_changes["prefix"]
            if "axes" in metadata_changes:
                file_item.metadata.axes = metadata_changes["axes"]
            if "unit" in metadata_changes:
                file_item.metadata.unit = metadata_changes["unit"]
            if "PhysicalSizeX" in metadata_changes:
                file_item.metadata.PhysicalSizeX = metadata_changes["PhysicalSizeX"]
            if "PhysicalSizeY" in metadata_changes:
                file_item.metadata.PhysicalSizeY = metadata_changes["PhysicalSizeY"]
            if "reference_channel" in metadata_changes:
                file_item.metadata.reference_channel = metadata_changes[
                    "reference_channel"
                ]
            if "num_tiles" in metadata_changes:
                file_item.metadata.num_tiles = metadata_changes["num_tiles"]
            if "overlap" in metadata_changes:
                file_item.metadata.overlap = metadata_changes["overlap"]
            if "ncc_threshold" in metadata_changes:
                file_item.threshold = metadata_changes["ncc_threshold"]

            if max_size_changed:
                max_size = metadata_changes["max_size"]
                original_shape = file_item.original_shape

                if len(original_shape) >= 2:
                    max_viable_size = min(
                        min(int(original_shape[-2]), int(original_shape[-1])), max_size
                    )
                    file_item.metadata.max_size = max_viable_size

                    if len(original_shape) == 3:
                        file_item.shape = (
                            original_shape[0],
                            max_viable_size,
                            max_viable_size,
                        )
                    else:
                        file_item.shape = (max_viable_size, max_viable_size)
                else:
                    file_item.metadata.max_size = max_size
                    file_item.shape = (max_size, max_size)

                files_to_update.append(file_item)

        if files_to_update:
            self.vm.file_shape_update_sig.emit(files_to_update)

    def _set_widget_states(self, enabled):
        """Enable or disable all input widgets."""
        input_widgets = [
            self.prefix_input,
            self.channel_input,
            self.axes_input,
            self.unit_input,
            self.size_x_input,
            self.size_y_input,
            self.max_size_input,
            self.num_tiles_input,
            self.overlap_input,
            self.threshold_slider,
        ]
        for widget in input_widgets:
            widget.setDisabled(not enabled)

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
        is_disabled = len(metadata_list) == 0

        self._set_widget_states(not is_disabled)

        if is_disabled:
            self.prefix_input.setText("")
            self.axes_input.setText("")
            self.unit_input.setText("")
            self.size_x_input.setText("")
            self.size_y_input.setText("")
            self.channel_input.setText("")
            self.max_size_input.setText("")
            self.num_tiles_input.setText("")
            self.overlap_input.setText("")
            self.threshold_slider.setValue(70)
            self.threshold_value_label.setText("0.70")
            self.prefix_checkbox.setChecked(False)
            return

        def set_field(widget, attribute_name, is_float=False):
            all_values = [
                getattr(item.metadata, attribute_name) for item in metadata_list
            ]
            if is_float:
                # For floats, round to a certain precision before comparing
                try:
                    rounded_values = {round(float(val), 6) for val in all_values}
                    if len(rounded_values) == 1:
                        widget.setText(str(all_values[0]))  # show original value
                    else:
                        widget.setText("...")
                except (ValueError, TypeError):
                    widget.setText("...")  # In case of non-float values
            else:
                unique_values = set(all_values)
                if len(unique_values) == 1:
                    widget.setText(str(unique_values.pop()))
                else:
                    widget.setText("...")

        set_field(self.prefix_input, "prefix")
        set_field(self.axes_input, "axes")
        set_field(self.unit_input, "unit")
        set_field(self.size_x_input, "PhysicalSizeX", is_float=True)
        set_field(self.size_y_input, "PhysicalSizeY", is_float=True)
        set_field(self.channel_input, "reference_channel")
        set_field(self.max_size_input, "max_size")
        set_field(self.num_tiles_input, "num_tiles")
        set_field(self.overlap_input, "overlap")

        if metadata_list:
            all_values = [item.threshold for item in metadata_list]
            try:
                rounded_values = {round(float(val), 2) for val in all_values}
                if len(rounded_values) == 1:
                    threshold = all_values[0]
                    slider_value = int(threshold * 100)
                    self.threshold_slider.setValue(slider_value)
                    self.threshold_value_label.setText(f"{threshold:.2f}")
                else:
                    self.threshold_slider.setValue(70)
                    self.threshold_value_label.setText("...")
            except (ValueError, TypeError):
                self.threshold_slider.setValue(70)
                self.threshold_value_label.setText("...")

        # Handle checkbox state - use the persisted flag from metadata
        if metadata_list:
            use_status_values = {
                item.metadata.use_status_as_prefix for item in metadata_list
            }
            if len(use_status_values) == 1:
                self.prefix_checkbox.setChecked(use_status_values.pop())
            else:
                self.prefix_checkbox.setChecked(False)
        else:
            self.prefix_checkbox.setChecked(False)

        self.on_prefix_checkbox_changed(self.prefix_checkbox.checkState())

    def get_metadata_changes(self):
        metadata_changes = {}

        prefix = self.prefix_input.text()
        if prefix and prefix != "...":
            metadata_changes["prefix"] = prefix

        axes = self.axes_input.text()
        if axes and axes != "...":
            metadata_changes["axes"] = axes

        unit = self.unit_input.text()
        if unit and unit != "...":
            metadata_changes["unit"] = unit

        try:
            size_x = self.size_x_input.text()
            if size_x and size_x != "...":
                metadata_changes["PhysicalSizeX"] = float(size_x)
        except ValueError:
            pass

        try:
            size_y = self.size_y_input.text()
            if size_y and size_y != "...":
                metadata_changes["PhysicalSizeY"] = float(size_y)
        except ValueError:
            pass

        try:
            channel = self.channel_input.text()
            if channel and channel != "...":
                metadata_changes["reference_channel"] = int(channel)
        except ValueError:
            pass

        try:
            max_size = self.max_size_input.text()
            if max_size and max_size != "...":
                metadata_changes["max_size"] = int(max_size)
        except ValueError:
            pass

        try:
            num_tiles = self.num_tiles_input.text()
            if num_tiles and num_tiles != "...":
                metadata_changes["num_tiles"] = int(num_tiles)
        except ValueError:
            pass

        try:
            overlap = self.overlap_input.text()
            if overlap and overlap != "...":
                metadata_changes["overlap"] = int(overlap)
        except ValueError:
            pass

        metadata_changes["ncc_threshold"] = self.threshold_slider.value() / 100.0

        metadata_changes["use_status_as_prefix"] = self.prefix_checkbox.isChecked()

        return metadata_changes

    def update_statistics(self, stats: dict):
        self.total_beads_label.setText(
            f"Total Beads: {stats.get('total_beads', 'N/A')}"
        )

        filtered_perc = stats.get("filtered_beads_percentage")
        if filtered_perc is not None:
            self.filtered_beads_label.setText(f"Filtered Beads: {filtered_perc:.2f}%")

        mean_rows = stats.get("mean_rows")
        if mean_rows is not None:
            self.mean_rows_label.setText(f"Mean Rows per Protein: {mean_rows:.2f}")
        else:
            self.mean_rows_label.setText("Mean Rows per Protein: N/A")
        error_rate = stats.get("error_rate")
        if error_rate is not None:
            self.error_label.setText(f"Error Rate: {error_rate:.4f}%")
        else:
            self.error_label.setText("Error Rate: N/A")

        counts_table_data = stats.get("counts_table")
        if counts_table_data is not None and not counts_table_data.empty:
            self.counts_table.setRowCount(len(counts_table_data))
            i = 0
            for _, row in counts_table_data.iterrows():
                self.counts_table.setItem(
                    i, 0, QTableWidgetItem(str(row["Protein name"]))
                )
                self.counts_table.setItem(i, 1, QTableWidgetItem(str(row["row_count"])))
                i += 1
        else:
            self.counts_table.setRowCount(0)

    def export_all(self):
        folder = QFileDialog.getExistingDirectory()
        if folder:
            self.export_all_sig.emit(folder, self.vm.selected_files)

    def get_stardist_settings(self):
        """Get StarDist configuration from UI."""
        return {
            "use_stardist": self.use_stardist_checkbox.isChecked(),
            "model_name": self.stardist_model_path_input.text(),
        }

    def on_metadata_corrected(self, corrections: dict):
        for key, value in corrections.items():
            if key == "max_size":
                self.max_size_input.setText(str(value))
