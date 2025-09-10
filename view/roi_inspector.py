import math
import re

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDoubleValidator,
    QImage,
    QIntValidator,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
)

from utils import adjust_contrast, to_uint8,find_min_std_partition


def merge_bead_data_with_protein_profile(bead_data, protein_profile, merge_columns):
    """
    Merge bead data with protein profile. If protein_profile is empty,
    create one by grouping combinations and partitioning counts into valid/invalid groups.
    """
    # Convert merge columns to int in both dataframes
    for col in merge_columns:
        bead_data[col] = bead_data[col].astype(int)
        if not protein_profile.empty:
            protein_profile[col] = protein_profile[col].astype(int)
    
    
    # Handle empty protein_profile case
    if protein_profile.empty:
        # Group by merge columns to get count per combination
        combination_counts = bead_data.groupby(merge_columns).size().reset_index(name='count')
        
        # Extract the counts for partitioning
        counts = combination_counts['count'].tolist()
        
        if len(counts) > 1:
            # Use find_min_std_partition to separate into two groups
            groups, min_std = find_min_std_partition(counts)
            
            # Determine which group has lower values (invalid) and higher values (valid)
            # Compare the minimum values of each group instead of means
            group1_min = min(groups[0]) if groups[0] else float('inf')
            group2_min = min(groups[1]) if groups[1] else float('inf')
            
            if group1_min <= group2_min:
                invalid_counts_set = set(groups[0])
                valid_counts_set = set(groups[1])
            else:
                invalid_counts_set = set(groups[1])
                valid_counts_set = set(groups[0])
        else:
            # If only one combination, consider it invalid
            invalid_counts_set = set(counts)
            valid_counts_set = set()
        
        # Create protein profile dataframe
        protein_profile_data = []
        valid_protein_counter = 1
        
        # Assign protein names to each combination
        for _, row in combination_counts.iterrows():
            if row['count'] in invalid_counts_set:
                protein_name = "Invalid"
            else:
                # Assign unique protein names (Protein 1, Protein 2, etc.) to valid combinations
                protein_name = f"Protein {valid_protein_counter}"
                valid_protein_counter += 1
            
            # Create row for protein profile
            profile_row = {}
            for col in merge_columns:
                profile_row[col] = row[col]
            profile_row['Protein name'] = protein_name
            protein_profile_data.append(profile_row)
        
        # Create the protein profile dataframe
        protein_profile = pd.DataFrame(protein_profile_data)
    
    # Merge bead_data with protein_profile
    # Create the filtered row
    # filtered_row = {col: 255 for col in merge_columns}
    # filtered_row['Protein name'] = 'Filtered'

    # protein_profile = pd.concat([protein_profile, pd.DataFrame([filtered_row])], ignore_index=True)
    
    bead_data = bead_data.merge(protein_profile, how="left", on=merge_columns)
    
    # Fill NaN values with "Invalid"
    bead_data['Protein name'].fillna("Invalid", inplace=True)
    # For all merge columns being 255
    mask_all_255 = (bead_data[merge_columns] == 255).all(axis=1)
    bead_data.loc[mask_all_255, 'Protein name'] = "Filtered"
    
    return bead_data


class NullableIntValidator(QIntValidator):
    def validate(self, input_str, pos):
        if input_str == "":
            return (self.State.Acceptable, input_str, pos)
        return super().validate(input_str, pos)


class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.target_item = QGraphicsPixmapItem()
        self._scene.addItem(self.target_item)
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def set_images(self, target_pixmap: QPixmap):
        self.target_item.setPixmap(target_pixmap)
        QTimer.singleShot(0, self.reset_zoom)  # center after render updates

    def reset_zoom(self):
        self.get_scene().setSceneRect(self.get_scene().itemsBoundingRect())
        self.fitInView(self.target_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(self.target_item)

    def get_scene(self):
        s = self.scene()
        assert s is not None
        return s

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if event is None:
            return
        angle = event.angleDelta().y()
        if angle > 0:
            zoom_factor = 1.15  # Zoom in
        else:
            zoom_factor = 1 / 1.15  # Zoom out

        self.scale(zoom_factor, zoom_factor)

    def add_bead_centers(self, centers: list[tuple[int, int]], radius: int = 1):
        """Add overlay dots at given (x, y) centers."""
        pen = QPen(Qt.GlobalColor.red)
        brush = QBrush(Qt.GlobalColor.red)
        for x, y in centers:
            dot = self._scene.addEllipse(
                x - radius, y - radius, 2 * radius, 2 * radius, pen, brush
            )
            assert dot is not None
            dot.setZValue(1)  # Make sure it appears above the image

from image_processing import gaussian_kernel
class ROI_Inspector(QDialog):
    show_bead_signal = pyqtSignal(np.ndarray)

    def __init__(self, snapshot_data: dict):
        super().__init__(None)

        self.target_image = snapshot_data["bf_image"].copy()
        self.beads = snapshot_data.get("beads", None)
        self.cycles = snapshot_data.get("cycles", {})
        self.bboxs = snapshot_data.get("bboxs", None)
        self.labeled_image = snapshot_data.get("labeled_image", None)
        self.protein_profile = snapshot_data.get("protein_profile", pd.DataFrame())
        self.bright_fields = snapshot_data.get("bright_fields", None)
        merge_columns = []
        for i in range(len(self.cycles)):
            merge_columns.append(f"cy{i}")
            
        # Merge bead data with protein profile if both are provided
        if (
            self.beads is not None
            and self.protein_profile is not None
            and merge_columns
        ):
            self.merged_bead_data = merge_bead_data_with_protein_profile(
                    self.beads.copy(), self.protein_profile.copy(), merge_columns
                )
            
            print(self.merged_bead_data.head())
        else:
            self.merged_bead_data = None
            self.mean_rows_per_protein = None

        if self.labeled_image is not None and len(self.labeled_image) == 0:
            self.labeled_image = None
        # adjust cycles contrast:
        if self.beads is None:
            print("Warning: No beads data provided.")
        self.adjust_contrast = False
        self.downscaled = False
        self._setup_ui()
        self.create_direct_overlay()
        self.image_view.mouseDoubleClickEvent = self.inspect_roi

    def _setup_ui(self):
        self.setWindowTitle(f"Bead Data Analysis")
        self.resize(1400, 800)  # Increased width to accommodate sidebar

        # Main splitter to separate sidebar from main content
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        if self.merged_bead_data is not None:
            # Create sidebar for invalid beads
            self._setup_sidebar()
            main_splitter.addWidget(self.sidebar_widget)

        # Main content widget
        main_content_widget = self._create_main_content_widget()
        main_splitter.addWidget(main_content_widget)

        # Set splitter proportions (sidebar takes ~20% of space)
        main_splitter.setSizes([280, 1120])

        # Set main layout
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

    def _setup_sidebar(self):
        assert self.merged_bead_data is not None
        """Create sidebar with list of invalid/filtered beads and resample buttons."""
        self.sidebar_widget = QGroupBox("Bead Inspector")
        sidebar_layout = QVBoxLayout()

        # List widget for beads
        self.beads_list = QListWidget()
        self.beads_list.itemClicked.connect(self._on_invalid_bead_selected)

        # Resample buttons
        self.resample_invalids_button = QPushButton("Resample Invalids")
        self.resample_invalids_button.clicked.connect(self._populate_invalid_beads_list)
        self.resample_filtered_button = QPushButton("Resample Filtereds")
        self.resample_filtered_button.clicked.connect(self._populate_filtered_beads_list)

        # Add count label
        invalid_count = len(
            self.merged_bead_data[self.merged_bead_data["Protein name"] == "Invalid"]
        )
        filtered_count = len(
            self.merged_bead_data[self.merged_bead_data["Protein name"] == "Filtered"]
        )
        self.invalid_count_label = QLabel(
            f"Invalid Beads: {invalid_count}; Filtered: {filtered_count}"
        )

        sidebar_layout.addWidget(self.invalid_count_label)
        sidebar_layout.addWidget(self.beads_list)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.resample_invalids_button)
        button_layout.addWidget(self.resample_filtered_button)
        sidebar_layout.addLayout(button_layout)

        sidebar_layout.addStretch()  # Push everything to top

        self.sidebar_widget.setLayout(sidebar_layout)
        self.sidebar_widget.setMaximumWidth(300)
        self.sidebar_widget.setMinimumWidth(250)
        # Populate the list with invalid beads
        self._populate_invalid_beads_list()

    def _populate_filtered_beads_list(self):
        """Populate the sidebar list with a random selection of 10 filtered beads."""
        if self.merged_bead_data is not None:
            filtered_beads = self.merged_bead_data[
                self.merged_bead_data["Protein name"] == "Filtered"
            ]

            # Randomly sample 10 rows (or fewer if less than 10)
            sampled_beads = filtered_beads.sample(
                n=min(10, len(filtered_beads)), random_state=None
            )

            self.beads_list.clear()

            for idx, row in sampled_beads.iterrows():
                x, y = int(row["x"]), int(row["y"])
                item_text = f"Bead {idx}: ({x}, {y})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, (idx, x, y))
                self.beads_list.addItem(item)

    def _populate_invalid_beads_list(self):
        """Populate the sidebar list with a random selection of 10 invalid beads."""
        if self.merged_bead_data is not None:
            invalid_beads = self.merged_bead_data[
                self.merged_bead_data["Protein name"] == "Invalid"
            ]

            # Randomly sample 10 rows (or fewer if less than 10)
            sampled_beads = invalid_beads.sample(
                n=min(10, len(invalid_beads)), random_state=None
            )

            self.beads_list.clear()  # Clear previous items if any

            for idx, row in sampled_beads.iterrows():
                x, y = int(row["x"]), int(row['y'])
                item_text = f"Bead {idx}: ({x}, {y})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, (idx, x, y))
                self.beads_list.addItem(item)

    def _on_invalid_bead_selected(self, item):
        """Handle selection of an invalid bead from the sidebar list."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            idx, x, y = data
            # Set the coordinates in the input fields
            self.x_input.setText(str(x))
            self.y_input.setText(str(y))
            # Call inspect_roi with None event to trigger inspection
            self.inspect_roi(None)

    def _resample_selected_bead(self):
        """Handle resampling of selected invalid bead."""
        current_item = self.beads_list.currentItem()
        if current_item:
            data = current_item.data(Qt.ItemDataRole.UserRole)
            if data:
                idx, x, y = data
                # Here you would implement your resampling logic
                # For now, just show a message
                QMessageBox.information(
                    self,
                    "Resample Bead",
                    f"Resampling bead {idx} at coordinates ({x}, {y})\n"
                    "This would trigger your resampling algorithm.",
                )

    def _create_main_content_widget(self):
        """Create the main content widget with image view and controls."""
        main_widget = QGroupBox()
        main_layout = QVBoxLayout(main_widget)

        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.setChecked(self.adjust_contrast)
        self.enhance_contrast_checkbox.stateChanged.connect(
            self._on_contrast_checkbox_changed
        )

        instruction_text = "Double-click image to inspect ROI"
        self.preview_label = QLabel(instruction_text)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_view = ZoomableImageView()
        self.image_view.setMinimumSize(800, 500)

        self.control_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._setup_editable_controls()

        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.enhance_contrast_checkbox)
        main_layout.addWidget(self.image_view)
        main_layout.addLayout(self.control_layout)
        main_layout.addLayout(self.button_layout)

        return main_widget

    def _on_contrast_checkbox_changed(self, state):
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self.create_direct_overlay()

    def _setup_editable_controls(self):
        """Create UI controls for when manual editing is enabled."""

        trans_group = QGroupBox("ROI Center")
        trans_layout = QHBoxLayout()
        int_validator = NullableIntValidator(-99999, 99999)

        self.x_input = QLineEdit("0")
        self.x_input.setValidator(int_validator)
        self.x_input.setFixedWidth(50)

        self.y_input = QLineEdit("0")
        self.y_input.setValidator(int_validator)
        self.y_input.setFixedWidth(50)

        self.apply_trans_button = QPushButton("Inspect")
        trans_layout.addWidget(QLabel("x:"))
        trans_layout.addWidget(self.x_input)
        trans_layout.addWidget(QLabel("y:"))
        trans_layout.addWidget(self.y_input)
        trans_layout.addWidget(self.apply_trans_button)
        trans_group.setLayout(trans_layout)

        radius_group = QGroupBox("ROI Radius")
        radius_layout = QHBoxLayout()
        self.radius_input = QLineEdit("2")
        self.radius_input.setPlaceholderText("Radius")
        self.radius_input.setValidator(QIntValidator(1, 10000))
        radius_layout.addWidget(self.radius_input)
        radius_group.setLayout(radius_layout)

        scale_group = QGroupBox("Surrounding Scale")
        scale_layout = QHBoxLayout()
        self.scale_input = QLineEdit("2.0")
        self.scale_input.setPlaceholderText("2.0")
        self.scale_input.setValidator(QDoubleValidator(1.0, 10000, 6))
        scale_layout.addWidget(self.scale_input)
        scale_group.setLayout(scale_layout)

        roi_mode_group = QGroupBox("ROI Mode")
        roi_mode_layout = QHBoxLayout()
        self.roi_mode_input = QCheckBox("ROI Mode")
        self.roi_mode_input.setChecked(False)
        roi_mode_layout.addWidget(self.roi_mode_input)
        roi_mode_group.setLayout(roi_mode_layout)
        self.roi_mode_input.clicked.connect(lambda _: self.create_direct_overlay())
        self.apply_trans_button.clicked.connect(self.inspect_roi)

        self.reset_button = QPushButton("Reset Zoom")
        self.reset_button.clicked.connect(self.reset_zoom)

        self.control_layout.addWidget(trans_group)
        self.control_layout.addWidget(radius_group)
        self.control_layout.addWidget(scale_group)
        if self.labeled_image is not None:
            self.control_layout.addWidget(roi_mode_group)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.reset_button)
        self._setup_confirm_cancel_buttons()

    def _setup_confirm_cancel_buttons(self):
        self.cancel_button = QPushButton("Exit")
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addWidget(self.cancel_button)

    def inspect_roi(self, event=None):
        x, y = None, None
        if event is None or not hasattr(event, "position"):
            try:
                xtext = self.x_input.text()
                ytext = self.y_input.text()
                if xtext == "":
                    xtext = "0"
                if ytext == "":
                    ytext = "0"
                x = int(xtext)
                y = int(ytext)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    "Please enter valid integer values for x and y.",
                )
                return
        else:
            pos = event.position()
            scene_pos = self.image_view.mapToScene(int(pos.x()), int(pos.y()))
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            self.x_input.setText(str(x))
            self.y_input.setText(str(y))
            print(f"Double-click at scene position ({x}, {y})")

        self.image_view.centerOn(x, y)

        radius = self.radius_input.text()
        try:
            radius = int(radius) if radius != "" else 2
            if radius <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a valid positive integer for radius.",
            )
            return
        scale = self.scale_input.text()
        try:
            scale = float(scale) if scale != "" else 2.0
            if scale <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a valid positive number for scale.",
            )
            return
        self.image_view.fitInView(
            x - radius * scale,
            y - radius * scale,
            2 * radius * scale,
            2 * radius * scale,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

        # Use merged data instead of original beads data
        data_to_query = (
            self.merged_bead_data if self.merged_bead_data is not None else self.beads
        )
        assert isinstance(data_to_query, pd.DataFrame)
        output = data_to_query.query(f"x=={x} & y=={y}")
        bbox = None
        idx = None
        if not output.empty:
            idx = output.index[0]
            if self.bboxs is not None:
                bbox = self.bboxs.loc[idx]
                h, w = self.target_image.shape
                y1, x1, y2, x2 = map(int, re.findall(r"-?\d+", bbox))
                # Clip to image bounds
                x1 = np.clip(x1, 0, w - 1)
                x2 = np.clip(x2, 0, w - 1)
                y1 = np.clip(y1, 0, h - 1)
                y2 = np.clip(y2, 0, h - 1)
                bbox = (x1, y1, x2, y2)
        if self.labeled_image is None or not self.roi_mode_input.isChecked():
            bbox = None
        assert isinstance(x, int) and isinstance(y, int)
        print(f"Inspecting ROI at ({x}, {y}) with radius {radius} and scale {scale}")
        rois = {}
        bright_field_roi = {}
        for key, cycle in (self.cycles or {}).items():
            if cycle.ndim == 3:
                h, w = cycle.shape[1], cycle.shape[2]
            elif cycle.ndim == 2:
                h, w = cycle.shape[0], cycle.shape[1]
            else:
                print(f"Cycle {key} has unexpected shape {cycle.shape}")
                continue
            x0 = max(0, x - int(radius * scale))
            x1 = min(w, x + int(radius * scale) + 1)
            y0 = max(0, y - int(radius * scale))
            y1 = min(h, y + int(radius * scale) + 1)
            if bbox is not None:
                x0, y0, x1, y1 = expand_bbox(bbox, scale)
            if cycle.ndim == 3:
                roi = cycle[:, y0:y1, x0:x1]
            else:
                roi = cycle[y0:y1, x0:x1]
            rois[key] = roi
            if self.bright_fields is not None and key in self.bright_fields:
                bright_field_roi[key] = self.bright_fields[key][y0:y1, x0:x1]
            print(
                f"Cycle {key}: extracted ROI shape {roi.shape} at ({x0}:{x1}, {y0}:{y1})"
            )

        popup = ROI_Grid_Display(
            rois, (x, y), radius, scale, bbox, output, bright_field_roi
        )
        popup.exec()  # modal dialog to display ROIs

    def reset_zoom(self, event=None):
        self.image_view.reset_zoom()
        if event:
            event.accept()

    def _setup_view_only_controls(self):
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.close_button)
        self.button_layout.addStretch()

    def create_direct_overlay(self):
        print("Creating direct overlay")
        rgb_image = None
        if self.labeled_image is None or not self.roi_mode_input.isChecked():
            target_gray = to_uint8(self.target_image)

            if self.adjust_contrast:
                target_gray = to_uint8(
                    adjust_contrast(target_gray.astype(np.float32), 30, 99)
                )

            # Convert grayscale to RGB
            rgb_image = np.stack([target_gray] * 3, axis=-1)  # Shape: (H, W, 3)
        else:
            rgb_image = self.labeled_image
        # assert isinstance(rgb_image, np.ndarray)
        h, w = rgb_image.shape[:-1]

        # Draw red centers - use merged data if available
        data_to_use = (
            self.merged_bead_data if self.merged_bead_data is not None else self.beads
        )
        # Color invalid beads differently (e.g., yellow instead of red)
        assert isinstance(data_to_use, pd.DataFrame)
        if self.merged_bead_data is not None:
            xs = data_to_use["x"].astype(float).round().astype(int).to_numpy()
            ys = data_to_use["y"].astype(float).round().astype(int).to_numpy()
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs, ys = xs[valid], ys[valid]
            protein_names = data_to_use.loc[valid, "Protein name"].to_numpy()

            # Boolean masks for invalid/valid
            invalid_mask = protein_names == "Invalid" | protein_names == "Filtered"
            valid_mask = ~invalid_mask

            # Yellow for invalid
            rgb_image[ys[invalid_mask], xs[invalid_mask]] = [255, 255, 0]
            # Red for valid
            rgb_image[ys[valid_mask], xs[valid_mask]] = [255, 0, 0]
        else:
            xs = data_to_use["x"].astype(float).round().astype(int).to_numpy()
            ys = data_to_use["y"].astype(float).round().astype(int).to_numpy()
            rgb_image[ys, xs] = [255, 0, 0]  # Red

        # Convert to QImage and show
        rgb_image = np.ascontiguousarray(rgb_image)
        qimage = QImage(rgb_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        target_pixmap = QPixmap.fromImage(qimage)

        self.image_view.set_images(target_pixmap)


class ROI_Grid_Display(QDialog):
    def __init__(
        self,
        rois: dict,
        center: tuple,
        radius: int,
        scale: float,
        bbox,
        output: pd.DataFrame | None = None,
        bright_field_roi: dict | None = None,
    ):
        super().__init__(None)
        self.setWindowTitle("ROI Grid Display")
        self.resize(800, 600)
        layout = QVBoxLayout(self)

        info_label = QLabel(
            f"Center: {center}, Radius: {radius}, Scale: {scale:.2f}. "
            f"Showing {len(rois)} cycles."
        )
        layout.addWidget(info_label, 0)

        grid_layout = QGridLayout()
        row = 1
        # channels are columns
        # rows are cycles
        num_channels = len(rois.get("cy0", np.array([])))
        offset = 1
        if bright_field_roi is not None and len(bright_field_roi):
            bf_label = QLabel("Bright Field")
            grid_layout.addWidget(bf_label, 0, offset)
            offset += 1
        for i in range(len(rois)):
            cycle_label = QLabel(f"Cycle {i}")
            grid_layout.addWidget(cycle_label, i + 1, 0)
        for i in range(num_channels):
            channel_label = QLabel(f"Channel {i}")
            grid_layout.addWidget(channel_label, 0, i + offset)
        if output is not None and len(output):
            out_label = QLabel(f"Output")
            grid_layout.addWidget(out_label, 0, num_channels + offset)
        # add grayscale bright field images if provided
        starting_col = 1
        if bright_field_roi is not None and len(bright_field_roi):
            for key, bf_roi in bright_field_roi.items():
                if bf_roi.ndim == 2:
                    bf_img = to_uint8(bf_roi)
                    bf_img = QImage(
                        bf_img.data,
                        bf_img.shape[1],
                        bf_img.shape[0],
                        bf_img.strides[0],
                        QImage.Format.Format_Grayscale8,
                    )
                    bf_pixmap = QPixmap.fromImage(bf_img)
                    bf_label = OverlayLabel(
                        bf_pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio),
                        bbox,
                    )
                    grid_layout.addWidget(bf_label, row, starting_col)
                row += 1
            starting_col += 1
        row = 1
        for key, roi in rois.items():
            col = starting_col
            if roi.ndim == 3:
                # Multi-channel
                for c in range(roi.shape[0]):
                    roi_colorized = colorize_grayscale(to_uint8(roi[c]), c % 4)
                    roi_label = OverlayLabel(
                        roi_colorized.scaled(
                            50, 50, Qt.AspectRatioMode.KeepAspectRatio
                        ),
                        bbox,
                        roi=roi[c],
                    )
                    grid_layout.addWidget(roi_label, row, col)

                    col += 1
            elif roi.ndim == 2:
                # Single channel
                pixmap = colorize_grayscale(to_uint8(roi), 0)
                pixmap_label = QLabel()
                pixmap_label.setPixmap(
                    pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio)
                )
                grid_layout.addWidget(pixmap_label, row, col)
                col += 1
            if output is not None and len(output):
                try:
                    pred = output[key]
                except:
                    pred = None
                if pred is None:
                    pred = "N/A"
                else:
                    pred = str(pred.iloc[0])
                output_label = QLabel(pred)
                grid_layout.addWidget(output_label, row, col)

            row += 1

        layout.addLayout(grid_layout, 6)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        self.setLayout(layout)


from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel
from scipy.signal import correlate2d


class OverlayLabel(QLabel):
    def __init__(self, pixmap, bbox=None, parent=None, roi=None):
        super().__init__(parent)
        self.pixmap_to_draw = pixmap
        self.rect_to_draw = bbox_to_qrectf(bbox) if bbox else None
        if roi is not None:
            # calculate template matching score with roi
            print(roi.shape)
            self.overlay_text = f"{calculate_template_match(roi):.2f}"

        # Resize label to fit pixmap + text
        self.setFixedSize(
            self.pixmap_to_draw.width(),
            self.pixmap_to_draw.height() + 20,  # space for text
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the pixmap
        painter.drawPixmap(0, 0, self.pixmap_to_draw)

        # Draw the rectangle overlay if needed
        if self.rect_to_draw:
            pen = QPen(QColor("white"))
            pen.setWidthF(0.5)
            painter.setPen(pen)
            painter.drawRect(self.rect_to_draw)

        # Draw the text below the pixmap
        if hasattr(self, "overlay_text"):
            text_rect = QRectF(0, self.pixmap_to_draw.height(), self.width(), 20)
            painter.setPen(QPen(QColor("white")))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.overlay_text)

        painter.end()


def calculate_template_match(roi):
    gaussian_5x5 = np.array(
        [
            [0, 4, 7, 4, 0],
            [0, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [0, 16, 26, 16, 4],
            [0, 4, 7, 4, 0],
        ],
        dtype=np.float32,
    )

    # Normalize so sum = 1
    gaussian_5x5 /= gaussian_5x5.sum()
    gaussian_5x5 = gaussian_kernel(5)
    roi = adjust_contrast(roi.astype(np.float32), 10, 90)
    score = correlate2d(roi.astype(np.float32), gaussian_5x5, mode="valid")
    return np.median(score)


def bbox_to_qrectf(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return QRectF(x1, y1, width, height)


def readable_matrix_string(matrix: np.ndarray) -> str:
    if matrix.shape != (2, 3):
        return str(matrix)
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    angle_rad = math.atan2(c, a)
    angle_deg = math.degrees(angle_rad)
    scale_x = math.sqrt(a**2 + c**2)
    scale_y = math.sqrt(b**2 + d**2)
    return f"Translation: ({tx:.2f}, {ty:.2f}), Rotation: {angle_deg:.2f}°, Scale: (x: {scale_x:.2f}, y: {scale_y:.2f})"


colors = [
    (27, 158, 119),
    (217, 95, 2),
    (117, 112, 179),
    (231, 41, 138),
    (102, 166, 30),
    (230, 171, 2),
    (166, 118, 29),
    (102, 102, 102),
]


def colorize_grayscale(gray_img: np.ndarray, color_indx: int) -> QPixmap:
    """Colorize grayscale image and make black pixels fully transparent."""
    h, w = gray_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    color = colors[color_indx]
    for i, v in enumerate(color):
        rgba[:, :, i] = v

    rgba[:, :, 3] = gray_img

    qimage = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage)


def transform_to_matrix(t: QTransform):
    matrix = np.array(
        [
            [t.m11(), t.m21(), t.dx()],
            [t.m12(), t.m22(), t.dy()],
        ],
        dtype=np.float32,
    )
    return matrix


def expand_bbox(bbox, scale):
    x1, y1, x2, y2 = bbox

    # Compute center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Compute current width and height
    width = x2 - x1
    height = y2 - y1

    # Scale width and height
    new_width = width * scale
    new_height = height * scale

    # Compute new coordinates
    new_x1 = int(cx - new_width // 2)
    new_y1 = int(cy - new_height // 2)
    new_x2 = int(cx + new_width // 2)
    new_y2 = int(cy + new_height // 2)

    return (new_x1, new_y1, new_x2, new_y2)
