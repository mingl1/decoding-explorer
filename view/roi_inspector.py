import math
import os
import re

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDoubleValidator,
    QImage,
    QIntValidator,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QGraphicsItem,
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
    QRadioButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from enum import Enum
from utils import adjust_contrast, find_min_std_partition, to_uint8
from view.crop_dialog import CropCircleROI, CropPolyROI, CropRectROI, DrawMode
from image_processing import gaussian_kernel


class BeadColor(Enum):
    VALID = (0, 255, 0)       # Green
    INVALID = (255, 0, 0)     # Red
    FILTERED = (255, 255, 0)  # Yellow

    @property
    def qcolor(self):
        r, g, b = self.value
        return QColor(r, g, b, 150)


def point_in_polygon(x, y, polygon):
    num = len(polygon)
    j = num - 1
    c = False
    for i in range(num):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            c = not c
        j = i
    return c


def merge_bead_data_with_protein_profile(bead_data, protein_profile, merge_columns):
    for col in merge_columns:
        bead_data[col] = bead_data[col].astype(int)
        if not protein_profile.empty:
            protein_profile[col] = protein_profile[col].astype(int)

    if protein_profile.empty:
        combination_counts = (
            bead_data.groupby(merge_columns).size().reset_index(name="count")
        )
        counts = combination_counts["count"].tolist()

        if len(counts) > 1:
            groups, min_std = find_min_std_partition(counts)
            group1_min = min(groups[0]) if groups[0] else float("inf")
            group2_min = min(groups[1]) if groups[1] else float("inf")

            if group1_min <= group2_min:
                invalid_counts_set = set(groups[0])
                valid_counts_set = set(groups[1])
            else:
                invalid_counts_set = set(groups[1])
                valid_counts_set = set(groups[0])
        else:
            invalid_counts_set = set(counts)
            valid_counts_set = set()

        protein_profile_data = []
        valid_protein_counter = 1

        for _, row in combination_counts.iterrows():
            if row["count"] in invalid_counts_set:
                protein_name = "Invalid"
            else:
                protein_name = f"Protein {valid_protein_counter}"
                valid_protein_counter += 1

            profile_row = {}
            for col in merge_columns:
                profile_row[col] = row[col]
            profile_row["Protein name"] = protein_name
            protein_profile_data.append(profile_row)

        protein_profile = pd.DataFrame(protein_profile_data)

    bead_data = bead_data.merge(protein_profile, how="left", on=merge_columns)
    bead_data["Protein name"].fillna("Invalid", inplace=True)
    mask_any_255 = (bead_data[merge_columns] == 255).any(axis=1)
    bead_data.loc[mask_any_255, "Protein name"] = "Filtered"

    return bead_data


class NullableIntValidator(QIntValidator):
    def validate(self, input_str, pos):
        if input_str == "":
            return (self.State.Acceptable, input_str, pos)
        return super().validate(input_str, pos)


class ZoomableImageView(pg.GraphicsView):
    mouse_moved = pyqtSignal(int, int)
    double_clicked = pyqtSignal(QPointF)
    roi_added = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent, background="k")
        self._vb = pg.ViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self._vb.setMouseMode(pg.ViewBox.PanMode)
        self.setCentralItem(self._vb)
        self.image_items = []
        self.setMouseTracking(True)
        viewport = self.viewport()
        if viewport:
            viewport.setMouseTracking(True)
        self.draw_mode = DrawMode.SELECT
        self.temp_roi = None
        self.draw_start = None

    def set_images(self, arrays, opacities=None, reset_zoom=True):
        for item in self.image_items:
            self._vb.removeItem(item)
        self.image_items.clear()
        if opacities is None:
            opacities = [1.0] * len(arrays)
        for rgba, opacity in zip(arrays, opacities):
            item = pg.ImageItem(axisOrder="row-major")
            item.setImage(rgba, autoLevels=False, levels=[0, 255])
            item.setOpacity(opacity)
            self.image_items.append(item)
            self._vb.addItem(item)
        if reset_zoom:
            QTimer.singleShot(0, self.reset_zoom)

    def toggle_layer_visibility(self, index, visible):
        if 0 <= index < len(self.image_items):
            self.image_items[index].setVisible(visible)

    def reset_zoom(self):
        if not self.image_items:
            return
        rect = self.image_items[0].boundingRect()
        self._vb.setRange(rect=rect, padding=0)

    def get_scene(self):
        return self.sceneObj

    def mousePressEvent(self, event):
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.draw_mode != DrawMode.SELECT:
                scene_pos = self.mapToScene(event.pos())
                view_pos = self._vb.mapSceneToView(scene_pos)
                self._handle_roi_drawing_press(view_pos, event)
                event.accept()
                return
        super().mousePressEvent(event)

    def _handle_roi_drawing_press(self, view_pos, event):
        if self.draw_mode in (DrawMode.RECT, DrawMode.CIRCLE):
            self.draw_start = view_pos
            if self.draw_mode == DrawMode.RECT:
                self.temp_roi = CropRectROI(view_pos, QSize(1, 1))
            else:
                self.temp_roi = CropCircleROI(view_pos, QSize(1, 1))
            self._vb.addItem(self.temp_roi)
        elif self.draw_mode == DrawMode.POLY:
            if self.temp_roi is None:
                self.temp_roi = CropPolyROI()
                self._vb.addItem(self.temp_roi)
            else:
                if len(self.temp_roi.points) >= 3:
                    p0_screen = self.mapFromScene(self._vb.mapViewToScene(self.temp_roi.points[0]))
                    curr_screen = event.pos()
                    dist = ((p0_screen.x() - curr_screen.x()) ** 2 + (p0_screen.y() - curr_screen.y()) ** 2) ** 0.5
                    if dist < 15:
                        self.temp_roi.complete()
                        self.roi_added.emit(self.temp_roi)
                        self.temp_roi = None
                        self.viewport().unsetCursor()
                        return
            self.temp_roi.add_point(QPointF(view_pos.x(), view_pos.y()))

    def mouseMoveEvent(self, event):
        if event is None:
            return
        pos = event.position()
        scene_pos = self.mapToScene(int(pos.x()), int(pos.y()))
        view_pos = self._vb.mapSceneToView(scene_pos)
        if self.draw_mode != DrawMode.SELECT:
            self._handle_roi_drawing_move(view_pos, event)
            event.accept()
            return
        self.mouse_moved.emit(int(view_pos.x()), int(view_pos.y()))
        super().mouseMoveEvent(event)

    def _handle_roi_drawing_move(self, view_pos, event):
        if self.draw_mode in (DrawMode.RECT, DrawMode.CIRCLE) and self.temp_roi and self.draw_start:
            x1, y1 = self.draw_start.x(), self.draw_start.y()
            x2, y2 = view_pos.x(), view_pos.y()
            if self.draw_mode == DrawMode.RECT:
                x, y = min(x1, x2), min(y1, y2)
                w, h = abs(x2 - x1), abs(y2 - y1)
                self.temp_roi.setPos([x, y])
                self.temp_roi.setSize([w, h])
            else:
                r = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                self.temp_roi.setPos([x1 - r, y1 - r])
                self.temp_roi.setSize([r * 2, r * 2])
        elif self.draw_mode == DrawMode.POLY and self.temp_roi:
            if len(self.temp_roi.points) >= 3:
                p0 = self.temp_roi.points[0]
                p0_screen = self.mapFromScene(self._vb.mapViewToScene(p0))
                curr_screen = event.pos()
                dist = ((p0_screen.x() - curr_screen.x()) ** 2 + (p0_screen.y() - curr_screen.y()) ** 2) ** 0.5
                if dist < 15:
                    self.temp_roi.set_temp_point(p0)
                    self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)
                    return
            self.temp_roi.set_temp_point(QPointF(view_pos.x(), view_pos.y()))
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, event):
        if event is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.draw_mode != DrawMode.SELECT:
                self._handle_roi_drawing_release()
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def _handle_roi_drawing_release(self):
        if self.draw_mode in (DrawMode.RECT, DrawMode.CIRCLE) and self.temp_roi and self.draw_start:
            size = self.temp_roi.size()
            if size[0] > 5 and size[1] > 5:
                self.roi_added.emit(self.temp_roi)
            else:
                self._vb.removeItem(self.temp_roi)
            self.temp_roi = None
            self.draw_start = None

    def mouseDoubleClickEvent(self, event):
        if event is None:
            return
        if self.draw_mode == DrawMode.POLY and self.temp_roi:
            if self.temp_roi.complete():
                self.roi_added.emit(self.temp_roi)
            else:
                self._vb.removeItem(self.temp_roi)
            self.temp_roi = None
            self.viewport().unsetCursor()
            event.accept()
            return
        scene_pos = self.mapToScene(event.pos())
        view_pos = self._vb.mapSceneToView(scene_pos)
        self.double_clicked.emit(view_pos)
        event.accept()


def colorize_grayscale_rgba(gray_img: np.ndarray, color: str) -> np.ndarray:
    h, w = gray_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if color == "red":
        rgba[:, :, 0] = gray_img
    elif color == "green":
        rgba[:, :, 1] = gray_img
    elif color == "blue":
        rgba[:, :, 2] = gray_img
    elif color == "magenta":
        rgba[:, :, 0] = gray_img
        rgba[:, :, 2] = gray_img
    elif color == "cyan":
        rgba[:, :, 1] = gray_img
        rgba[:, :, 2] = gray_img
    elif color == "yellow":
        rgba[:, :, 0] = gray_img
        rgba[:, :, 1] = gray_img

    mask = gray_img > 0
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    return np.ascontiguousarray(rgba)


class ROIInspector(QDialog):
    show_bead_signal = pyqtSignal(np.ndarray)
    bead_crop_confirmed = pyqtSignal(list, str)

    def __init__(self, snapshot_data: dict, file_item=None, initial_tab: str = "inspect"):
        super().__init__(None)
        self.file_item = file_item
        self.initial_tab = initial_tab
        self.target_image = snapshot_data["bf_image"].copy()
        self.beads = snapshot_data.get("beads", None)
        self.cycles = snapshot_data.get("cycles", {})
        self.bboxs = snapshot_data.get("bboxs", None)
        self.labeled_image = snapshot_data.get("labeled_image", None)
        self.protein_profile = snapshot_data.get("protein_profile", pd.DataFrame())
        self.bright_fields = snapshot_data.get("bright_fields", None)
        self.created_rois = []
        self.bg_images = {
            False: None,
            True: None
        }
        self._minimap_cache = {
            False: None,
            True: None
        }

        merge_columns = []
        if self.beads is not None:
            merge_columns = [
                col
                for col in self.beads.columns
                if col.startswith("cy") and col[2:].isdigit()
            ]

        if (
            self.beads is not None
            and self.protein_profile is not None
            and merge_columns
        ):
            self.merged_bead_data = merge_bead_data_with_protein_profile(
                self.beads.copy(), self.protein_profile.copy(), merge_columns
            )
        else:
            self.merged_bead_data = None

        if self.labeled_image is not None and len(self.labeled_image) == 0:
            self.labeled_image = None

        self.adjust_contrast = False
        self._setup_ui()
        self.create_direct_overlay()

        if self.file_item is not None:
            if getattr(self.file_item, "bead_crop_mode", "include") == "exclude":
                self.exclude_radio.setChecked(True)
            else:
                self.include_radio.setChecked(True)

            if getattr(self.file_item, "bead_crop_rois", None) is not None:
                self._load_existing_rois(self.file_item.bead_crop_rois)

    def _setup_ui(self):
        self.setWindowTitle("Bead Data Analysis")
        self.resize(1400, 800)

        main_layout = QHBoxLayout(self)

        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setMaximumWidth(350)
        self.sidebar_tabs.setMinimumWidth(280)

        self._setup_sampling_tab()
        self._setup_crop_tab()

        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)
        self.image_view.double_clicked.connect(self._on_canvas_double_clicked)
        self.image_view.roi_added.connect(self._on_roi_added)

        self.selected_bead_scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen(QColor(0, 255, 255), width=2),
            brush=pg.mkBrush(QColor(0, 255, 255, 100))
        )
        self.image_view._vb.addItem(self.selected_bead_scatter, ignoreBounds=True)
        self.selected_bead_scatter.setVisible(False)

        self.minimap_label = QLabel(self.image_view)
        self.minimap_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.minimap_label.move(10, 10)
        self.minimap_label.setVisible(False)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.preview_label = QLabel("Double-click image to inspect ROI")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.setChecked(self.adjust_contrast)
        self.enhance_contrast_checkbox.stateChanged.connect(self._on_contrast_checkbox_changed)

        self.control_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._setup_editable_controls()

        right_layout.addWidget(self.preview_label)
        right_layout.addWidget(self.enhance_contrast_checkbox)
        right_layout.addWidget(self.image_view)
        right_layout.addLayout(self.control_layout)
        right_layout.addLayout(self.button_layout)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.sidebar_tabs)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([300, 1100])

        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        if self.initial_tab == "crop":
            self.sidebar_tabs.setCurrentIndex(1)
        else:
            self.sidebar_tabs.setCurrentIndex(0)
        self.sidebar_tabs.currentChanged.connect(self._on_tab_changed)

    def _setup_sampling_tab(self):
        sampling_widget = QWidget()
        sampling_layout = QVBoxLayout(sampling_widget)

        self.beads_list = QListWidget()
        self.beads_list.itemClicked.connect(self._on_invalid_bead_selected)

        self.resample_invalids_button = QPushButton("Resample Invalids")
        self.resample_invalids_button.clicked.connect(self._populate_invalid_beads_list)
        self.resample_filtered_button = QPushButton("Resample Filtereds")
        self.resample_filtered_button.clicked.connect(self._populate_filtered_beads_list)

        self.invalid_count_label = QLabel("Invalid Beads: 0; Filtered: 0")
        self._update_invalid_count_label()

        sampling_layout.addWidget(self.invalid_count_label)
        sampling_layout.addWidget(self.beads_list)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.resample_invalids_button)
        btn_layout.addWidget(self.resample_filtered_button)
        sampling_layout.addLayout(btn_layout)

        self.sidebar_tabs.addTab(sampling_widget, "Sampling")
        self._populate_invalid_beads_list()

    def _setup_crop_tab(self):
        crop_widget = QWidget()
        crop_layout = QVBoxLayout(crop_widget)

        tools_group = QGroupBox("ROI Drawing Tools")
        tools_layout = QHBoxLayout(tools_group)

        self.select_tool_btn = QPushButton("Select/Pan")
        self.select_tool_btn.setCheckable(True)
        self.select_tool_btn.setChecked(True)
        self.rect_tool_btn = QPushButton("Rect")
        self.rect_tool_btn.setCheckable(True)
        self.circle_tool_btn = QPushButton("Circle")
        self.circle_tool_btn.setCheckable(True)
        self.poly_tool_btn = QPushButton("Lasso")
        self.poly_tool_btn.setCheckable(True)

        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.select_tool_btn)
        self.tool_group.addButton(self.rect_tool_btn)
        self.tool_group.addButton(self.circle_tool_btn)
        self.tool_group.addButton(self.poly_tool_btn)
        self.tool_group.setExclusive(True)

        tools_layout.addWidget(self.select_tool_btn)
        tools_layout.addWidget(self.rect_tool_btn)
        tools_layout.addWidget(self.circle_tool_btn)
        tools_layout.addWidget(self.poly_tool_btn)

        self.select_tool_btn.clicked.connect(lambda: self._set_draw_mode(DrawMode.SELECT))
        self.rect_tool_btn.clicked.connect(lambda: self._set_draw_mode(DrawMode.RECT))
        self.circle_tool_btn.clicked.connect(lambda: self._set_draw_mode(DrawMode.CIRCLE))
        self.poly_tool_btn.clicked.connect(lambda: self._set_draw_mode(DrawMode.POLY))

        crop_layout.addWidget(tools_group)

        mode_group = QGroupBox("Crop Mode")
        mode_layout = QHBoxLayout(mode_group)
        self.include_radio = QRadioButton("Include Regions")
        self.exclude_radio = QRadioButton("Exclude Regions")
        self.include_radio.setChecked(True)
        mode_layout.addWidget(self.include_radio)
        mode_layout.addWidget(self.exclude_radio)
        self.include_radio.toggled.connect(self._update_selected_beads_count)
        self.exclude_radio.toggled.connect(self._update_selected_beads_count)
        crop_layout.addWidget(mode_group)

        self.show_minimap_checkbox = QCheckBox("Show Minimap")
        self.show_minimap_checkbox.setChecked(True)
        self.show_minimap_checkbox.stateChanged.connect(self._toggle_minimap_visibility)
        crop_layout.addWidget(self.show_minimap_checkbox)

        self.beads_count_label = QLabel("Selected Beads: 0 / 0 (0.0%)")
        self.beads_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.beads_count_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        crop_layout.addWidget(self.beads_count_label)

        roi_group = QGroupBox("Defined Crop Regions")
        roi_layout = QVBoxLayout(roi_group)

        self.roi_list_widget = QListWidget()
        roi_layout.addWidget(self.roi_list_widget)

        list_btn_layout = QHBoxLayout()
        self.delete_roi_btn = QPushButton("Delete Selected")
        self.delete_roi_btn.clicked.connect(self._delete_selected_roi)
        self.clear_rois_btn = QPushButton("Clear All")
        self.clear_rois_btn.clicked.connect(self._clear_all_rois)
        list_btn_layout.addWidget(self.delete_roi_btn)
        list_btn_layout.addWidget(self.clear_rois_btn)
        roi_layout.addLayout(list_btn_layout)

        crop_layout.addWidget(roi_group)

        self.confirm_crop_btn = QPushButton("Confirm Crop")
        self.confirm_crop_btn.clicked.connect(self._validate_and_confirm_crop)
        crop_layout.addWidget(self.confirm_crop_btn)

        self.sidebar_tabs.addTab(crop_widget, "Crop Regions")

    def _update_invalid_count_label(self):
        if self.merged_bead_data is not None:
            invalid_count = len(
                self.merged_bead_data[self.merged_bead_data["Protein name"] == "Invalid"]
            )
            filtered_count = len(
                self.merged_bead_data[self.merged_bead_data["Protein name"] == "Filtered"]
            )
            self.invalid_count_label.setText(
                f"Invalid Beads: {invalid_count}; Filtered: {filtered_count}"
            )

    def _populate_filtered_beads_list(self):
        if self.merged_bead_data is not None:
            filtered_beads = self.merged_bead_data[
                self.merged_bead_data["Protein name"] == "Filtered"
            ]
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
        if self.merged_bead_data is not None:
            invalid_beads = self.merged_bead_data[
                self.merged_bead_data["Protein name"] == "Invalid"
            ]
            sampled_beads = invalid_beads.sample(
                n=min(10, len(invalid_beads)), random_state=None
            )
            self.beads_list.clear()
            for idx, row in sampled_beads.iterrows():
                x, y = int(row["x"]), int(row["y"])
                item_text = f"Bead {idx}: ({x}, {y})"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, (idx, x, y))
                self.beads_list.addItem(item)

    def _on_invalid_bead_selected(self, item):
        data = item.data(Qt.ItemDataRole.UserRole)
        if data:
            idx, x, y = data
            self.x_input.setText(str(x))
            self.y_input.setText(str(y))
            self.inspect_roi(None)

    def _resample_selected_bead(self):
        current_item = self.beads_list.currentItem()
        if current_item:
            data = current_item.data(Qt.ItemDataRole.UserRole)
            if data:
                idx, x, y = data
                QMessageBox.information(
                    self,
                    "Resample Bead",
                    f"Resampling bead {idx} at coordinates ({x}, {y})\n"
                    "This would trigger your resampling algorithm.",
                )

    def _on_contrast_checkbox_changed(self, state):
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self.create_direct_overlay(reset_zoom=False)

    def _setup_editable_controls(self):
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

        h, w = self.target_image.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        self.image_view._vb.setRange(xRange=(x - 50, x + 50), yRange=(y - 50, y + 50))
        if hasattr(self, "selected_bead_scatter"):
            self.selected_bead_scatter.setData(x=[x], y=[y])
            self.selected_bead_scatter.setVisible(True)

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
                x1 = np.clip(x1, 0, w - 1)
                x2 = np.clip(x2, 0, w - 1)
                y1 = np.clip(y1, 0, h - 1)
                y2 = np.clip(y2, 0, h - 1)
                bbox = (x1, y1, x2, y2)
        if self.labeled_image is None or not self.roi_mode_input.isChecked():
            bbox = None
        assert isinstance(x, int) and isinstance(y, int)
        rois = {}
        bright_field_roi = {}
        for key, cycle in (self.cycles or {}).items():
            if cycle.ndim == 3:
                h, w = cycle.shape[1], cycle.shape[2]
            elif cycle.ndim == 2:
                h, w = cycle.shape[0], cycle.shape[1]
            else:
                continue
            x0 = max(0, x - int(radius * scale))
            x1 = min(w, x + int(radius * scale) + 1)
            y0 = max(0, y - int(radius * scale))
            y1 = min(h, y + int(radius * scale) + 1)
            if bbox is not None:
                x0, y0, x1, y1 = expand_bbox(bbox, scale)
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(w, x1)
                y1 = min(h, y1)
            if x0 >= x1 or y0 >= y1:
                continue
            if cycle.ndim == 3:
                roi = cycle[:, y0:y1, x0:x1]
            else:
                roi = cycle[y0:y1, x0:x1]
            rois[key] = roi
            if self.bright_fields is not None and key in self.bright_fields:
                bright_field_roi[key] = self.bright_fields[key][y0:y1, x0:x1]

        popup = ROIGridDisplay(
            rois, (x, y), radius, scale, bbox, output, bright_field_roi
        )
        popup.exec()

    def reset_zoom(self, event=None):
        self.image_view.reset_zoom()
        if event:
            event.accept()

    def _get_background_image(self, contrast):
        if self.bg_images.get(contrast) is None:
            if contrast:
                img_u8 = to_uint8(adjust_contrast(self.target_image.astype(np.float32), 30, 99))
            else:
                img_u8 = to_uint8(self.target_image)
                
            h, w = img_u8.shape
            rgb_image = np.stack([img_u8] * 3, axis=-1)
            
            data_to_use = self.merged_bead_data if self.merged_bead_data is not None else self.beads
            if data_to_use is not None and not data_to_use.empty:
                xs = data_to_use["x"].astype(float).round().astype(int).to_numpy()
                ys = data_to_use["y"].astype(float).round().astype(int).to_numpy()
                valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                xs, ys = xs[valid], ys[valid]
                if "Protein name" in data_to_use.columns:
                    protein_names = data_to_use.loc[valid, "Protein name"].to_numpy()
                    invalid_mask = (protein_names == "Invalid")
                    filtered_mask = (protein_names == "Filtered")
                    valid_mask = ~(invalid_mask | filtered_mask)
                    
                    rgb_image[ys[invalid_mask], xs[invalid_mask]] = BeadColor.INVALID.value
                    rgb_image[ys[filtered_mask], xs[filtered_mask]] = BeadColor.FILTERED.value
                    rgb_image[ys[valid_mask], xs[valid_mask]] = BeadColor.VALID.value
                else:
                    rgb_image[ys, xs] = BeadColor.VALID.value
                        
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = rgb_image
            rgba[:, :, 3] = 255
            self.bg_images[contrast] = rgba
            
        return self.bg_images[contrast]

    def _update_minimap(self):
        contrast = self.adjust_contrast
        if self._minimap_cache.get(contrast) is not None:
            pixmap, size = self._minimap_cache[contrast]
            self.minimap_label.setPixmap(pixmap)
            self.minimap_label.setFixedSize(size)
            return

        minimap_size = 200
        if contrast:
            img_u8 = to_uint8(adjust_contrast(self.target_image.astype(np.float32), 30, 99))
        else:
            img_u8 = to_uint8(self.target_image)
            
        h, w = img_u8.shape
        qimg = QImage(img_u8.data, w, h, w, QImage.Format.Format_Grayscale8)
        scaled_qimg = qimg.scaled(minimap_size, minimap_size, Qt.AspectRatioMode.KeepAspectRatio)
        rgb_qimg = scaled_qimg.convertToFormat(QImage.Format.Format_RGB32)
        
        painter = QPainter(rgb_qimg)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        data_to_use = self.merged_bead_data if self.merged_bead_data is not None else self.beads
        if data_to_use is not None and not data_to_use.empty:
            xs = data_to_use["x"].astype(float).to_numpy()
            ys = data_to_use["y"].astype(float).to_numpy()
            
            scaled_w = rgb_qimg.width()
            scaled_h = rgb_qimg.height()
            
            x_scaled = (xs / w) * scaled_w
            y_scaled = (ys / h) * scaled_h
            
            if "Protein name" in data_to_use.columns:
                protein_names = data_to_use["Protein name"].to_numpy()
                invalid_mask = (protein_names == "Invalid")
                filtered_mask = (protein_names == "Filtered")
                valid_mask = ~(invalid_mask | filtered_mask)
                
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(BeadColor.VALID.qcolor)
                for x, y in zip(x_scaled[valid_mask], y_scaled[valid_mask]):
                    painter.drawEllipse(QPointF(x, y), 2, 2)
                    
                painter.setBrush(BeadColor.INVALID.qcolor)
                for x, y in zip(x_scaled[invalid_mask], y_scaled[invalid_mask]):
                    painter.drawEllipse(QPointF(x, y), 2, 2)
                    
                painter.setBrush(BeadColor.FILTERED.qcolor)
                for x, y in zip(x_scaled[filtered_mask], y_scaled[filtered_mask]):
                    painter.drawEllipse(QPointF(x, y), 2, 2)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(BeadColor.VALID.qcolor)
                for x, y in zip(x_scaled, y_scaled):
                    painter.drawEllipse(QPointF(x, y), 2, 2)
                    
        painter.end()
        pixmap = QPixmap.fromImage(rgb_qimg)
        size = rgb_qimg.size()
        self._minimap_cache[contrast] = (pixmap, size)
        self.minimap_label.setPixmap(pixmap)
        self.minimap_label.setFixedSize(size)

    def create_direct_overlay(self, reset_zoom=True):
        is_crop_tab = (self.sidebar_tabs.currentIndex() == 1)
        show_rois = is_crop_tab
        if self.labeled_image is not None and hasattr(self, "roi_mode_input") and self.roi_mode_input.isChecked():
            self.image_view.set_images([self.labeled_image], [1.0], reset_zoom=reset_zoom)
            self.minimap_label.setVisible(False)
            show_rois = False
            for roi_info in self.created_rois:
                roi_info["roi_item"].setVisible(show_rois)
            return

        bg_img = self._get_background_image(self.adjust_contrast)
        self.image_view.set_images([bg_img], [1.0], reset_zoom=reset_zoom)

        show_minimap = self.show_minimap_checkbox.isChecked() if hasattr(self, "show_minimap_checkbox") else True
        
        if is_crop_tab and show_minimap:
            self._update_minimap()
            self.minimap_label.setVisible(True)
        else:
            self.minimap_label.setVisible(False)

        for roi_info in self.created_rois:
            roi_info["roi_item"].setVisible(show_rois)

    def _on_canvas_double_clicked(self, view_pos):
        if self.image_view.draw_mode == DrawMode.SELECT:
            x, y = int(view_pos.x()), int(view_pos.y())
            self.x_input.setText(str(x))
            self.y_input.setText(str(y))
            self.inspect_roi(None)

    def _on_tab_changed(self, index):
        self.create_direct_overlay(reset_zoom=False)

    def _toggle_minimap_visibility(self, state):
        is_crop_tab = (self.sidebar_tabs.currentIndex() == 1)
        show_minimap = self.show_minimap_checkbox.isChecked()
        self.minimap_label.setVisible(is_crop_tab and show_minimap)

    def _set_draw_mode(self, mode):
        self.image_view.draw_mode = mode
        if mode == DrawMode.SELECT:
            self.image_view._vb.setMouseMode(pg.ViewBox.PanMode)
            self.image_view.viewport().unsetCursor()
        else:
            self.image_view._vb.setMouseMode(pg.ViewBox.RectMode)
            self.image_view.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _on_roi_added(self, roi_item):
        roi_type = ""
        if isinstance(roi_item, CropRectROI):
            roi_type = "rect"
            name = f"Rect ROI {len([r for r in self.created_rois if r['type'] == 'rect']) + 1}"
            roi_item.sigRegionChangeFinished.connect(self._update_selected_beads_count)
        elif isinstance(roi_item, CropCircleROI):
            roi_type = "circle"
            name = f"Circle ROI {len([r for r in self.created_rois if r['type'] == 'circle']) + 1}"
            roi_item.sigRegionChangeFinished.connect(self._update_selected_beads_count)
        elif isinstance(roi_item, CropPolyROI):
            roi_type = "polygon"
            name = f"Poly ROI {len([r for r in self.created_rois if r['type'] == 'polygon']) + 1}"
            roi_item.sigRegionChangeFinished.connect(self._update_selected_beads_count)

        self.created_rois.append({
            "roi_item": roi_item,
            "type": roi_type,
            "name": name
        })
        self.roi_list_widget.addItem(name)
        self._update_selected_beads_count()

    def _delete_selected_roi(self):
        current_row = self.roi_list_widget.currentRow()
        if current_row < 0:
            return
        roi_info = self.created_rois.pop(current_row)
        self.image_view._vb.removeItem(roi_info["roi_item"])
        self.roi_list_widget.takeItem(current_row)
        self._update_selected_beads_count()

    def _clear_all_rois(self):
        for roi_info in self.created_rois:
            self.image_view._vb.removeItem(roi_info["roi_item"])
        self.created_rois.clear()
        self.roi_list_widget.clear()
        self._update_selected_beads_count()

    def _get_roi_definitions(self):
        rois = []
        for info in self.created_rois:
            item = info["roi_item"]
            t = info["type"]
            if t == "rect":
                pos = item.pos()
                size = item.size()
                rois.append({
                    "type": "rect",
                    "x1": float(pos[0]),
                    "y1": float(pos[1]),
                    "x2": float(pos[0] + size[0]),
                    "y2": float(pos[1] + size[1])
                })
            elif t == "circle":
                pos = item.pos()
                size = item.size()
                rois.append({
                    "type": "circle",
                    "cx": float(pos[0] + size[0]/2),
                    "cy": float(pos[1] + size[1]/2),
                    "r": float(size[0]/2)
                })
            elif t == "polygon":
                pts = []
                for pt in item.points:
                    pt_scene = item.mapToScene(pt)
                    pts.append((float(pt_scene.x()), float(pt_scene.y())))
                rois.append({
                    "type": "polygon",
                    "points": pts
                })
        return rois

    def _update_selected_beads_count(self):
        if self.beads is None or self.beads.empty:
            self.beads_count_label.setText("Selected Beads: 0 / 0 (0.0%)")
            return
        rois = self._get_roi_definitions()
        if not rois:
            self.beads_count_label.setText(f"Selected Beads: {len(self.beads)} / {len(self.beads)} (100.0%)")
            return

        in_any_roi = np.zeros(len(self.beads), dtype=bool)
        for roi in rois:
            if roi["type"] == "rect":
                x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                in_roi = (
                    (self.beads["x"] >= x1)
                    & (self.beads["x"] < x2)
                    & (self.beads["y"] >= y1)
                    & (self.beads["y"] < y2)
                )
            elif roi["type"] == "circle":
                cx, cy, r = roi["cx"], roi["cy"], roi["r"]
                in_roi = ((self.beads["x"] - cx) ** 2 + (self.beads["y"] - cy) ** 2) <= r ** 2
            elif roi["type"] == "polygon":
                pts = roi["points"]
                from matplotlib.path import Path
                path = Path(pts)
                in_roi = path.contains_points(self.beads[["x", "y"]].to_numpy())
            else:
                in_roi = np.zeros(len(self.beads), dtype=bool)
            in_any_roi = in_any_roi | in_roi

        is_exclude = self.exclude_radio.isChecked() if hasattr(self, "exclude_radio") else False
        if is_exclude:
            keep = ~in_any_roi
        else:
            keep = in_any_roi

        selected_count = np.sum(keep)
        total_count = len(self.beads)
        pct = (selected_count / total_count * 100) if total_count > 0 else 0.0
        self.beads_count_label.setText(f"Selected Beads: {selected_count} / {total_count} ({pct:.1f}%)")

    def _validate_and_confirm_crop(self):
        rois = self._get_roi_definitions()
        if not rois:
            QMessageBox.warning(
                self,
                "No Regions Defined",
                "Please define at least one crop region before confirming.",
            )
            return

        crop_mode = "exclude" if self.exclude_radio.isChecked() else "include"
        self.bead_crop_confirmed.emit(rois, crop_mode)

        if self.file_item is not None:
            self.beads = self.file_item.beads
            merge_columns = []
            if self.beads is not None:
                merge_columns = [
                    col
                    for col in self.beads.columns
                    if col.startswith("cy") and col[2:].isdigit()
                ]
            if self.beads is not None and self.protein_profile is not None and merge_columns:
                self.merged_bead_data = merge_bead_data_with_protein_profile(
                    self.beads.copy(), self.protein_profile.copy(), merge_columns
                )

            self._populate_invalid_beads_list()
            self._update_invalid_count_label()
            self.bg_images[False] = None
            self.bg_images[True] = None
            self._minimap_cache = {False: None, True: None}
            self.create_direct_overlay(reset_zoom=False)

        QMessageBox.information(
            self,
            "Crop Confirmed",
            "Bead crop regions have been successfully saved and applied.",
        )

    def _load_existing_rois(self, rois: list[dict]):
        for roi in rois:
            roi_type = roi["type"]
            if roi_type == "rect":
                x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                roi_item = CropRectROI([x1, y1], [x2 - x1, y2 - y1])
                name = f"Rect ROI {len([r for r in self.created_rois if r['type'] == 'rect']) + 1}"
                roi_item.sigRegionChangeFinished.connect(self._update_selected_beads_count)
            elif roi_type == "circle":
                cx, cy, r = roi["cx"], roi["cy"], roi["r"]
                roi_item = CropCircleROI([cx - r, cy - r], [r * 2, r * 2])
                name = f"Circle ROI {len([r for r in self.created_rois if r['type'] == 'circle']) + 1}"
                roi_item.sigRegionChangeFinished.connect(self._update_selected_beads_count)
            elif roi_type == "polygon":
                points = roi["points"]
                roi_item = CropPolyROI()
                for pt in points:
                    roi_item.add_point(QPointF(pt[0], pt[1]))
                roi_item.complete()
                name = f"Poly ROI {len([r for r in self.created_rois if r['type'] == 'polygon']) + 1}"
            else:
                continue

            self.image_view._vb.addItem(roi_item)
            self.created_rois.append({
                "roi_item": roi_item,
                "type": roi_type,
                "name": name
            })
            self.roi_list_widget.addItem(name)
        self._update_selected_beads_count()
        self.create_direct_overlay(reset_zoom=False)


class ROIGridDisplay(QDialog):
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
        cy0_roi = rois.get("cy0", np.array([]))
        if cy0_roi.ndim == 3:
            num_channels = cy0_roi.shape[0]
        elif cy0_roi.ndim == 2:
            num_channels = 1
        else:
            num_channels = 0
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
            out_label = QLabel("Output")
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


from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QPixmap
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


ROI_Inspector = ROIInspector
ROI_Grid_Display = ROIGridDisplay
