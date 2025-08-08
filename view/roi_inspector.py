import math

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDoubleValidator,
    QImage,
    QIntValidator,
    QKeyEvent,
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QGridLayout
    
)
import re
from utils import adjust_contrast, to_uint8
import pandas as pd

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


class ROI_Inspector(QDialog):
    show_bead_signal = pyqtSignal(np.ndarray)

    def __init__(self, snapshot_data: dict):
        super().__init__(None)

        self.target_image = snapshot_data["bf_image"].copy()
        self.beads = snapshot_data.get("beads", None)
        self.cycles = snapshot_data.get("cycles",None)
        self.bboxs = snapshot_data.get("bboxs",None)
        self.labeled_image = snapshot_data.get("labeled_image",None)
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
        self.resize(1000, 800)
        main_layout = QVBoxLayout(self)
        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.setChecked(self.adjust_contrast)
        self.enhance_contrast_checkbox.stateChanged.connect(
            self._on_contrast_checkbox_changed
        )
        instruction_text = "Double-click image to inspect ROI"
        self.preview_label = QLabel(instruction_text)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_view = ZoomableImageView(self)
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
        self.setLayout(main_layout)

    def _on_contrast_checkbox_changed(self, state):
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self.create_direct_overlay()

    def _setup_editable_controls(self):
        """Create UI controls for when manual editing is enabled."""

        trans_group = QGroupBox("ROI Center")
        trans_layout = QHBoxLayout()
        int_validator = NullableIntValidator(-99999, 99999)

        self.dx_input = QLineEdit("0")
        self.dx_input.setValidator(int_validator)
        self.dx_input.setFixedWidth(50)

        self.dy_input = QLineEdit("0")
        self.dy_input.setValidator(int_validator)
        self.dy_input.setFixedWidth(50)

        self.apply_trans_button = QPushButton("Inspect")
        trans_layout.addWidget(QLabel("x:"))
        trans_layout.addWidget(self.dx_input)
        trans_layout.addWidget(QLabel("y:"))
        trans_layout.addWidget(self.dy_input)
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
        # self.dx_input.returnPressed.connect(self.apply_manual_translation)
        # self.dy_input.returnPressed.connect(self.apply_manual_translation)
        # self.rotation_input.returnPressed.connect(self.apply_rotation)

        self.reset_button = QPushButton("Reset Zoom")
        self.reset_button.clicked.connect(self.reset_zoom)

        self.control_layout.addWidget(trans_group)
        self.control_layout.addWidget(radius_group)
        self.control_layout.addWidget(scale_group)
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
                xtext = self.dx_input.text()
                ytext = self.dy_input.text()
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
            self.dx_input.setText(str(x))
            self.dy_input.setText(str(y))
            print(f"Double-click at scene position ({x}, {y})")
            # MOVE TO CENTER

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
        assert isinstance(self.beads,pd.DataFrame)
        output = self.beads.query(f"x=={x} & y=={y}")
        bbox = None
        idx = None
        if not output.empty:
            idx = output.index[0]
            if self.bboxs is not None:
                bbox = self.bboxs.loc[idx]
                h,w = self.target_image.shape
                y1, x1, y2, x2 = map(int, re.findall(r"-?\d+", bbox))
                # Clip to image bounds
                x1 = np.clip(x1, 0, w - 1)
                x2 = np.clip(x2, 0, w - 1)
                y1 = np.clip(y1, 0, h - 1)
                y2 = np.clip(y2, 0, h - 1)
                bbox = (x1,y1,x2,y2)
        if not self.roi_mode_input.isChecked():
            bbox = None
        assert isinstance(x, int) and isinstance(y, int)
        print(f"Inspecting ROI at ({x}, {y}) with radius {radius} and scale {scale}")
        rois = {}
        for key, cycle in (self.cycles or {}).items():
            if cycle.ndim == 3:
                h, w = cycle.shape[1], cycle.shape[2]
            elif cycle.ndim == 2:
                h, w = cycle.shape[0], cycle.shape[1]
            else:
                print(f"Cycle {key} has unexpected shape {cycle.shape}")
                continue
            x0 = max(0, x - int(radius * scale))
            x1 = min(w, x + int(radius * scale))
            y0 = max(0, y - int(radius * scale))
            y1 = min(h, y + int(radius * scale))
            if bbox is not None:
                x0,y0,x1,y1 = expand_bbox(bbox,scale)
            if cycle.ndim == 3:
                roi = cycle[:, y0:y1, x0:x1]
            else:
                roi = cycle[y0:y1, x0:x1]
            rois[key] = roi
            print(
                f"Cycle {key}: extracted ROI shape {roi.shape} at ({x0}:{x1}, {y0}:{y1})"
            )
                

        
                

        popup = ROI_Grid_Display(rois, (x, y), radius, scale, bbox, output)
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
        if not self.roi_mode_input.isChecked():
            target_gray = to_uint8(self.target_image)

            if self.adjust_contrast:
                target_gray = to_uint8(
                    adjust_contrast(target_gray.astype(np.float32), 30, 99)
                )

            # Convert grayscale to RGB
            rgb_image = np.stack([target_gray] * 3, axis=-1)  # Shape: (H, W, 3)
        else:
            rgb_image = self.labeled_image
        assert isinstance(rgb_image,np.ndarray)
        h, w = rgb_image.shape[:-1]

        # Draw red centers
        if self.beads is not None and not self.beads.empty:
            xs = self.beads["x"].astype(int).to_numpy()
            ys = self.beads["y"].astype(int).to_numpy()

            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs, ys = xs[valid], ys[valid]

            rgb_image[ys, xs] = [255, 0, 0]  # RE
        # for bbox_str in self.bboxs:
        #     # Parse numbers from string "(x1, y1, x2, y2)"
        #     y1, x1, y2, x2 = map(int, re.findall(r"-?\d+", bbox_str))

        #     # Clip to image bounds
        #     x1 = np.clip(x1, 0, w - 1)
        #     x2 = np.clip(x2, 0, w - 1)
        #     y1 = np.clip(y1, 0, h - 1)
        #     y2 = np.clip(y2, 0, h - 1)

        #     # Draw rectangle edges in white
        #     rgb_image[y1, x1:x2+1] = [255, 255, 255]  # Top
        #     rgb_image[y2, x1:x2+1] = [255, 255, 255]  # Bottom
        #     rgb_image[y1:y2+1, x1] = [255, 255, 255]  # Left
        #     rgb_image[y1:y2+1, x2] = [255, 255, 255]  # Right
            # draw bounding box white outline
        # self.bbox, same length as self.beads, indexed same

        # Convert to QImage and show
        rgb_image = np.ascontiguousarray(rgb_image)
        qimage = QImage(rgb_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        target_pixmap = QPixmap.fromImage(qimage)

        self.image_view.set_images(target_pixmap)


class ROI_Grid_Display(QDialog):
    def __init__(self, rois: dict, center: tuple, radius: int, scale: float, bbox, output:pd.DataFrame|None=None):
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
        num_channels = len(rois.get("cy0",np.array([])))
        for i in range(len(rois)):
            cycle_label = QLabel(f"Cycle {i}")
            grid_layout.addWidget(cycle_label,i+1,0)
        for i in range(num_channels):
            channel_label = QLabel(f"Channel {i}")
            grid_layout.addWidget(channel_label,0,i+1)
        if output is not None and len(output):
            out_label = QLabel(f"Output")
            grid_layout.addWidget(out_label,0,num_channels+1)
        
        for key, roi in rois.items():
            col = 1
            if roi.ndim == 3:
                # Multi-channel
                for c in range(roi.shape[0]):
                    roi_colorized = colorize_grayscale(
                        to_uint8(roi[c]), c
                    )
                    roi_label = OverlayLabel(roi_colorized.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio),bbox)
                    grid_layout.addWidget(roi_label,row,col)
                    col+=1
            elif roi.ndim == 2:
                # Single channel
                pixmap = colorize_grayscale(to_uint8(roi), 0)
                pixmap_label = QLabel()
                pixmap_label.setPixmap(
                    pixmap.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio)
                )
            if output is not None and len(output):
                try:
                    pred = output[key]
                except:
                    pred = None
                if pred is None:
                    pred= "N/A"
                else:
                    pred= str(pred.iloc[0])
                output_label = QLabel(pred)
                grid_layout.addWidget(output_label,row,col)
            row+=1
            
        layout.addLayout(grid_layout,6)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        self.setLayout(layout)

from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QRectF

class OverlayLabel(QLabel):
    def __init__(self, pixmap, bbox, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        # self.setFixedSize(pixmap.size())

        if bbox is not None:
            self.rect_to_draw = bbox_to_qrectf(bbox)
        else:
            self.rect_to_draw = None

    def paintEvent(self, event):
        super().paintEvent(event) 
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(QColor('white'))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        if self.rect_to_draw is not None:
            painter.drawRect(self.rect_to_draw)

        painter.end()

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

colors = [(27,158,119),(217,95,2),(117,112,179),(231,41,138)]
def colorize_grayscale(gray_img: np.ndarray, color_indx:int) -> QPixmap:
    """Colorize grayscale image and make black pixels fully transparent."""
    h, w = gray_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    color = colors[color_indx]
    for i,v in enumerate(color):
        rgba[:,:,i] = v

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
