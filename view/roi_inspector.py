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
)

from utils import adjust_contrast, to_uint8


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
        self.cycles = snapshot_data.get("cycles", None)
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

        self.apply_trans_button.clicked.connect(self.inspect_roi)
        # self.dx_input.returnPressed.connect(self.apply_manual_translation)
        # self.dy_input.returnPressed.connect(self.apply_manual_translation)
        # self.rotation_input.returnPressed.connect(self.apply_rotation)

        self.reset_button = QPushButton("Reset Zoom")
        self.reset_button.clicked.connect(self.reset_zoom)

        self.control_layout.addWidget(trans_group)
        self.control_layout.addWidget(radius_group)
        self.control_layout.addWidget(scale_group)
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
            if cycle.ndim == 3:
                roi = cycle[:, y0:y1, x0:x1]
            else:
                roi = cycle[y0:y1, x0:x1]
            rois[key] = roi
            print(
                f"Cycle {key}: extracted ROI shape {roi.shape} at ({x0}:{x1}, {y0}:{y1})"
            )
        popup = ROI_Grid_Display(rois, (x, y), radius, scale)
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
        target_gray = to_uint8(self.target_image)
        h, w = target_gray.shape

        if self.adjust_contrast:
            target_gray = to_uint8(
                adjust_contrast(target_gray.astype(np.float32), 30, 99)
            )

        # Convert grayscale to RGB
        rgb_image = np.stack([target_gray] * 3, axis=-1)  # Shape: (H, W, 3)

        # Draw red centers
        if self.beads is not None and not self.beads.empty:
            xs = self.beads["x"].astype(int).to_numpy()
            ys = self.beads["y"].astype(int).to_numpy()

            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs, ys = xs[valid], ys[valid]

            rgb_image[ys, xs] = [255, 0, 0]  # RED

        # Convert to QImage and show
        rgb_image = np.ascontiguousarray(rgb_image)
        qimage = QImage(rgb_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        target_pixmap = QPixmap.fromImage(qimage)

        self.image_view.set_images(target_pixmap)


class ROI_Grid_Display(QDialog):
    def __init__(self, rois: dict, center: tuple, radius: int, scale: float):
        super().__init__(None)
        self.setWindowTitle("ROI Grid Display")
        self.resize(800, 600)
        layout = QVBoxLayout(self)

        info_label = QLabel(
            f"Center: {center}, Radius: {radius}, Scale: {scale:.2f}. "
            f"Showing {len(rois)} cycles."
        )
        layout.addWidget(info_label)

        grid_layout = QVBoxLayout()
        for key, roi in rois.items():
            if roi.ndim == 3:
                # Multi-channel
                channels = []
                for c in range(roi.shape[0]):
                    channels.append(
                        colorize_grayscale(
                            to_uint8(roi[c]), ["red", "green", "blue"][c % 3]
                        )
                    )
                combined = QPixmap(channels[0].size())
                combined.fill(Qt.GlobalColor.transparent)
                painter = QPainter(combined)
                for ch in channels:
                    painter.drawPixmap(0, 0, ch)
                painter.end()
                label = QLabel(f"Cycle {key} (multi-channel)")
                pixmap_label = QLabel()
                pixmap_label.setPixmap(
                    combined.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
                )
            elif roi.ndim == 2:
                # Single channel
                pixmap = colorize_grayscale(to_uint8(roi), "green")
                label = QLabel(f"Cycle {key} (single-channel)")
                pixmap_label = QLabel()
                pixmap_label.setPixmap(
                    pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
                )
            else:
                continue
            row_layout = QHBoxLayout()
            row_layout.addWidget(label)
            row_layout.addWidget(pixmap_label)
            grid_layout.addLayout(row_layout)

        layout.addLayout(grid_layout)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        self.setLayout(layout)


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


def colorize_grayscale(gray_img: np.ndarray, color: str) -> QPixmap:
    """Colorize grayscale image and make black pixels fully transparent."""
    h, w = gray_img.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if color == "red":
        rgba[:, :, 0] = gray_img  # R
    elif color == "green":
        rgba[:, :, 1] = gray_img  # G
    elif color == "blue":
        rgba[:, :, 2] = gray_img  # B

    # Make black (value 0) transparent
    mask = gray_img > 0
    rgba[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha

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
