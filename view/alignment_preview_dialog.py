import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import (
    QImage,
    QKeyEvent,
    QPainter,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from utils import adjust_contrast, to_uint8


class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.target_item = QGraphicsPixmapItem()
        self.moving_items: list[QGraphicsPixmapItem] = []
        self._scene.addItem(self.target_item)
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def set_images(self, target_pixmap: QPixmap, moving_pixmaps: list[QPixmap]):
        self.target_item.setPixmap(target_pixmap)
        self.target_item.setZValue(0)

        for item in self.moving_items:
            self._scene.removeItem(item)
        self.moving_items.clear()

        for pixmap in moving_pixmaps:
            item = QGraphicsPixmapItem()
            item.setOpacity(0.5)
            item.setPixmap(pixmap)
            self.moving_items.append(item)
            self._scene.addItem(item)

        QTimer.singleShot(0, self.reset_zoom)

    def toggle_moving_item_visibility(self, index: int, visible: bool):
        if 0 <= index < len(self.moving_items):
            self.moving_items[index].setVisible(visible)

    def reset_zoom(self):
        if not self.scene():
            return
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
            zoom_factor = 1.15
        else:
            zoom_factor = 1 / 1.15

        self.scale(zoom_factor, zoom_factor)


class AlignmentPreviewDialog(QDialog):
    def __init__(
        self,
        target_image: np.ndarray,
        moving_images: list[np.ndarray],
        can_edit: bool = False,
        can_emit: bool = False,
    ):
        super().__init__(None)

        self.target_image = target_image.copy()
        self.moving_images = [img.copy() for img in moving_images]
        self.can_edit = can_edit
        self.can_emit = can_emit
        self.adjust_contrast = True
        self.result_accepted = False

        self._setup_ui()
        self.create_direct_overlay()
        self.image_view.mouseDoubleClickEvent = self.reset_zoom

    def _setup_ui(self):
        self.setWindowTitle("Alignment Preview")
        self.resize(1000, 800)
        main_layout = QVBoxLayout(self)

        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.setChecked(self.adjust_contrast)
        self.enhance_contrast_checkbox.stateChanged.connect(
            self._on_contrast_checkbox_changed
        )

        instruction_text = "Mouse wheel: zoom, Drag: pan, Double-click: reset view"
        self.preview_label = QLabel(
            f"Red = Target, Other colors = Aligned | {instruction_text}"
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)

        self.control_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()

        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if self.can_emit:
            self._setup_confirm_cancel_buttons()
        else:
            self._setup_view_only_controls()

        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.enhance_contrast_checkbox)
        main_layout.addWidget(self.image_view)

        self.visibility_groupbox = QGroupBox("Moving Layers")
        visibility_layout = QVBoxLayout()
        self.visibility_checkboxes = []
        for i in range(len(self.moving_images)):
            checkbox = QCheckBox(f"Moving Image {i + 1}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                lambda state, index=i: self.image_view.toggle_moving_item_visibility(
                    index, state == Qt.CheckState.Checked.value
                )
            )
            self.visibility_checkboxes.append(checkbox)
            visibility_layout.addWidget(checkbox)
        self.visibility_groupbox.setLayout(visibility_layout)
        self.control_layout.addWidget(self.visibility_groupbox)
        self.control_layout.addStretch()

        main_layout.addLayout(self.control_layout)
        main_layout.addLayout(self.button_layout)
        self.setLayout(main_layout)

    def _on_contrast_checkbox_changed(self, state):
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self.create_direct_overlay()

    def _setup_confirm_cancel_buttons(self):
        self.confirm_button = QPushButton("Confirm Alignment")
        self.confirm_button.clicked.connect(self.accept_alignment)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.confirm_button)
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addStretch()

    def _setup_view_only_controls(self):
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.close_button)
        self.button_layout.addStretch()

    def accept_alignment(self):
        self.result_accepted = True
        self.accept()

    def keyPressEvent(self, event: QKeyEvent):
        if not self.can_edit:
            super().keyPressEvent(event)
            return
        # Editable key events would go here

    def create_direct_overlay(self):
        target_img = self.target_image
        if self.target_image.ndim == 3:
            target_img = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)

        target_gray = to_uint8(target_img)
        if self.adjust_contrast:
            target_gray = to_uint8(
                adjust_contrast(target_gray.astype(np.float32), 30, 99)
            )
        target_pixmap = colorize_grayscale(target_gray, "red")

        moving_pixmaps = []
        colors = ["green", "blue", "magenta", "cyan", "yellow"]
        for i, moving_img in enumerate(self.moving_images):
            if moving_img.ndim == 3:
                moving_img = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)

            moving_gray = to_uint8(moving_img)

            if self.adjust_contrast:
                moving_gray = to_uint8(
                    adjust_contrast(moving_gray.astype(np.float32), 30, 99)
                )

            color = colors[i % len(colors)]
            moving_pixmap = colorize_grayscale(moving_gray, color)
            moving_pixmaps.append(moving_pixmap)

        self.image_view.set_images(target_pixmap, moving_pixmaps)

    def reset_zoom(self, event=None):
        self.image_view.reset_zoom()
        if event:
            event.accept()


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