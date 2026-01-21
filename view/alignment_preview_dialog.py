import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QDoubleValidator,
    QImage,
    QIntValidator,
    QKeyEvent,
    QPainter,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
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
    moving_images_changed = pyqtSignal(list)
    transformation_matrices = pyqtSignal(list)

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
        self.original_moving_images = [img.copy() for img in moving_images]
        self.can_edit = can_edit
        self.can_emit = can_emit
        self.adjust_contrast = True
        self.result_accepted = False
        self.selected_moving_index = 0 if moving_images else -1
        self.move_step = 1

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

        instruction_text = (
            "Arrow keys/Inputs: move, Mouse wheel: zoom, Drag: pan, Double-click: reset view"
            if self.can_edit
            else "Mouse wheel: zoom, Drag: pan, Double-click: reset view"
        )
        self.preview_label = QLabel(
            f"Red = Target, Other colors = Aligned | {instruction_text}"
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.offset_label = QLabel()
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offset_label.setVisible(self.can_edit)

        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)

        self.control_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()

        if self.can_edit:
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._setup_editable_controls()
        else:
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            if self.can_emit:
                self._setup_confirm_cancel_buttons()
            else:
                self._setup_view_only_controls()

        main_layout.addWidget(self.preview_label)
        main_layout.addWidget(self.offset_label)
        main_layout.addWidget(self.enhance_contrast_checkbox)
        main_layout.addWidget(self.image_view)

        self.visibility_groupbox = QGroupBox("Moving Layers")
        visibility_layout = QVBoxLayout()
        self.visibility_checkboxes = []
        self.selection_buttons = []
        self.button_group = QButtonGroup(self)

        for i in range(len(self.moving_images)):
            layer_layout = QHBoxLayout()

            if self.can_edit:
                radio = QRadioButton()
                radio.setChecked(i == self.selected_moving_index)
                radio.toggled.connect(lambda checked, index=i: self._on_layer_selected(index) if checked else None)
                self.selection_buttons.append(radio)
                self.button_group.addButton(radio)
                layer_layout.addWidget(radio)

            checkbox = QCheckBox(f"Moving Image {i + 1}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                lambda state, index=i: self._on_visibility_changed(index, state)
            )
            self.visibility_checkboxes.append(checkbox)
            layer_layout.addWidget(checkbox)
            visibility_layout.addLayout(layer_layout)

        self.visibility_groupbox.setLayout(visibility_layout)
        self.control_layout.addWidget(self.visibility_groupbox)
        self.control_layout.addStretch()

        main_layout.addLayout(self.control_layout)
        main_layout.addLayout(self.button_layout)
        self.setLayout(main_layout)

        if self.can_edit:
            self.update_offset_label()

    def _on_contrast_checkbox_changed(self, state):
        self.adjust_contrast = self.enhance_contrast_checkbox.isChecked()
        self.create_direct_overlay()

    def _on_layer_selected(self, index: int):
        """Called when a different moving layer is selected for editing."""
        self.selected_moving_index = index
        self.update_offset_label()

    def _on_visibility_changed(self, index: int, state):
        """Called when a moving layer visibility checkbox is toggled."""
        visible = state == Qt.CheckState.Checked.value
        self.image_view.toggle_moving_item_visibility(index, visible)

        # If editing is enabled and this layer is hidden, deselect it
        if self.can_edit and not visible and index == self.selected_moving_index:
            # Try to select another visible layer
            for i in range(len(self.moving_images)):
                if i != index and self.visibility_checkboxes[i].isChecked():
                    self.selection_buttons[i].setChecked(True)
                    break

    def _setup_editable_controls(self):
        """Create UI controls for when manual editing is enabled."""
        trans_group = QGroupBox("Translate (Display Pixels)")
        trans_layout = QHBoxLayout()
        int_validator = NullableIntValidator(-99999, 99999)

        self.dx_input = QLineEdit("0")
        self.dx_input.setValidator(int_validator)
        self.dx_input.setFixedWidth(50)

        self.dy_input = QLineEdit("0")
        self.dy_input.setValidator(int_validator)
        self.dy_input.setFixedWidth(50)

        self.apply_trans_button = QPushButton("Apply")
        trans_layout.addWidget(QLabel("dx:"))
        trans_layout.addWidget(self.dx_input)
        trans_layout.addWidget(QLabel("dy:"))
        trans_layout.addWidget(self.dy_input)
        trans_layout.addWidget(self.apply_trans_button)
        trans_group.setLayout(trans_layout)

        rot_group = QGroupBox("Rotate (°)")
        rot_layout = QHBoxLayout()
        self.rotation_input = QLineEdit()
        self.rotation_input.setPlaceholderText("Angle")
        self.rotation_input.setValidator(QDoubleValidator(-360.0, 360.0, 6))
        self.rotate_button = QPushButton("Apply")
        rot_layout.addWidget(self.rotation_input)
        rot_layout.addWidget(self.rotate_button)
        rot_group.setLayout(rot_layout)

        scale_group = QGroupBox("Scale")
        scale_layout = QHBoxLayout()
        self.scale_input = QLineEdit()
        self.scale_input.setPlaceholderText("1.0")
        self.scale_input.setValidator(QDoubleValidator(0.000001, 10000, 6))
        self.scale_button = QPushButton("Apply")
        scale_layout.addWidget(self.scale_input)
        scale_layout.addWidget(self.scale_button)
        scale_group.setLayout(scale_layout)

        flip_group = QGroupBox("Flip")
        flip_layout = QHBoxLayout()
        self.flip_horizontal_btn = QPushButton("Flip Horizontal")
        self.flip_vertical_btn = QPushButton("Flip Vertical")
        flip_layout.addWidget(self.flip_horizontal_btn)
        flip_layout.addWidget(self.flip_vertical_btn)
        flip_group.setLayout(flip_layout)

        self.apply_trans_button.clicked.connect(self.apply_manual_translation)
        self.rotate_button.clicked.connect(self.apply_rotation)
        self.scale_button.clicked.connect(self.apply_scale)
        self.flip_horizontal_btn.clicked.connect(self.apply_flip_horizontal)
        self.flip_vertical_btn.clicked.connect(self.apply_flip_vertical)

        self.reset_button = QPushButton("Reset Transformations")
        self.reset_button.clicked.connect(self.reset_transformations)

        self.control_layout.addWidget(trans_group)
        self.control_layout.addWidget(rot_group)
        self.control_layout.addWidget(scale_group)
        self.control_layout.addWidget(flip_group)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.reset_button)
        self._setup_confirm_cancel_buttons()

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

    def apply_manual_translation(self):
        """Applies translation based on the dx/dy input fields to selected moving image."""
        if self.selected_moving_index < 0 or self.selected_moving_index >= len(self.moving_images):
            return

        if not self.visibility_checkboxes[self.selected_moving_index].isChecked():
            QMessageBox.warning(
                self,
                "Layer Hidden",
                "Cannot edit a hidden layer. Please make the layer visible first.",
            )
            return

        try:
            xtext = self.dx_input.text()
            ytext = self.dy_input.text()
            if xtext == "":
                xtext = "0"
            if ytext == "":
                ytext = "0"
            dx = int(xtext)
            dy = int(ytext)
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid integer values for dx and dy.",
            )
            return

        if dx == 0 and dy == 0:
            return

        self.move_aligned_image(dx, dy)

    def apply_rotation(self):
        """Apply rotation to selected moving image."""
        if self.selected_moving_index < 0 or self.selected_moving_index >= len(self.moving_images):
            return

        if not self.visibility_checkboxes[self.selected_moving_index].isChecked():
            QMessageBox.warning(
                self,
                "Layer Hidden",
                "Cannot edit a hidden layer. Please make the layer visible first.",
            )
            return

        if not self.rotation_input.text():
            return

        try:
            angle = float(self.rotation_input.text())
            item = self.image_view.moving_items[self.selected_moving_index]
            transform = item.transform()
            center = item.boundingRect().center()

            t = QTransform()
            t.translate(center.x(), center.y())
            t.rotate(angle)
            t.translate(-center.x(), -center.y())

            item.setTransform(transform * t)
            self.update_offset_label()
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid rotation angle."
            )

    def apply_scale(self):
        """Apply scale to selected moving image."""
        if self.selected_moving_index < 0 or self.selected_moving_index >= len(self.moving_images):
            return

        if not self.visibility_checkboxes[self.selected_moving_index].isChecked():
            QMessageBox.warning(
                self,
                "Layer Hidden",
                "Cannot edit a hidden layer. Please make the layer visible first.",
            )
            return

        if not self.scale_input.text():
            return

        try:
            scale = float(self.scale_input.text())
            item = self.image_view.moving_items[self.selected_moving_index]
            transform = item.transform()
            center = item.boundingRect().center()

            t = QTransform()
            t.translate(center.x(), center.y())
            t.scale(scale, scale)
            t.translate(-center.x(), -center.y())

            item.setTransform(transform * t)
            self.update_offset_label()
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid scale factor."
            )

    def apply_flip_horizontal(self):
        """Apply horizontal flip to selected moving image."""
        if self.selected_moving_index < 0 or self.selected_moving_index >= len(self.moving_images):
            return

        if not self.visibility_checkboxes[self.selected_moving_index].isChecked():
            QMessageBox.warning(
                self,
                "Layer Hidden",
                "Cannot edit a hidden layer. Please make the layer visible first.",
            )
            return

        item = self.image_view.moving_items[self.selected_moving_index]
        transform = item.transform()
        center = item.boundingRect().center()

        t = QTransform()
        t.translate(center.x(), center.y())
        t.scale(-1, 1)
        t.translate(-center.x(), -center.y())

        item.setTransform(transform * t)
        self.update_offset_label()

    def apply_flip_vertical(self):
        """Apply vertical flip to selected moving image."""
        if self.selected_moving_index < 0 or self.selected_moving_index >= len(self.moving_images):
            return

        if not self.visibility_checkboxes[self.selected_moving_index].isChecked():
            QMessageBox.warning(
                self,
                "Layer Hidden",
                "Cannot edit a hidden layer. Please make the layer visible first.",
            )
            return

        item = self.image_view.moving_items[self.selected_moving_index]
        transform = item.transform()
        center = item.boundingRect().center()

        t = QTransform()
        t.translate(center.x(), center.y())
        t.scale(1, -1)
        t.translate(-center.x(), -center.y())

        item.setTransform(transform * t)
        self.update_offset_label()

    def move_aligned_image(self, dx, dy):
        """Move the selected moving image by dx, dy pixels."""
        if self.selected_moving_index < 0 or self.selected_moving_index >= len(self.moving_images):
            return

        item = self.image_view.moving_items[self.selected_moving_index]
        transform = item.transform()
        transform.translate(dx, dy)
        item.setTransform(transform)
        self.update_offset_label()

    def update_offset_label(self):
        """Update the label showing the current transformation matrix of selected moving image."""
        if self.selected_moving_index < 0 or self.selected_moving_index >= len(self.image_view.moving_items):
            self.offset_label.setText("No layer selected")
            return

        item = self.image_view.moving_items[self.selected_moving_index]
        transform_matrix = item.transform()
        transform_text = readable_matrix_string(transform_to_matrix(transform_matrix))
        self.offset_label.setText(f"Layer {self.selected_moving_index + 1}: {transform_text}")

    def reset_transformations(self):
        """Reset all transformations on all moving images."""
        for item in self.image_view.moving_items:
            item.resetTransform()
        self.image_view.reset_zoom()
        self.update_offset_label()

    def accept_alignment(self):
        """Accept the alignment and emit transformed images if editing was enabled."""
        self.result_accepted = True

        if self.can_edit and self.can_emit:
            # Apply transformations to original images and emit them
            transformed_images = []
            transformation_matrices_list = []

            h, w = self.target_image.shape[:2]

            for i, item in enumerate(self.image_view.moving_items):
                final_transformation = item.transform()
                transf_matrix = transform_to_matrix(final_transformation)
                transformation_matrices_list.append(transf_matrix)

                final_image = cv2.warpAffine(
                    self.original_moving_images[i], transf_matrix, (w, h)
                )
                transformed_images.append(final_image)

            self.transformation_matrices.emit(transformation_matrices_list)
            self.moving_images_changed.emit(transformed_images)

        self.accept()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard events for arrow key movement when editing."""
        if not self.can_edit:
            super().keyPressEvent(event)
            return

        # Prevent arrow keys from being processed if a text input has focus
        if self.focusWidget() in [
            self.dx_input,
            self.dy_input,
            self.rotation_input,
            self.scale_input,
        ]:
            super().keyPressEvent(event)
            return

        # Check if selected layer is visible
        if self.selected_moving_index >= 0 and not self.visibility_checkboxes[
            self.selected_moving_index
        ].isChecked():
            super().keyPressEvent(event)
            return

        key_map = {
            Qt.Key.Key_Left: (-self.move_step, 0),
            Qt.Key.Key_Right: (self.move_step, 0),
            Qt.Key.Key_Up: (0, -self.move_step),
            Qt.Key.Key_Down: (0, self.move_step),
        }

        if event.key() in key_map:
            self.move_aligned_image(*key_map[event.key()])
        else:
            super().keyPressEvent(event)

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


def readable_matrix_string(matrix: np.ndarray) -> str:
    """Convert a 2x3 transformation matrix to a human-readable string."""
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