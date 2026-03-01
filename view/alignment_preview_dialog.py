import math
from typing import Optional

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

        # For shift+drag to move selected layer
        self._is_dragging_layer = False
        self._drag_start_pos = None
        self._parent_dialog = None

        self.min_zoom = 0.1
        self.max_zoom = 150.0
        self.current_zoom = 1.0

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
        self.current_zoom = 1.0

    def get_scene(self):
        s = self.scene()
        assert s is not None
        return s

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming with limits."""
        if event is None:
            return
        angle = event.angleDelta().y()
        if angle > 0:
            zoom_factor = 1.15
        else:
            zoom_factor = 1 / 1.15

        # Calculate new zoom level
        new_zoom = self.current_zoom * zoom_factor

        # Enforce zoom limits
        if new_zoom < self.min_zoom:
            # Already at min zoom, don't zoom out further
            return
        elif new_zoom > self.max_zoom:
            # Already at max zoom, don't zoom in further
            return

        # Apply zoom
        self.scale(zoom_factor, zoom_factor)
        self.current_zoom = new_zoom

    def mousePressEvent(self, event):
        """Handle mouse press events for shift+drag layer movement."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.ShiftModifier
        ):
            if self._parent_dialog and self._parent_dialog.can_edit:
                if self._parent_dialog._get_editable_indices():
                    self._is_dragging_layer = True
                    self._drag_start_pos = self.mapToScene(event.pos())
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for shift+drag layer movement."""
        if self._is_dragging_layer and self._drag_start_pos is not None:
            current_pos = self.mapToScene(event.pos())
            delta = current_pos - self._drag_start_pos

            if self._parent_dialog:
                for selected_index in self._parent_dialog._get_editable_indices():
                    if not (0 <= selected_index < len(self.moving_items)):
                        continue
                    item = self.moving_items[selected_index]
                    transform = item.transform()

                    current_dx = transform.dx()
                    current_dy = transform.dy()

                    new_transform = QTransform(
                        transform.m11(),
                        transform.m12(),
                        transform.m21(),
                        transform.m22(),
                        current_dx + delta.x(),
                        current_dy + delta.y(),
                    )
                    item.setTransform(new_transform)
                self._parent_dialog.update_offset_label()

            self._drag_start_pos = current_pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events for shift+drag layer movement."""
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging_layer:
            self._is_dragging_layer = False
            self._drag_start_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AlignmentPreviewDialog(QDialog):
    moving_images_changed = pyqtSignal(list)
    transformation_matrices = pyqtSignal(list)

    def __init__(
        self,
        target_image: np.ndarray,
        moving_images: list[np.ndarray],
        can_edit: bool = False,
        can_emit: bool = False,
        initial_preview_size: Optional[int] = None,
        initial_checked_indices: Optional[list[int]] = None,
    ):
        super().__init__(None)

        self.full_target_image = target_image.copy()
        self.full_moving_images = [img.copy() for img in moving_images]
        self.max_preview_size = self._compute_max_preview_size()
        self.preview_size = self._clamp_preview_size(initial_preview_size)
        self.target_image = self._crop_to_preview_size(self.full_target_image)
        self.moving_images = [
            self._crop_to_preview_size(img) for img in self.full_moving_images
        ]
        self.original_moving_images = [img.copy() for img in self.moving_images]
        self.can_edit = can_edit
        self.can_emit = can_emit
        self.initial_checked_indices = (
            set(initial_checked_indices) if initial_checked_indices is not None else None
        )
        self.adjust_contrast = True
        self.result_accepted = False
        self.move_step = 1

        self._setup_ui()
        self.create_direct_overlay()
        self.image_view.mouseDoubleClickEvent = self.reset_zoom

    def _image_hw(self, image: np.ndarray) -> tuple[int, int]:
        if image.ndim < 2:
            return (1, 1)
        return (int(image.shape[-2]), int(image.shape[-1]))

    def _compute_max_preview_size(self) -> int:
        sizes = []
        target_h, target_w = self._image_hw(self.full_target_image)
        sizes.extend([target_h, target_w])
        for img in self.full_moving_images:
            h, w = self._image_hw(img)
            sizes.extend([h, w])
        if not sizes:
            return 1
        return max(1, min(sizes))

    def _clamp_preview_size(self, preview_size: Optional[int]) -> int:
        if preview_size is None:
            return self.max_preview_size
        return max(1, min(int(preview_size), self.max_preview_size))

    def _crop_to_preview_size(self, image: np.ndarray) -> np.ndarray:
        if image.ndim < 2:
            return image.copy()
        return image[..., : self.preview_size, : self.preview_size].copy()

    def _apply_preview_size(self, preview_size: int):
        self.preview_size = self._clamp_preview_size(preview_size)
        self.target_image = self._crop_to_preview_size(self.full_target_image)
        self.moving_images = [
            self._crop_to_preview_size(img) for img in self.full_moving_images
        ]
        self.original_moving_images = [img.copy() for img in self.moving_images]

    def _build_preview_label_text(self) -> str:
        instruction_text = (
            "Arrow keys/Inputs: move checked layers, Shift+Drag: move checked layers, Mouse wheel: zoom, Drag: pan, Double-click: reset view"
            if self.can_edit
            else "Mouse wheel: zoom, Drag: pan, Double-click: reset view"
        )
        if not self.can_edit:
            return f"Red = Target, Other colors = Aligned | {instruction_text}"
        return (
            f"Red = Target, Other colors = Aligned | {instruction_text}\n"
            f"Hint: Align the top-left region first. Preview starts at {self.preview_size}x{self.preview_size} and can be adjusted up to {self.max_preview_size}."
        )

    def _setup_ui(self):
        self.setWindowTitle("Alignment Preview")
        self.resize(1000, 800)
        main_layout = QVBoxLayout(self)

        self.enhance_contrast_checkbox = QCheckBox("Enhance Contrast")
        self.enhance_contrast_checkbox.setChecked(self.adjust_contrast)
        self.enhance_contrast_checkbox.stateChanged.connect(
            self._on_contrast_checkbox_changed
        )

        self.preview_label = QLabel(self._build_preview_label_text())
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.offset_label = QLabel()
        self.offset_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offset_label.setVisible(self.can_edit)

        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)
        self.image_view._parent_dialog = self

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

        for i in range(len(self.moving_images)):
            layer_layout = QHBoxLayout()

            checkbox = QCheckBox(f"Moving Image {i + 1}")
            if self.initial_checked_indices is None:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(i in self.initial_checked_indices)
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
        for i, item in enumerate(self.image_view.moving_items):
            item.setVisible(self.visibility_checkboxes[i].isChecked())

    def _get_editable_indices(self) -> list[int]:
        return [
            i for i, checkbox in enumerate(self.visibility_checkboxes) if checkbox.isChecked()
        ]

    def _has_identity_transform(self, transform: QTransform) -> bool:
        matrix = transform_to_matrix(transform)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return bool(np.allclose(matrix, identity, atol=1e-6))

    def _on_visibility_changed(self, index: int, state):
        """Called when a moving layer visibility checkbox is toggled."""
        visible = state == Qt.CheckState.Checked.value
        self.image_view.toggle_moving_item_visibility(index, visible)
        self.update_offset_label()

    def _setup_editable_controls(self):
        """Create UI controls for when manual editing is enabled."""
        preview_group = QGroupBox("Preview (Top-Left Crop)")
        preview_layout = QHBoxLayout()
        self.preview_size_input = QLineEdit(str(self.preview_size))
        self.preview_size_input.setValidator(QIntValidator(1, self.max_preview_size))
        self.preview_size_input.setFixedWidth(80)
        self.apply_preview_size_button = QPushButton("Apply")
        self.apply_preview_size_button.clicked.connect(self.apply_preview_size)
        preview_layout.addWidget(QLabel("Size:"))
        preview_layout.addWidget(self.preview_size_input)
        preview_layout.addWidget(QLabel(f"(max {self.max_preview_size})"))
        preview_layout.addWidget(self.apply_preview_size_button)
        preview_group.setLayout(preview_layout)

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

        self.control_layout.addWidget(preview_group)
        self.control_layout.addWidget(trans_group)
        self.control_layout.addWidget(rot_group)
        self.control_layout.addWidget(scale_group)
        self.control_layout.addWidget(flip_group)
        self.control_layout.addStretch()
        self.control_layout.addWidget(self.reset_button)
        self._setup_confirm_cancel_buttons()

    def apply_preview_size(self):
        if not self.preview_size_input.text():
            return

        try:
            new_size = int(self.preview_size_input.text())
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid integer preview size."
            )
            return

        clamped_size = self._clamp_preview_size(new_size)
        if clamped_size != new_size:
            self.preview_size_input.setText(str(clamped_size))
            QMessageBox.information(
                self,
                "Preview Size Adjusted",
                f"Preview size was adjusted to {clamped_size} (max {self.max_preview_size}).",
            )
        if clamped_size == self.preview_size:
            return

        saved_transforms = [item.transform() for item in self.image_view.moving_items]
        saved_visibility = [checkbox.isChecked() for checkbox in self.visibility_checkboxes]
        self._apply_preview_size(clamped_size)
        self.create_direct_overlay()

        for i, item in enumerate(self.image_view.moving_items):
            if i < len(saved_transforms):
                item.setTransform(saved_transforms[i])
            if i < len(saved_visibility):
                item.setVisible(saved_visibility[i])

        self.preview_size_input.setText(str(clamped_size))
        self.preview_label.setText(self._build_preview_label_text())
        self.update_offset_label()

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
        """Applies translation based on the dx/dy input fields to checked moving images."""
        editable_indices = self._get_editable_indices()
        if not editable_indices:
            QMessageBox.warning(
                self, "No Layer Selected", "Check at least one moving layer to edit."
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

        self.move_aligned_images(dx, dy)

    def apply_rotation(self):
        """Apply rotation to checked moving images."""
        editable_indices = self._get_editable_indices()
        if not editable_indices:
            QMessageBox.warning(
                self, "No Layer Selected", "Check at least one moving layer to edit."
            )
            return

        if not self.rotation_input.text():
            return

        try:
            angle = float(self.rotation_input.text())
            for idx in editable_indices:
                item = self.image_view.moving_items[idx]
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
        """Apply scale to checked moving images."""
        editable_indices = self._get_editable_indices()
        if not editable_indices:
            QMessageBox.warning(
                self, "No Layer Selected", "Check at least one moving layer to edit."
            )
            return

        if not self.scale_input.text():
            return

        try:
            scale = float(self.scale_input.text())
            for idx in editable_indices:
                item = self.image_view.moving_items[idx]
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
        """Apply horizontal flip to checked moving images."""
        editable_indices = self._get_editable_indices()
        if not editable_indices:
            QMessageBox.warning(
                self, "No Layer Selected", "Check at least one moving layer to edit."
            )
            return

        for idx in editable_indices:
            item = self.image_view.moving_items[idx]
            transform = item.transform()
            center = item.boundingRect().center()

            t = QTransform()
            t.translate(center.x(), center.y())
            t.scale(-1, 1)
            t.translate(-center.x(), -center.y())

            item.setTransform(transform * t)
        self.update_offset_label()

    def apply_flip_vertical(self):
        """Apply vertical flip to checked moving images."""
        editable_indices = self._get_editable_indices()
        if not editable_indices:
            QMessageBox.warning(
                self, "No Layer Selected", "Check at least one moving layer to edit."
            )
            return

        for idx in editable_indices:
            item = self.image_view.moving_items[idx]
            transform = item.transform()
            center = item.boundingRect().center()

            t = QTransform()
            t.translate(center.x(), center.y())
            t.scale(1, -1)
            t.translate(-center.x(), -center.y())

            item.setTransform(transform * t)
        self.update_offset_label()

    def move_aligned_images(self, dx, dy):
        """Move checked moving images by dx, dy pixels in screen coordinates."""
        editable_indices = self._get_editable_indices()
        if not editable_indices:
            return

        for idx in editable_indices:
            item = self.image_view.moving_items[idx]
            transform = item.transform()
            current_dx = transform.dx()
            current_dy = transform.dy()
            new_transform = QTransform(
                transform.m11(),
                transform.m12(),
                transform.m21(),
                transform.m22(),
                current_dx + dx,
                current_dy + dy,
            )
            item.setTransform(new_transform)
        self.update_offset_label()

    def update_offset_label(self):
        """Update the label showing the current transformation matrix of checked moving images."""
        editable_indices = self._get_editable_indices()
        if not editable_indices:
            self.offset_label.setText("No checked layer")
            return

        label_lines = []
        for idx in editable_indices:
            if idx >= len(self.image_view.moving_items):
                continue
            item = self.image_view.moving_items[idx]
            transform_matrix = item.transform()
            transform_text = readable_matrix_string(
                transform_to_matrix(transform_matrix)
            )
            label_lines.append(f"Layer {idx + 1}: {transform_text}")
        if not label_lines:
            self.offset_label.setText("No checked layer")
            return
        self.offset_label.setText(" | ".join(label_lines))

    def reset_transformations(self):
        """Reset all transformations on all moving images."""
        for item in self.image_view.moving_items:
            item.resetTransform()
        self.image_view.reset_zoom()
        self.update_offset_label()

    def accept_alignment(self):
        """Accept the alignment and emit transformed images if editing was enabled."""
        if self.can_edit and self.can_emit:
            unchecked_with_changes = []
            for i, item in enumerate(self.image_view.moving_items):
                if self.visibility_checkboxes[i].isChecked():
                    continue
                if not self._has_identity_transform(item.transform()):
                    unchecked_with_changes.append(i + 1)

            if unchecked_with_changes:
                layer_text = ", ".join(str(idx) for idx in unchecked_with_changes)
                reply = QMessageBox.question(
                    self,
                    "Unchecked Layers Have Edits",
                    (
                        f"Layer(s) {layer_text} are unchecked and have unapplied edits. "
                        "Their transforms will not be saved. Continue?"
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            # Apply transformations to original images and emit them
            transformed_images = []
            transformation_matrices_list = []

            h, w = self.target_image.shape[:2]

            for i, item in enumerate(self.image_view.moving_items):
                if not self.visibility_checkboxes[i].isChecked():
                    transformation_matrices_list.append(None)
                    transformed_images.append(self.original_moving_images[i].copy())
                    continue
                final_transformation = item.transform()
                transf_matrix = transform_to_matrix(final_transformation)
                transformation_matrices_list.append(transf_matrix)

                final_image = cv2.warpAffine(
                    self.original_moving_images[i], transf_matrix, (w, h)
                )
                transformed_images.append(final_image)

            self.transformation_matrices.emit(transformation_matrices_list)
            self.moving_images_changed.emit(transformed_images)

        self.result_accepted = True
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

        if not self._get_editable_indices():
            super().keyPressEvent(event)
            return

        key_map = {
            Qt.Key.Key_Left: (-self.move_step, 0),
            Qt.Key.Key_Right: (self.move_step, 0),
            Qt.Key.Key_Up: (0, -self.move_step),
            Qt.Key.Key_Down: (0, self.move_step),
        }

        if event.key() in key_map:
            self.move_aligned_images(*key_map[event.key()])
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
