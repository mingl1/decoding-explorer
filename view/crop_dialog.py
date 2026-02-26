import numpy as np
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QIntValidator,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
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

from utils import to_uint8


class ResizeHandle(QGraphicsEllipseItem):
    """Draggable resize handle for crop rectangle."""

    def __init__(self, handle_type: str, parent=None):
        """
        Initialize resize handle.

        Args:
            handle_type: One of 'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w' for position
            parent: Parent graphics item
        """
        self.base_size = 12  # Base size in screen pixels
        size = self.base_size
        super().__init__(-size/2, -size/2, size, size, parent)

        self.handle_type = handle_type
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)  # Keep constant screen size
        self.setAcceptHoverEvents(True)
        self.setCursor(self._get_cursor())

        # Visual style
        self.setBrush(QBrush(QColor(255, 255, 0, 200)))  # Yellow
        self.setPen(QPen(QColor(0, 0, 0), 2))
        self.setZValue(12)  # Above crop rect

    def _get_cursor(self):
        """Get appropriate cursor for handle position."""
        cursors = {
            'nw': Qt.CursorShape.SizeFDiagCursor,
            'n': Qt.CursorShape.SizeVerCursor,
            'ne': Qt.CursorShape.SizeBDiagCursor,
            'e': Qt.CursorShape.SizeHorCursor,
            'se': Qt.CursorShape.SizeFDiagCursor,
            's': Qt.CursorShape.SizeVerCursor,
            'sw': Qt.CursorShape.SizeBDiagCursor,
            'w': Qt.CursorShape.SizeHorCursor,
        }
        return cursors.get(self.handle_type, Qt.CursorShape.ArrowCursor)


class InteractiveCropRect(QGraphicsRectItem):
    """Interactive crop rectangle with draggable resize handles."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.DashLine))
        self.setZValue(10)

        # Create 8 resize handles
        self.handles = {}
        handle_positions = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']
        for pos in handle_positions:
            handle = ResizeHandle(pos, self)
            self.handles[pos] = handle

        self.dragging_handle = None
        self.drag_start_pos = None
        self.drag_start_rect = None
        self.min_size = 10  # Minimum crop size in pixels

    def setRect(self, *args):
        """Override setRect to update handle positions."""
        super().setRect(*args)
        self._update_handle_positions()

    def _update_handle_positions(self):
        """Position handles at rectangle corners and edges."""
        rect = self.rect()
        x1, y1 = rect.left(), rect.top()
        x2, y2 = rect.right(), rect.bottom()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        positions = {
            'nw': (x1, y1),
            'n': (cx, y1),
            'ne': (x2, y1),
            'e': (x2, cy),
            'se': (x2, y2),
            's': (cx, y2),
            'sw': (x1, y2),
            'w': (x1, cy),
        }

        for pos, (x, y) in positions.items():
            self.handles[pos].setPos(x, y)

    def get_handle_at_pos(self, scene_pos):
        """Check if a handle is at the given position."""
        for handle in self.handles.values():
            # Get handle's bounding rect in scene coordinates
            handle_scene_rect = handle.sceneBoundingRect()
            if handle_scene_rect.contains(scene_pos):
                return handle
        return None


class ZoomableImageView(QGraphicsView):
    """Zoomable image view for crop dialog."""

    mouse_moved = pyqtSignal(int, int)  # x, y in scene coordinates
    crop_changed = pyqtSignal(float, float, float, float)  # x1, y1, x2, y2 in scene coordinates

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.image_items: list[QGraphicsPixmapItem] = []
        self.setScene(self._scene)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Enable mouse tracking
        self.setMouseTracking(True)
        viewport = self.viewport()
        if viewport:
            viewport.setMouseTracking(True)

        # Zoom limits
        self.min_zoom = 0.1  # 10% of original size
        self.max_zoom = 150.0  # 10x original size
        self.current_zoom = 1.0

        # Handle dragging state
        self.dragging_handle = None
        self.drag_start_pos = None
        self.drag_start_rect = None
        self.crop_rect_item: InteractiveCropRect | None = None

    def set_images(self, pixmaps: list[QPixmap], opacities: list[float] | None = None):
        """Set images to display."""
        # Clear existing items
        for item in self.image_items:
            self._scene.removeItem(item)
        self.image_items.clear()

        if opacities is None:
            opacities = [1.0] * len(pixmaps)

        for pixmap, opacity in zip(pixmaps, opacities):
            item = QGraphicsPixmapItem(pixmap)
            item.setOpacity(opacity)
            self.image_items.append(item)
            self._scene.addItem(item)

        QTimer.singleShot(0, self.reset_zoom)

    def toggle_layer_visibility(self, index: int, visible: bool):
        """Toggle visibility of a specific layer."""
        if 0 <= index < len(self.image_items):
            self.image_items[index].setVisible(visible)

    def reset_zoom(self):
        """Reset zoom to fit all items."""
        if not self.scene():
            return
        self.get_scene().setSceneRect(self.get_scene().itemsBoundingRect())
        if self.image_items:
            self.fitInView(self.image_items[0], Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(self.image_items[0])

        # Reset zoom tracking
        self.current_zoom = 1.0

    def get_scene(self):
        """Get the scene."""
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
        """Handle mouse press for handle dragging."""
        if event is None:
            return

        if event.button() == Qt.MouseButton.LeftButton and self.crop_rect_item:
            # Check if we clicked on a handle by looking at items at the click position
            items = self.items(event.pos())
            for item in items:
                if isinstance(item, ResizeHandle):
                    # Start dragging this handle
                    self.dragging_handle = item
                    self.drag_start_pos = self.mapToScene(event.pos())
                    self.drag_start_rect = QRectF(self.crop_rect_item.rect())
                    self.setDragMode(QGraphicsView.DragMode.NoDrag)
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Track mouse movement and handle dragging."""
        if event is None:
            return

        pos = event.position()
        scene_pos = self.mapToScene(int(pos.x()), int(pos.y()))

        # Handle dragging
        if self.dragging_handle and self.drag_start_pos and self.drag_start_rect:
            self._resize_crop_rect(scene_pos)
            event.accept()
        else:
            # Emit scene coordinates
            self.mouse_moved.emit(int(scene_pos.x()), int(scene_pos.y()))

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop dragging."""
        if event is None:
            return

        if event.button() == Qt.MouseButton.LeftButton and self.dragging_handle:
            # End dragging
            self.dragging_handle = None
            self.drag_start_pos = None
            self.drag_start_rect = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

            # Emit final crop bounds
            if self.crop_rect_item:
                rect = self.crop_rect_item.rect()
                self.crop_changed.emit(rect.left(), rect.top(), rect.right(), rect.bottom())

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _resize_crop_rect(self, scene_pos):
        """Resize crop rectangle based on handle drag."""
        if not self.dragging_handle or not self.drag_start_pos or not self.drag_start_rect or not self.crop_rect_item:
            return

        delta_x = scene_pos.x() - self.drag_start_pos.x()
        delta_y = scene_pos.y() - self.drag_start_pos.y()

        handle_type = self.dragging_handle.handle_type
        rect = QRectF(self.drag_start_rect)

        # Update rectangle based on which handle is being dragged
        if 'n' in handle_type:  # North handles
            new_top = rect.top() + delta_y
            if rect.bottom() - new_top >= self.crop_rect_item.min_size:
                rect.setTop(new_top)

        if 's' in handle_type:  # South handles
            new_bottom = rect.bottom() + delta_y
            if new_bottom - rect.top() >= self.crop_rect_item.min_size:
                rect.setBottom(new_bottom)

        if 'w' in handle_type:  # West handles
            new_left = rect.left() + delta_x
            if rect.right() - new_left >= self.crop_rect_item.min_size:
                rect.setLeft(new_left)

        if 'e' in handle_type:  # East handles
            new_right = rect.right() + delta_x
            if new_right - rect.left() >= self.crop_rect_item.min_size:
                rect.setRight(new_right)

        # Update the crop rectangle
        self.crop_rect_item.setRect(rect)

        # Emit updated bounds for live preview in text fields
        self.crop_changed.emit(rect.left(), rect.top(), rect.right(), rect.bottom())


class CropDialog(QDialog):
    """Interactive crop dialog with coordinate input and mouse tracking."""

    crop_confirmed = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2

    def __init__(self, images: list[np.ndarray], parent=None):
        """
        Initialize crop dialog.

        Args:
            images: List of grayscale images. First is reference/target, rest are moving.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.original_images = images
        self.crop_rect: InteractiveCropRect | None = None
        self.highlight_rect: QGraphicsRectItem | None = None
        self.updating_from_drag = False  # Flag to prevent circular updates

        # Use original images directly (no downsampling)
        self.preview_images = images

        self._setup_ui()
        self._create_overlay()

        # Set initial crop to full image
        h, w = self.preview_images[0].shape
        self.start_x_input.setText("0")
        self.start_y_input.setText("0")
        self.end_x_input.setText(str(w))
        self.end_y_input.setText(str(h))
        self._update_crop_preview()

    def _setup_ui(self):
        """Create UI layout."""
        self.setWindowTitle("Crop Image")
        self.resize(1000, 800)
        main_layout = QVBoxLayout(self)

        # Instructions
        instruction_text = "Mouse wheel: zoom, Drag: pan, Double-click: reset view, Drag handles: resize crop"
        self.instruction_label = QLabel(instruction_text)
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Coordinate label
        self.coord_label = QLabel("Current Position: X: 0, Y: 0")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Image view
        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)
        self.image_view.mouse_moved.connect(self._on_mouse_move)
        self.image_view.crop_changed.connect(self._on_crop_changed_by_drag)
        self.image_view.mouseDoubleClickEvent = self.reset_zoom

        # Crop controls
        crop_group = QGroupBox("Crop Bounds (Full Resolution Pixels)")
        crop_layout = QFormLayout()

        validator = QIntValidator(0, 999999)

        self.start_x_input = QLineEdit()
        self.start_x_input.setValidator(validator)
        self.start_x_input.textChanged.connect(self._update_crop_preview)

        self.start_y_input = QLineEdit()
        self.start_y_input.setValidator(validator)
        self.start_y_input.textChanged.connect(self._update_crop_preview)

        self.end_x_input = QLineEdit()
        self.end_x_input.setValidator(validator)
        self.end_x_input.textChanged.connect(self._update_crop_preview)

        self.end_y_input = QLineEdit()
        self.end_y_input.setValidator(validator)
        self.end_y_input.textChanged.connect(self._update_crop_preview)

        crop_layout.addRow("Start X (left):", self.start_x_input)
        crop_layout.addRow("Start Y (top):", self.start_y_input)
        crop_layout.addRow("End X (right):", self.end_x_input)
        crop_layout.addRow("End Y (bottom):", self.end_y_input)

        self.size_label = QLabel("Crop Size: 0 x 0 pixels")
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        crop_layout.addRow(self.size_label)

        crop_group.setLayout(crop_layout)

        # Layer visibility controls (if multiple images)
        if len(self.preview_images) > 1:
            self.visibility_groupbox = QGroupBox("Layers to be cropped")
            visibility_layout = QVBoxLayout()
            self.visibility_checkboxes = []

            colors = ["red (reference)", "green", "blue", "cyan", "magenta", "yellow"]
            for i in range(len(self.preview_images)):
                color_label = colors[i % len(colors)]
                if i == 0:
                    checkbox = QCheckBox(f"Reference ({color_label})")
                else:
                    checkbox = QCheckBox(f"Moving Image {i} ({color_label})")
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(
                    lambda state, index=i: self._on_visibility_changed(index, state)
                )
                self.visibility_checkboxes.append(checkbox)
                visibility_layout.addWidget(checkbox)

            self.visibility_groupbox.setLayout(visibility_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm Crop")
        self.confirm_button.clicked.connect(self._validate_and_confirm)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()

        # Add to main layout
        main_layout.addWidget(self.instruction_label)
        main_layout.addWidget(self.coord_label)
        main_layout.addWidget(self.image_view)
        main_layout.addWidget(crop_group)
        if len(self.preview_images) > 1:
            main_layout.addWidget(self.visibility_groupbox)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _create_overlay(self):
        """Create image overlay with colorization."""
        colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]
        pixmaps = []
        opacities = []

        for i, img in enumerate(self.preview_images):
            color = colors[i % len(colors)]
            pixmap = colorize_grayscale(to_uint8(img), color)
            pixmaps.append(pixmap)

            # Reference is opaque, others are semi-transparent
            if i == 0:
                opacities.append(1.0)
            else:
                opacities.append(0.5)

        self.image_view.set_images(pixmaps, opacities)

        # Add interactive crop rectangle overlay
        self.crop_rect = InteractiveCropRect()
        self.image_view.get_scene().addItem(self.crop_rect)
        self.image_view.crop_rect_item = self.crop_rect  # Store reference in view

        # Add pixel highlight rectangle
        self.highlight_rect = QGraphicsRectItem()
        highlight_pen = QPen(Qt.GlobalColor.yellow, 1, Qt.PenStyle.SolidLine)
        self.highlight_rect.setPen(highlight_pen)
        self.highlight_rect.setZValue(11)  # Above crop rect
        self.image_view.get_scene().addItem(self.highlight_rect)

    def _on_mouse_move(self, scene_x: int, scene_y: int):
        """Update coordinate label and pixel highlight on mouse move."""
        # Update coordinate label (scene coordinates are already at full resolution)
        self.coord_label.setText(f"Current Position: X: {scene_x}, Y: {scene_y}")

        # Update pixel highlight rectangle (small square at cursor)
        if self.highlight_rect:
            highlight_size = 4
            self.highlight_rect.setRect(
                scene_x - highlight_size / 2,
                scene_y - highlight_size / 2,
                highlight_size,
                highlight_size,
            )

    def _on_visibility_changed(self, index: int, state):
        """Toggle layer visibility."""
        visible = state == Qt.CheckState.Checked.value
        self.image_view.toggle_layer_visibility(index, visible)

    def _on_crop_changed_by_drag(self, x1: float, y1: float, x2: float, y2: float):
        """Update text fields when crop bounds change via handle dragging."""
        # Prevent circular updates
        self.updating_from_drag = True

        # Update text fields (coordinates are already at full resolution)
        self.start_x_input.setText(str(int(x1)))
        self.start_y_input.setText(str(int(y1)))
        self.end_x_input.setText(str(int(x2)))
        self.end_y_input.setText(str(int(y2)))

        # Update size label
        width = int(x2 - x1)
        height = int(y2 - y1)
        self.size_label.setText(f"Crop Size: {width} x {height} pixels")

        self.updating_from_drag = False

    def _update_crop_preview(self):
        """Update crop rectangle overlay based on input fields."""
        # Skip if we're updating from a drag operation to prevent circular updates
        if self.updating_from_drag:
            return

        try:
            x1 = int(self.start_x_input.text()) if self.start_x_input.text() else 0
            y1 = int(self.start_y_input.text()) if self.start_y_input.text() else 0
            x2 = int(self.end_x_input.text()) if self.end_x_input.text() else 0
            y2 = int(self.end_y_input.text()) if self.end_y_input.text() else 0

            # Update crop rectangle (no scaling needed - coordinates are at full resolution)
            width = x2 - x1
            height = y2 - y1

            if width > 0 and height > 0:
                if self.crop_rect:
                    self.crop_rect.setRect(x1, y1, width, height)

                # Update size label
                self.size_label.setText(f"Crop Size: {width} x {height} pixels")
            else:
                self.size_label.setText("Crop Size: Invalid")

        except (ValueError, ZeroDivisionError):
            pass  # Invalid input, don't update

    def _validate_and_confirm(self):
        """Validate crop bounds and emit signal."""
        try:
            x1 = int(self.start_x_input.text())
            y1 = int(self.start_y_input.text())
            x2 = int(self.end_x_input.text())
            y2 = int(self.end_y_input.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid integer values for all crop bounds.",
            )
            return

        # Validate bounds
        h, w = self.original_images[0].shape

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            QMessageBox.warning(
                self,
                "Invalid Bounds",
                f"Crop bounds must be within image dimensions (0-{w}, 0-{h}).",
            )
            return

        if x2 <= x1 or y2 <= y1:
            QMessageBox.warning(
                self,
                "Invalid Bounds",
                "End coordinates must be greater than start coordinates.",
            )
            return

        # Emit signal with full resolution coordinates
        self.crop_confirmed.emit(x1, y1, x2, y2)
        self.accept()

    def reset_zoom(self, event=None):
        """Reset zoom to fit view."""
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
