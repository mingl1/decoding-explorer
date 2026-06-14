from enum import Enum
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal, QPointF, QSize, QObject
from PyQt6.QtGui import (
    QColor,
    QIntValidator,
    QPen,
    QBrush,
    QPainter,
    QPolygonF,
    QPainterPath,
    QKeyEvent,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QGraphicsPolygonItem,
    QStyleOptionGraphicsItem,
    QListWidget,
    QListWidgetItem,
    QWidget,
    QButtonGroup,
)

from utils import adjust_contrast, to_uint8


def point_in_polygon(x, y, polygon):
    num = len(polygon)
    j = num - 1
    c = False
    for i in range(num):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            c = not c
        j = i
    return c


class DrawMode(Enum):
    SELECT = "select"
    RECT = "rect"
    CIRCLE = "circle"
    POLY = "poly"


class CropRectROI(pg.RectROI):
    def __init__(self, pos, size, pen=None):
        if pen is None:
            pen = pg.mkPen(QColor(255, 255, 0), width=2, style=Qt.PenStyle.DashLine)
        super().__init__(pos, size, pen=pen, rotatable=False, removable=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addScaleHandle([0.5, 1], [0.5, 0])
        self.addScaleHandle([1, 0.5], [0, 0.5])


class CropCircleROI(pg.EllipseROI):
    def __init__(self, pos, size, pen=None):
        if pen is None:
            pen = pg.mkPen(QColor(0, 255, 255), width=2, style=Qt.PenStyle.DashLine)
        super().__init__(pos, size, pen=pen, rotatable=False, aspectLocked=True, removable=True)
        self.addScaleHandle([1, 1], [0.5, 0.5])


class SignalHelper(QObject):
    sigRegionChangeFinished = pyqtSignal(object)


class CropPolyROI(QGraphicsPolygonItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sig_helper = SignalHelper()
        self.sigRegionChangeFinished = self.sig_helper.sigRegionChangeFinished
        self.points = []
        col = QColor(255, 0, 255)
        self.line_color = col
        self.fill_color = QColor(col.red(), col.green(), col.blue(), 50)
        self.completed = False
        self.temp_point = None
        self.point_size = 6
        self.setPen(QPen(self.line_color, 2, Qt.PenStyle.SolidLine))
        self.setBrush(QBrush(self.fill_color))
        self.setZValue(12)

    def add_point(self, pt):
        self.points.append(pt)
        self.update_polygon()

    def update_polygon(self):
        polygon = QPolygonF(self.points)
        self.setPolygon(polygon)

    def set_temp_point(self, pt):
        self.prepareGeometryChange()
        self.temp_point = pt
        self.update()

    def complete(self):
        if len(self.points) >= 3:
            self.completed = True
            self.temp_point = None
            self.update_polygon()
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
            return True
        return False

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.sigRegionChangeFinished.emit(self)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        lod = QStyleOptionGraphicsItem.levelOfDetailFromTransform(painter.worldTransform())
        r = (self.point_size / 2) / lod
        if len(self.points) > 1:
            pen = QPen(self.line_color, 2 / lod)
            painter.setPen(pen)
            path = QPainterPath()
            path.moveTo(self.points[0])
            for pt in self.points[1:]:
                path.lineTo(pt)
            if self.completed:
                path.lineTo(self.points[0])
                painter.fillPath(path, QBrush(self.fill_color))
            painter.drawPath(path)
            if not self.completed and self.temp_point:
                dash_pen = QPen(self.line_color, 1.5 / lod, Qt.PenStyle.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(self.points[-1], self.temp_point)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        for pt in self.points:
            painter.drawEllipse(pt, r, r)


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
        super().__init__(-size / 2, -size / 2, size, size, parent)

        self.handle_type = handle_type
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
        )  # Keep constant screen size
        self.setAcceptHoverEvents(True)
        self.setCursor(self._get_cursor())

        # Visual style
        self.setBrush(pg.mkBrush(QColor(255, 255, 0, 200)))  # Yellow
        self.setPen(QPen(QColor(0, 0, 0), 2))
        self.setZValue(12)  # Above crop rect

    def _get_cursor(self):
        """Get appropriate cursor for handle position."""
        cursors = {
            "nw": Qt.CursorShape.SizeFDiagCursor,
            "n": Qt.CursorShape.SizeVerCursor,
            "ne": Qt.CursorShape.SizeBDiagCursor,
            "e": Qt.CursorShape.SizeHorCursor,
            "se": Qt.CursorShape.SizeFDiagCursor,
            "s": Qt.CursorShape.SizeVerCursor,
            "sw": Qt.CursorShape.SizeBDiagCursor,
            "w": Qt.CursorShape.SizeHorCursor,
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
        handle_positions = ["nw", "n", "ne", "e", "se", "s", "sw", "w"]
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
            "nw": (x1, y1),
            "n": (cx, y1),
            "ne": (x2, y1),
            "e": (x2, cy),
            "se": (x2, y2),
            "s": (cx, y2),
            "sw": (x1, y2),
            "w": (x1, cy),
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


class ZoomableImageView(pg.GraphicsView):
    mouse_moved = pyqtSignal(int, int)
    crop_changed = pyqtSignal(float, float, float, float)
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
        self.dragging_handle = None
        self.drag_start_pos = None
        self.drag_start_rect = None
        self.crop_rect_item = None
        self.draw_mode = DrawMode.SELECT
        self.temp_roi = None
        self.draw_start = None

    def set_images(self, arrays, opacities=None):
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
        QTimer.singleShot(0, self.reset_zoom)

    def toggle_layer_visibility(self, index, visible):
        if 0 <= index < len(self.image_items):
            self.image_items[index].setVisible(visible)

    def reset_zoom(self):
        if not self.image_items:
            return
        self._vb.autoRange(padding=0)

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
            if self._try_select_resize_handle(event):
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

    def _try_select_resize_handle(self, event):
        if self.crop_rect_item:
            for item in self.items(event.pos()):
                if isinstance(item, ResizeHandle):
                    self.dragging_handle = item
                    scene_pos = self.mapToScene(event.pos())
                    self.drag_start_pos = self._vb.mapSceneToView(scene_pos)
                    self.drag_start_rect = QRectF(self.crop_rect_item.rect())
                    return True
        return False

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
        if self.dragging_handle and self.drag_start_pos and self.drag_start_rect:
            self._resize_crop_rect(view_pos)
            event.accept()
        else:
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
            if self.dragging_handle:
                self._handle_crop_rect_resize_release()
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

    def _handle_crop_rect_resize_release(self):
        self.dragging_handle = None
        self.drag_start_pos = None
        self.drag_start_rect = None
        if self.crop_rect_item:
            rect = self.crop_rect_item.rect()
            self.crop_changed.emit(
                rect.left(), rect.top(), rect.right(), rect.bottom()
            )

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
        self.reset_zoom()
        event.accept()

    def _resize_crop_rect(self, view_pos):
        """Resize crop rectangle based on handle drag."""
        if (
            not self.dragging_handle
            or not self.drag_start_pos
            or not self.drag_start_rect
            or not self.crop_rect_item
        ):
            return

        delta_x = view_pos.x() - self.drag_start_pos.x()
        delta_y = view_pos.y() - self.drag_start_pos.y()

        handle_type = self.dragging_handle.handle_type
        rect = QRectF(self.drag_start_rect)

        # Update rectangle based on which handle is being dragged
        if "n" in handle_type:  # North handles
            new_top = rect.top() + delta_y
            if rect.bottom() - new_top >= self.crop_rect_item.min_size:
                rect.setTop(new_top)

        if "s" in handle_type:  # South handles
            new_bottom = rect.bottom() + delta_y
            if new_bottom - rect.top() >= self.crop_rect_item.min_size:
                rect.setBottom(new_bottom)

        if "w" in handle_type:  # West handles
            new_left = rect.left() + delta_x
            if rect.right() - new_left >= self.crop_rect_item.min_size:
                rect.setLeft(new_left)

        if "e" in handle_type:  # East handles
            new_right = rect.right() + delta_x
            if new_right - rect.left() >= self.crop_rect_item.min_size:
                rect.setRight(new_right)

        self.crop_rect_item.setRect(rect)
        self.crop_changed.emit(rect.left(), rect.top(), rect.right(), rect.bottom())


class CropDialog(QDialog):
    crop_confirmed = pyqtSignal(int, int, int, int)
    bead_crop_confirmed = pyqtSignal(list)

    def __init__(self, images: list[np.ndarray], parent=None, mode: str = "image", beads=None):
        super().__init__(parent)
        self.mode = mode
        self.beads = beads
        self.created_rois = []
        self.original_images = images
        self.crop_rect = None
        self.highlight_rect = None
        self.updating_from_drag = False
        self.preview_images = images

        self._setup_ui()
        self._create_overlay()

        if self.mode == "image":
            h, w = self.preview_images[0].shape
            self.start_x_input.setText("0")
            self.start_y_input.setText("0")
            self.end_x_input.setText(str(w))
            self.end_y_input.setText(str(h))
            self._update_crop_preview()

    def _setup_ui(self):
        self.setWindowTitle("Crop Image" if self.mode == "image" else "Crop Beads")
        self.resize(1200, 800)
        main_layout = QVBoxLayout(self)

        instruction_text = "Mouse wheel: zoom, Drag: pan, Double-click: reset view"
        if self.mode == "image":
            instruction_text += ", Drag handles: resize crop"
        else:
            instruction_text += ", Choose a tool to draw crop regions"
        self.instruction_label = QLabel(instruction_text)
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.coord_label = QLabel("Current Position: X: 0, Y: 0")
        self.coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_view = ZoomableImageView(self)
        self.image_view.setMinimumSize(800, 500)
        self.image_view.mouse_moved.connect(self._on_mouse_move)

        self.auto_contrast_checkbox = QCheckBox("Auto Contrast")
        self.auto_contrast_checkbox.setChecked(False)
        self.auto_contrast_checkbox.stateChanged.connect(self._on_auto_contrast_changed)

        main_layout.addWidget(self.instruction_label)
        main_layout.addWidget(self.coord_label)

        if self.mode == "bead":
            content_layout = QHBoxLayout()
            content_layout.addWidget(self.image_view, stretch=3)

            self.side_panel = QWidget()
            side_layout = QVBoxLayout(self.side_panel)

            tools_group = QGroupBox("ROI Drawing Tools")
            tools_main_layout = QVBoxLayout(tools_group)

            btn_layout = QHBoxLayout()
            self.select_tool_btn = QPushButton("Select")
            self.select_tool_btn.setCheckable(True)
            self.select_tool_btn.setChecked(True)
            self.select_tool_btn.setToolTip("Select (S)")
            self.rect_tool_btn = QPushButton("Rectangle")
            self.rect_tool_btn.setCheckable(True)
            self.rect_tool_btn.setToolTip("Rectangle (R)")
            self.circle_tool_btn = QPushButton("Circle")
            self.circle_tool_btn.setCheckable(True)
            self.circle_tool_btn.setToolTip("Circle (C)")
            self.poly_tool_btn = QPushButton("Lasso")
            self.poly_tool_btn.setCheckable(True)
            self.poly_tool_btn.setToolTip("Lasso (L)")

            self.tool_group = QButtonGroup(self)
            self.tool_group.addButton(self.select_tool_btn)
            self.tool_group.addButton(self.rect_tool_btn)
            self.tool_group.addButton(self.circle_tool_btn)
            self.tool_group.addButton(self.poly_tool_btn)
            self.tool_group.setExclusive(True)

            btn_layout.addWidget(self.select_tool_btn)
            btn_layout.addWidget(self.rect_tool_btn)
            btn_layout.addWidget(self.circle_tool_btn)
            btn_layout.addWidget(self.poly_tool_btn)

            tools_main_layout.addLayout(btn_layout)

            keybind_indicator = QLabel("Keybinds: S = Select | R = Rectangle | C = Circle | L = Lasso")
            keybind_indicator.setStyleSheet("color: gray; font-size: 10px;")
            keybind_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tools_main_layout.addWidget(keybind_indicator)

            self.select_tool_btn.clicked.connect(lambda: self._set_draw_mode(DrawMode.SELECT))
            self.rect_tool_btn.clicked.connect(lambda: self._set_draw_mode(DrawMode.RECT))
            self.circle_tool_btn.clicked.connect(lambda: self._set_draw_mode(DrawMode.CIRCLE))
            self.poly_tool_btn.clicked.connect(lambda: self._set_draw_mode(DrawMode.POLY))

            side_layout.addWidget(tools_group)

            self.show_beads_checkbox = QCheckBox("Show Beads")
            self.show_beads_checkbox.setChecked(True)
            self.show_beads_checkbox.stateChanged.connect(self._toggle_bead_visibility)
            side_layout.addWidget(self.show_beads_checkbox)

            self.beads_count_label = QLabel("Selected Beads: 0 / 0 (0.0%)")
            self.beads_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.beads_count_label.setStyleSheet("font-weight: bold; font-size: 13px;")
            side_layout.addWidget(self.beads_count_label)

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

            side_layout.addWidget(roi_group)
            side_layout.addWidget(self.auto_contrast_checkbox)
            side_layout.addStretch()

            self.side_panel.setLayout(side_layout)
            content_layout.addWidget(self.side_panel, stretch=1)
            main_layout.addLayout(content_layout)

            self.image_view.roi_added.connect(self._on_roi_added)
        else:
            self.image_view.crop_changed.connect(self._on_crop_changed_by_drag)
            main_layout.addWidget(self.image_view)

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
            main_layout.addWidget(crop_group)
            main_layout.addWidget(self.auto_contrast_checkbox)

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
            main_layout.addWidget(self.visibility_groupbox)

        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm Crop")
        self.confirm_button.clicked.connect(self._validate_and_confirm)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _create_overlay(self):
        colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]
        arrays = []
        opacities = []

        for i, img in enumerate(self.preview_images):
            color = colors[i % len(colors)]
            rgba = colorize_grayscale(to_uint8(img), color)
            arrays.append(rgba)

            if i == 0:
                opacities.append(1.0)
            else:
                opacities.append(0.5)

        self.image_view.set_images(arrays, opacities)

        if self.mode == "image":
            self.crop_rect = InteractiveCropRect()
            self.image_view._vb.addItem(self.crop_rect, ignoreBounds=True)
            self.image_view.crop_rect_item = self.crop_rect

            self.highlight_rect = QGraphicsRectItem()
            highlight_pen = QPen(Qt.GlobalColor.yellow, 1, Qt.PenStyle.SolidLine)
            self.highlight_rect.setPen(highlight_pen)
            self.highlight_rect.setZValue(11)
            self.image_view._vb.addItem(self.highlight_rect, ignoreBounds=True)
        else:
            if self.beads is not None and not self.beads.empty:
                self.beads_scatter = pg.ScatterPlotItem(
                    size=4,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(QColor(0, 255, 0, 150))
                )
                self.image_view._vb.addItem(self.beads_scatter, ignoreBounds=True)
                self.beads_scatter.setData(x=self.beads["x"].values, y=self.beads["y"].values)
            self._update_selected_beads_count()

    def _on_mouse_move(self, scene_x, scene_y):
        self.coord_label.setText(f"Current Position: X: {scene_x}, Y: {scene_y}")
        if self.highlight_rect:
            highlight_size = 4
            self.highlight_rect.setRect(
                scene_x - highlight_size / 2,
                scene_y - highlight_size / 2,
                highlight_size,
                highlight_size,
            )

    def _on_visibility_changed(self, index, state):
        visible = state == Qt.CheckState.Checked.value
        self.image_view.toggle_layer_visibility(index, visible)

    def _refresh_pixmaps(self):
        colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]
        for i, img in enumerate(self.preview_images):
            if i >= len(self.image_view.image_items):
                break
            color = colors[i % len(colors)]
            if self.auto_contrast_checkbox.isChecked():
                img_u8 = to_uint8(adjust_contrast(img, 30, 99))
            else:
                img_u8 = to_uint8(img)
            rgba = colorize_grayscale(img_u8, color)
            self.image_view.image_items[i].setImage(
                rgba, autoLevels=False, levels=[0, 255]
            )

    def _on_auto_contrast_changed(self):
        self._refresh_pixmaps()
        if len(self.preview_images) > 1:
            for i, cb in enumerate(self.visibility_checkboxes):
                self.image_view.toggle_layer_visibility(i, cb.isChecked())

    def _on_crop_changed_by_drag(self, x1, y1, x2, y2):
        self.updating_from_drag = True
        self.start_x_input.setText(str(int(x1)))
        self.start_y_input.setText(str(int(y1)))
        self.end_x_input.setText(str(int(x2)))
        self.end_y_input.setText(str(int(y2)))
        width = int(x2 - x1)
        height = int(y2 - y1)
        self.size_label.setText(f"Crop Size: {width} x {height} pixels")
        self.updating_from_drag = False

    def _update_crop_preview(self):
        if self.updating_from_drag:
            return
        try:
            x1 = int(self.start_x_input.text()) if self.start_x_input.text() else 0
            y1 = int(self.start_y_input.text()) if self.start_y_input.text() else 0
            x2 = int(self.end_x_input.text()) if self.end_x_input.text() else 0
            y2 = int(self.end_y_input.text()) if self.end_y_input.text() else 0
            width = x2 - x1
            height = y2 - y1
            if width > 0 and height > 0:
                if self.crop_rect:
                    self.crop_rect.setRect(x1, y1, width, height)
                self.size_label.setText(f"Crop Size: {width} x {height} pixels")
            else:
                self.size_label.setText("Crop Size: Invalid")
        except (ValueError, ZeroDivisionError):
            pass

    def _set_draw_mode(self, mode):
        self.image_view.draw_mode = mode
        if mode == DrawMode.SELECT:
            self.image_view._vb.setMouseMode(pg.ViewBox.PanMode)
            self.image_view.viewport().unsetCursor()
        else:
            self.image_view._vb.setMouseMode(pg.ViewBox.RectMode)
            self.image_view.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _toggle_bead_visibility(self, state):
        visible = state == Qt.CheckState.Checked.value
        if hasattr(self, "beads_scatter") and self.beads_scatter:
            self.beads_scatter.setVisible(visible)

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

        if self.mode == "bead":
            self.select_tool_btn.setChecked(True)
            self._set_draw_mode(DrawMode.SELECT)

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
                    pt_parent = item.mapToParent(pt)
                    pts.append((float(pt_parent.x()), float(pt_parent.y())))
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
            self.beads_count_label.setText(f"Selected Beads: 0 / {len(self.beads)} (0.0%)")
            return
        keep = np.zeros(len(self.beads), dtype=bool)
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
            keep = keep | in_roi
        selected_count = np.sum(keep)
        total_count = len(self.beads)
        pct = (selected_count / total_count * 100) if total_count > 0 else 0.0
        self.beads_count_label.setText(f"Selected Beads: {selected_count} / {total_count} ({pct:.1f}%)")

    def _validate_and_confirm(self):
        if self.mode == "bead":
            rois = self._get_roi_definitions()
            if not rois:
                QMessageBox.warning(
                    self,
                    "No Regions Defined",
                    "Please define at least one crop region before confirming.",
                )
                return
            self.bead_crop_confirmed.emit(rois)
            self.accept()
            return

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

        self.crop_confirmed.emit(x1, y1, x2, y2)
        self.accept()

    def reset_zoom(self, event=None):
        self.image_view.reset_zoom()
        if event:
            event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        from PyQt6.QtWidgets import QLineEdit
        if isinstance(self.focusWidget(), QLineEdit):
            super().keyPressEvent(event)
            return

        if self.mode == "bead":
            key = event.key()
            if key == Qt.Key.Key_S:
                self.select_tool_btn.setChecked(True)
                self._set_draw_mode(DrawMode.SELECT)
                event.accept()
                return
            elif key == Qt.Key.Key_R:
                self.rect_tool_btn.setChecked(True)
                self._set_draw_mode(DrawMode.RECT)
                event.accept()
                return
            elif key == Qt.Key.Key_C:
                self.circle_tool_btn.setChecked(True)
                self._set_draw_mode(DrawMode.CIRCLE)
                event.accept()
                return
            elif key == Qt.Key.Key_L:
                self.poly_tool_btn.setChecked(True)
                self._set_draw_mode(DrawMode.POLY)
                event.accept()
                return

        super().keyPressEvent(event)


def colorize_grayscale(gray_img: np.ndarray, color: str) -> np.ndarray:
    """Colorize grayscale image; black pixels are fully transparent. Returns RGBA uint8 array."""
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
    rgba[:, :, 3] = mask.astype(np.uint8) * 255

    return np.ascontiguousarray(rgba)
