import logging
import os

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from crop_anchor_finder import CropAnchorFinder, brightfield_binarize, infer_tile_size
from utils import numpy_to_qimage

logger = logging.getLogger(__name__)

_OVERVIEW_LONG_SIDE = 900
_PREVIEW_SIDE = 260


def _to_pixmap(gray_u8: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(np.ascontiguousarray(gray_u8)))


class CropAnchorDialog(QDialog):
    transform_ready_sig = pyqtSignal(object, dict)

    def __init__(self, reference_img, moving_files, ref_shape, pad=1.1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Crop Anchor")
        self.resize(1100, 780)

        self.reference_img = reference_img
        self.ref_h = int(ref_shape[0])
        self.ref_w = int(ref_shape[1])
        self.pad = float(pad)
        self.default_crop_size = int(round(self.pad * max(self.ref_h, self.ref_w)))

        self.pages_state = []
        for entry in moving_files:
            state = {
                "label": entry["label"],
                "file_item": entry["file_item"],
                "image": entry["image"],
                "candidates": [],
                "selected_index": -1,
                "order_ncc": [],
                "progress_value": 0,
                "approved": False,
                "box_items": [],
                "overview_u8": self._make_overview(entry["image"]),
                "patch_size_value": 0,
                "num_candidates_value": 5,
                "n_features_value": 50000,
                "crop_size_value": self.default_crop_size,
                "finder": None,
            }
            self.pages_state.append(state)

        self.current_page_idx = 0
        self._overview_pixmap_item = QGraphicsPixmapItem()

        self._build_ui()
        self._load_page(0)
        self._show_reference_patch()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self._prev_page)
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._next_page)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.page_label, stretch=1)
        nav_layout.addWidget(self.next_btn)
        main_layout.addLayout(nav_layout)

        content_layout = QHBoxLayout()

        left = QVBoxLayout()
        left_panel = QWidget()
        left_panel.setLayout(left)
        left_panel.setMaximumWidth(280)

        left.addWidget(QLabel("Anchor patch size (px)"))
        self.patch_size = QSpinBox()
        self.patch_size.setRange(0, 20000)
        self.patch_size.setSpecialValueText("Auto")
        self.patch_size.setSingleStep(100)
        self.patch_size.setValue(0)
        self.patch_size.valueChanged.connect(lambda: self._show_reference_patch())
        left.addWidget(self.patch_size)

        left.addWidget(QLabel("Number of candidates"))
        self.num_candidates = QSpinBox()
        self.num_candidates.setRange(1, 20)
        self.num_candidates.setValue(5)
        left.addWidget(self.num_candidates)

        left.addWidget(QLabel("ORB features"))
        self.n_features = QSpinBox()
        self.n_features.setRange(1000, 260000)
        self.n_features.setSingleStep(5000)
        self.n_features.setValue(50000)
        left.addWidget(self.n_features)

        left.addWidget(QLabel("Crop size (px, square)"))
        self.crop_size_px = QSpinBox()
        self.crop_size_px.setRange(1, 200000)
        self.crop_size_px.setSingleStep(100)
        self.crop_size_px.setValue(self.default_crop_size)
        left.addWidget(self.crop_size_px)

        self.find_button = QPushButton("Find candidates")
        self.find_button.clicked.connect(self._start_find)
        left.addWidget(self.find_button)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        left.addWidget(self.progress)

        left.addWidget(QLabel("Ranked Candidates (NCC)"))
        self.list_ncc = QListWidget()
        self.list_ncc.currentRowChanged.connect(lambda r: self._on_list_selected(r))
        left.addWidget(self.list_ncc, stretch=1)

        self.approve_button = QPushButton("Approve & Crop")
        self.approve_button.setEnabled(False)
        self.approve_button.clicked.connect(self._approve)
        left.addWidget(self.approve_button)

        self.close_button = QPushButton("Close Dialog")
        self.close_button.clicked.connect(self.accept)
        left.addWidget(self.close_button)

        content_layout.addWidget(left_panel)

        self.overview_scene = QGraphicsScene(self)
        self.overview_scene.addItem(self._overview_pixmap_item)
        self.overview_view = QGraphicsView(self.overview_scene)
        self.overview_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.overview_view.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.overview_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._ov_zoom = 1.0
        self.overview_view.viewport().installEventFilter(self)
        content_layout.addWidget(self.overview_view, stretch=1)

        right = QVBoxLayout()
        right_panel = QWidget()
        right_panel.setLayout(right)
        right_panel.setMaximumWidth(300)
        right.addWidget(QLabel("Reference top-left"))
        self.ref_label = QLabel()
        self.ref_label.setFixedSize(_PREVIEW_SIDE, _PREVIEW_SIDE)
        self.ref_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.ref_label)
        right.addWidget(QLabel("Crop preview"))
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(_PREVIEW_SIDE, _PREVIEW_SIDE)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.preview_label)
        right.addStretch(1)
        content_layout.addWidget(right_panel)

        main_layout.addLayout(content_layout)

    def _make_overview(self, mov):
        ds = min(1.0, _OVERVIEW_LONG_SIDE / float(max(mov.shape[:2])))
        small = cv2.resize(
            mov,
            (
                max(1, int(mov.shape[1] * ds)),
                max(1, int(mov.shape[0] * ds)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        return brightfield_binarize(small)

    def _save_current_page_state(self):
        if not self.pages_state:
            return
        state = self.pages_state[self.current_page_idx]
        state["patch_size_value"] = self.patch_size.value()
        state["num_candidates_value"] = self.num_candidates.value()
        state["n_features_value"] = self.n_features.value()
        state["crop_size_value"] = self.crop_size_px.value()
        state["progress_value"] = self.progress.value()

    def _load_page(self, idx):
        self._clear_boxes(self.current_page_idx)
        self._save_current_page_state()
        self.current_page_idx = idx
        state = self.pages_state[idx]

        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < len(self.pages_state) - 1)

        filename = os.path.basename(state["file_item"].path)
        self.page_label.setText(
            f"{state['label']} ({filename}) - Image [{idx + 1}/{len(self.pages_state)}]"
        )

        self.patch_size.setValue(state["patch_size_value"])
        self.num_candidates.setValue(state["num_candidates_value"])
        self.n_features.setValue(state["n_features_value"])
        self.crop_size_px.setValue(state["crop_size_value"])
        self.progress.setValue(state["progress_value"])

        finder_running = state.get("finder") is not None and state["finder"].isRunning()
        self.find_button.setEnabled(not finder_running)

        self._show_overview(state["overview_u8"])

        self.list_ncc.blockSignals(True)
        self.list_ncc.clear()
        self.list_ncc.blockSignals(False)

        self._clear_boxes()
        self.preview_label.clear()

        if state["candidates"]:
            self._draw_boxes()
            self._populate_list()
            sel_idx = state["selected_index"]
            if sel_idx >= 0:
                self.list_ncc.blockSignals(True)
                for row_idx, cand_idx in enumerate(state["order_ncc"]):
                    if cand_idx == sel_idx:
                        self.list_ncc.setCurrentRow(row_idx)
                        break
                self.list_ncc.blockSignals(False)
                self._show_preview(state["candidates"][sel_idx])

        self._update_approve_button_state()

    def _update_approve_button_state(self):
        state = self.pages_state[self.current_page_idx]
        if state["approved"]:
            self.approve_button.setText("Aligned & Cropped")
            self.approve_button.setEnabled(False)
        else:
            self.approve_button.setText("Approve & Crop")
            self.approve_button.setEnabled(state["selected_index"] >= 0)

    def _prev_page(self):
        if self.current_page_idx > 0:
            self._load_page(self.current_page_idx - 1)

    def _next_page(self):
        if self.current_page_idx < len(self.pages_state) - 1:
            self._load_page(self.current_page_idx + 1)

    def _show_overview(self, ov_u8):
        self._overview_pixmap_item.setPixmap(_to_pixmap(ov_u8))
        self.overview_scene.setSceneRect(self._overview_pixmap_item.boundingRect())
        self.overview_view.fitInView(
            self._overview_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio
        )

    def _show_reference_patch(self):
        val = self.patch_size.value()
        if val <= 0:
            val = infer_tile_size(self.reference_img)
        s = min(val, self.ref_h, self.ref_w)
        patch = brightfield_binarize(self.reference_img[:s, :s])
        pix = _to_pixmap(patch).scaled(
            _PREVIEW_SIDE,
            _PREVIEW_SIDE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.ref_label.setPixmap(pix)

    def _start_find(self):
        state = self.pages_state[self.current_page_idx]
        if state.get("finder") is not None and state["finder"].isRunning():
            return
        self._show_reference_patch()
        self.find_button.setEnabled(False)
        self.approve_button.setEnabled(False)
        self.list_ncc.clear()
        state["candidates"] = []
        state["selected_index"] = -1
        state["order_ncc"] = []
        self._clear_boxes()
        self.progress.setValue(0)

        page_idx = self.current_page_idx
        finder = CropAnchorFinder(
            self.reference_img,
            state["image"],
            patch_size=self.patch_size.value(),
            num_candidates=self.num_candidates.value(),
            n_features=self.n_features.value(),
        )
        state["finder"] = finder
        finder.progress.connect(
            lambda p, _msg, idx=page_idx: self._on_progress(p, _msg, idx)
        )
        finder.candidates_ready.connect(
            lambda c, idx=page_idx: self._on_candidates_ready(c, idx)
        )
        finder.error.connect(
            lambda msg, idx=page_idx: self._on_error(msg, idx)
        )
        finder.finished.connect(
            lambda idx=page_idx: self._on_finished(idx)
        )
        finder.start()

    def _on_progress(self, p, _msg, page_idx):
        self.pages_state[page_idx]["progress_value"] = p
        if page_idx == self.current_page_idx:
            self.progress.setValue(p)

    def _on_error(self, msg, page_idx):
        logger.error("Crop anchor search failed: %s", msg)
        self.pages_state[page_idx]["progress_value"] = 0
        if page_idx == self.current_page_idx:
            self.progress.setValue(0)
            self.list_ncc.addItem(f"Error: {msg}")

    def _on_finished(self, page_idx):
        if page_idx == self.current_page_idx:
            self.find_button.setEnabled(True)

    def _on_candidates_ready(self, candidates, page_idx):
        state = self.pages_state[page_idx]
        state["candidates"] = candidates or []
        state["order_ncc"] = sorted(
            range(len(state["candidates"])),
            key=lambda i: state["candidates"][i]["score"],
            reverse=True,
        )
        if state["candidates"]:
            state["selected_index"] = state["order_ncc"][0]

        if page_idx == self.current_page_idx:
            self._draw_boxes()
            self._populate_list()
            if state["candidates"]:
                self.list_ncc.setCurrentRow(0)
                self._update_approve_button_state()

    def _populate_list(self):
        state = self.pages_state[self.current_page_idx]
        self.list_ncc.blockSignals(True)
        self.list_ncc.clear()
        for rank, idx in enumerate(state["order_ncc"]):
            c = state["candidates"][idx]
            self.list_ncc.addItem(
                f"#{rank + 1}  ncc={c['score']:.3f}  inl={c.get('inliers', 0)}  "
                f"ratio={c.get('inlier_ratio', 0):.2f}  blob={c.get('blob_fraction', 0):.2f}  "
                f"resid={c.get('residual', 0):.1f}  ang={c['angle']:.1f}°"
            )
        self.list_ncc.blockSignals(False)

    def _crop_size(self):
        size = float(self.crop_size_px.value())
        return size, size

    def _box_polygon(self, cand) -> QPolygonF:
        T = np.asarray(cand["T"], dtype=np.float64).reshape(2, 3)
        cw, ch = self._crop_size()
        corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float64)

        state = self.pages_state[self.current_page_idx]
        ov_ds = min(1.0, _OVERVIEW_LONG_SIDE / float(max(state["image"].shape[:2])))

        poly = QPolygonF()
        for cx, cy in corners:
            px = (T[0, 0] * cx + T[0, 1] * cy + T[0, 2]) * ov_ds
            py = (T[1, 0] * cx + T[1, 1] * cy + T[1, 2]) * ov_ds
            poly.append(QPointF(px, py))
        return poly

    def _clear_boxes(self, page_idx=None):
        if page_idx is None:
            page_idx = self.current_page_idx
        state = self.pages_state[page_idx]
        box_items = state.get("box_items", [])
        for item in box_items:
            self.overview_scene.removeItem(item)
        state["box_items"] = []

    def _draw_boxes(self):
        self._clear_boxes()
        state = self.pages_state[self.current_page_idx]
        box_items = []
        for i, cand in enumerate(state["candidates"]):
            selected = i == state["selected_index"]
            pen = QPen(QColor("#ff3b3b") if selected else QColor("#ffd23b"))
            pen.setWidth(0)
            poly_item = self.overview_scene.addPolygon(
                self._box_polygon(cand), pen, QBrush(Qt.BrushStyle.NoBrush)
            )
            poly_item.setZValue(10)
            box_items.append(poly_item)
            text = self.overview_scene.addText(str(i + 1))
            text.setDefaultTextColor(pen.color())
            first = self._box_polygon(cand).first()
            text.setPos(first.x(), first.y())
            text.setZValue(11)
            box_items.append(text)
        state["box_items"] = box_items

    def _on_list_selected(self, row):
        state = self.pages_state[self.current_page_idx]
        if row < 0 or row >= len(state["order_ncc"]):
            return
        state["selected_index"] = state["order_ncc"][row]
        self._draw_boxes()
        self._show_preview(state["candidates"][state["selected_index"]])
        self._update_approve_button_state()

    def _show_preview(self, cand):
        state = self.pages_state[self.current_page_idx]
        ov_ds = min(1.0, _OVERVIEW_LONG_SIDE / float(max(state["image"].shape[:2])))
        T = np.asarray(cand["T"], dtype=np.float64).reshape(2, 3)
        cw, ch = self._crop_size()
        A = T[:, :2]
        t = T[:, 2] * ov_ds
        inv_a = np.linalg.inv(A)
        M = np.hstack([inv_a, (-inv_a @ t).reshape(2, 1)]).astype(np.float32)
        out_w = max(1, int(cw * ov_ds))
        out_h = max(1, int(ch * ov_ds))
        warped = cv2.warpAffine(state["overview_u8"], M, (out_w, out_h))
        pix = _to_pixmap(warped).scaled(
            _PREVIEW_SIDE,
            _PREVIEW_SIDE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    def _approve(self):
        state = self.pages_state[self.current_page_idx]
        if state["selected_index"] < 0 or state["selected_index"] >= len(
            state["candidates"]
        ):
            return
        cand = state["candidates"][state["selected_index"]]
        cw, ch = self._crop_size()

        self.transform_ready_sig.emit(
            state["file_item"],
            {
                "T": np.asarray(cand["T"], dtype=np.float64).reshape(2, 3),
                "crop_w": int(round(cw)),
                "crop_h": int(round(ch)),
                "angle": cand["angle"],
                "anchor": cand["anchor"],
            },
        )
        state["approved"] = True
        self._update_approve_button_state()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._overview_pixmap_item.pixmap().width():
            self.overview_view.fitInView(
                self._overview_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio
            )
            if self._ov_zoom != 1.0:
                self.overview_view.scale(self._ov_zoom, self._ov_zoom)

    def eventFilter(self, source, event):
        if source == self.overview_view.viewport() and event.type() == event.Type.Wheel:
            zooming_out = event.angleDelta().y() < 0
            if self._ov_zoom <= 0.1 and zooming_out:
                return True
            if self._ov_zoom >= 100 and not zooming_out:
                return True

            zoom_factor = 0.85 if zooming_out else 1.15
            self._ov_zoom *= zoom_factor
            self.overview_view.scale(zoom_factor, zoom_factor)
            return True
        return super().eventFilter(source, event)
