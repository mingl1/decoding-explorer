import os

from PyQt6.QtCore import QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtSvg import QSvgRenderer  # For rendering SVGs
from PyQt6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from model.file_item import FileItem
from utils import resource_path


class FileTableWidget(QTableWidget):
    def __init__(self, file_dropped_callback):
        super().__init__(0, 4)  # columns: path, shape, dtype, status
        self.setAcceptDrops(True)
        self.setSortingEnabled(True)
        self.file_dropped_callback = file_dropped_callback

        self.setHorizontalHeaderLabels(["Filename", "Status", "Shape (CYX)", "Dtype"])
        header = self.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header.setStretchLastSection(True)
        header.hide()
        self.svg_renderer = QSvgRenderer(resource_path("assets/upload.svg"))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            if self.file_dropped_callback:
                self.file_dropped_callback(paths)
            event.acceptProposedAction()

    def add_file_item(self, file_item: FileItem):
        row = self.rowCount()
        self.insertRow(row)
        filename_item = QTableWidgetItem(os.path.basename(file_item.path))
        shape_item = QTableWidgetItem(str(file_item.shape))
        dtype_item = QTableWidgetItem(str(file_item.dtype))
        status_item = QTableWidgetItem(file_item.status.name)
        status_item.setForeground(QColor("white"))
        status_item.setBackground(QColor(file_item.status.color))

        self.setItem(row, 0, filename_item)
        self.setItem(row, 1, status_item)
        self.setItem(row, 2, shape_item)
        self.setItem(row, 3, dtype_item)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.rowCount() == 0:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect = self.viewport().rect()

            # ---- Draw SVG ----
            if self.svg_renderer.isValid():
                svg_size = 64  # size in pixels
                svg_rect = QRectF(
                    rect.center().x() - svg_size // 2,
                    rect.center().y() - svg_size - 10,
                    svg_size,
                    svg_size,
                )
                self.svg_renderer.render(painter, svg_rect)

            # ---- Draw Text ----
            font = QFont()
            font.setItalic(True)
            font.setPointSize(12)
            painter.setFont(font)
            painter.setPen(QColor(150, 150, 150))

            text = "Drop TIFF files or folders here"
            painter.drawText(
                rect.adjusted(0, 30, 0, 0),  # shift down slightly under the SVG
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignCenter,
                text,
            )

    def update_file_status(self, file_path: str, status):
        for row in range(self.rowCount()):
            item = self.item(row, 3)  # Status column
            filename_item = self.item(row, 0)  # Filename column
            if (
                filename_item
                and item
                and filename_item.text() == os.path.basename(file_path)
            ):
                item.setBackground(QColor(status.color))
                break
