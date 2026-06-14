import os
from typing import List

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QPainter
from PyQt6.QtSvg import QSvgRenderer  # For rendering SVGs
from PyQt6.QtWidgets import (
    QFileDialog,
    QHeaderView,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from model.file_item import FileItem
from model.status_enum import FileStatus
from utils import resource_path
from viewmodel.file_manager_vm import FileManagerVM

COLUMN_FILENAME = 0
COLUMN_STATUS = 1
COLUMN_SHAPE = 2
COLUMN_DTYPE = 3
COLUMN_PHYSICAL_SIZE_X = 4
COLUMN_PHYSICAL_SIZE_Y = 5
COLUMN_ALIGNMENT_CHANNEL = 6


def format_shape_by_axes(shape: tuple, axes: str) -> str:
    """Format shape tuple as axis=value strings.

    Args:
        shape: The shape tuple (e.g., (3, 10000, 10000))
        axes: The axes string (e.g., "CYX")

    Returns:
        Formatted string like "C=3, Y=10000, X=10000"
    """
    if not shape:
        return str(shape)

    parts = []
    for axis_char, size in zip(axes, shape):
        parts.append(f"{axis_char}={size}")
    return ", ".join(parts)


class FileTableWidget(QTableWidget):
    # Signal emitted when table becomes empty
    table_emptied = pyqtSignal()

    def __init__(self, file_dropped_callback, vm: FileManagerVM):
        super().__init__(
            0, 7
        )  # columns: filename, status, shape, dtype, physical_size_x, physical_size_y, alignment_channel
        self.setAcceptDrops(True)
        self.setSortingEnabled(True)
        self.file_dropped_callback = file_dropped_callback
        self.vm = vm
        self.setHorizontalHeaderLabels(
            [
                "Filename",
                "Status",
                "Shape",
                "Dtype",
                "PhysicalSizeX",
                "PhysicalSizeY",
                "AlignmentChannel",
            ]
        )
        header = self.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header.setStretchLastSection(True)
        header.hide()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_context_menu)
        self.svg_renderer = QSvgRenderer(resource_path("assets/upload.svg"))
        self.browse_files_button = QPushButton("Browse Files", self)
        self.browse_folder_button = QPushButton("Browse Folder", self)
        self.browse_files_button.clicked.connect(self.browse_files)
        self.browse_folder_button.clicked.connect(self.browse_folder)
        self.cellClicked.connect(self.on_cell_clicked)
        self.update_button_visibility()

    def browse_files(self):
        """Open file dialog to browse for TIFF and CSV files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", "TIFF and CSV Files (*.tif *.tiff *.csv);;TIFF Files (*.tif *.tiff);;CSV Files (*.csv);;All Files (*)"
        )
        if files and self.file_dropped_callback:
            self.file_dropped_callback(files)

    def browse_folder(self):
        """Open folder dialog to browse for a folder containing TIFF files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder and self.file_dropped_callback:
            self.file_dropped_callback([folder])

    def update_button_visibility(self):
        """Show buttons when table is empty, hide when files are present."""
        is_empty = self.rowCount() == 0
        self.browse_files_button.setVisible(is_empty)
        self.browse_folder_button.setVisible(is_empty)

        # Show/hide table header based on whether there are files
        header = self.horizontalHeader()
        if header:
            if is_empty:
                header.hide()
                # Emit signal to notify MainWindow
                self.table_emptied.emit()
            # Note: header.show() is called in MainWindow.update_file_list()

        if is_empty:
            # Position buttons in center of viewport
            self._position_buttons()

    def _position_buttons(self):
        """Position the browse buttons centered in the viewport."""
        viewport = self.viewport()
        if viewport is None:
            return
        rect = viewport.rect()
        center_x = rect.center().x()
        center_y = rect.center().y() + 100  # Below the "or" divider

        button_width = 100
        button_height = 30
        gap = 20

        total_width = button_width * 2 + gap
        start_x = center_x - total_width // 2

        self.browse_files_button.setGeometry(
            start_x, center_y, button_width, button_height
        )
        self.browse_folder_button.setGeometry(
            start_x + button_width + gap, center_y, button_width, button_height
        )

    def resizeEvent(self, event):
        """Handle widget resize to reposition buttons."""
        super().resizeEvent(event)
        if self.rowCount() == 0:
            self._position_buttons()

    def on_cell_clicked(self, row, column):
        if column == COLUMN_STATUS:
            file_item = self.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if not file_item:
                return

            menu = QMenu(self)
            for status in FileStatus:
                if status.name.startswith("_"):
                    continue
                assert isinstance(status.value, str)
                action = QAction(status.value, self)
                action.triggered.connect(
                    lambda checked, s=status: self.change_status(file_item, s)
                )
                menu.addAction(action)

            # Show the menu at the cursor's position
            menu.exec(
                self.viewport().mapToGlobal(
                    self.visualItemRect(self.item(row, column)).bottomLeft()
                )
            )

    def change_status(self, file_item, status):
        self.vm.set_status(file_item, status)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls()]
            print(paths)
            if self.file_dropped_callback:
                self.file_dropped_callback(paths)
            event.acceptProposedAction()

    def add_file_item(self, file_item: FileItem):
        row = self.rowCount()
        self.insertRow(row)
        filename_item = QTableWidgetItem(os.path.basename(file_item.path))
        filename_item.setData(Qt.ItemDataRole.UserRole, file_item)
        shape_text = format_shape_by_axes(file_item.shape, file_item.metadata.axes)
        shape_item = QTableWidgetItem(shape_text)
        dtype_item = QTableWidgetItem(str(file_item.dtype))
        physical_size_x_item = QTableWidgetItem(str(file_item.metadata.PhysicalSizeX))
        physical_size_y_item = QTableWidgetItem(str(file_item.metadata.PhysicalSizeY))
        alignment_channel_item = QTableWidgetItem(
            str(file_item.metadata.reference_channel)
        )

        status_text = file_item.status.value
        status_item = QTableWidgetItem(status_text)
        status_item.setForeground(QColor("white"))

        if self.vm.reference_item and file_item.path == self.vm.reference_item.path:
            filename_item.setBackground(QColor("#1E90FF"))
            status_item.setText(f"{status_text} (Reference)")
        status_item.setBackground(QColor(file_item.status.color))
        self.setItem(row, COLUMN_FILENAME, filename_item)
        self.setItem(row, COLUMN_STATUS, status_item)
        self.setItem(row, COLUMN_SHAPE, shape_item)
        self.setItem(row, COLUMN_DTYPE, dtype_item)
        self.setItem(row, COLUMN_PHYSICAL_SIZE_X, physical_size_x_item)
        self.setItem(row, COLUMN_PHYSICAL_SIZE_Y, physical_size_y_item)
        self.setItem(row, COLUMN_ALIGNMENT_CHANNEL, alignment_channel_item)
        self.update_button_visibility()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.rowCount() == 0:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            rect = self.viewport().rect()

            # ---- Draw SVG ----
            if self.svg_renderer.isValid():
                svg_size = 96  # size in pixels
                svg_rect = QRectF(
                    rect.center().x() - svg_size // 2,
                    rect.center().y() - svg_size - 10,
                    svg_size,
                    svg_size,
                )
                self.svg_renderer.render(painter, svg_rect)

            # ---- Draw Text ----
            font = QFont()
            font.setPointSize(14)
            painter.setFont(font)
            painter.setPen(QColor("#888888"))

            text = "Drop TIFF images, folders, or CSV data here"
            painter.drawText(
                rect.adjusted(0, 30, 0, 0),  # shift down slightly under the SVG
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignCenter,
                text,
            )

            # ---- Draw "or" divider ----
            font.setPointSize(10)
            painter.setFont(font)
            painter.setPen(QColor("#888888"))

            or_text = "or"
            center_y = rect.center().y() + 60
            center_x = rect.center().x()

            # Calculate text width for positioning lines
            font_metrics = painter.fontMetrics()
            or_text_width = font_metrics.horizontalAdvance(or_text)

            # Draw "or" text centered
            or_rect = QRectF(
                center_x - or_text_width / 2,
                center_y - font_metrics.height() / 2,
                or_text_width,
                font_metrics.height(),
            )
            painter.drawText(or_rect, Qt.AlignmentFlag.AlignCenter, "or")

            # Draw horizontal lines on each side of "or"
            line_length = 60
            line_gap = 10  # gap between line and "or" text

            # Left line
            left_line_start_x = center_x - or_text_width / 2 - line_gap - line_length
            left_line_end_x = center_x - or_text_width / 2 - line_gap
            painter.drawLine(
                int(left_line_start_x),
                int(center_y),
                int(left_line_end_x),
                int(center_y),
            )

            # Right line
            right_line_start_x = center_x + or_text_width / 2 + line_gap
            right_line_end_x = center_x + or_text_width / 2 + line_gap + line_length
            painter.drawLine(
                int(right_line_start_x),
                int(center_y),
                int(right_line_end_x),
                int(center_y),
            )

    def update_file_display(self, files: list[FileItem]):
        for file_item in files:
            file_path = file_item.path
            for row in range(self.rowCount()):
                status_item = self.item(row, COLUMN_STATUS)
                filename_item = self.item(row, COLUMN_FILENAME)
                shape_item = self.item(row, COLUMN_SHAPE)
                dtype_item = self.item(row, COLUMN_DTYPE)
                physical_size_x_item = self.item(row, COLUMN_PHYSICAL_SIZE_X)
                physical_size_y_item = self.item(row, COLUMN_PHYSICAL_SIZE_Y)
                alignment_channel_item = self.item(row, COLUMN_ALIGNMENT_CHANNEL)
                if (
                    filename_item
                    and status_item
                    and shape_item
                    and dtype_item
                    and physical_size_x_item
                    and physical_size_y_item
                    and alignment_channel_item
                ):
                    stored_item = filename_item.data(Qt.ItemDataRole.UserRole)
                    stored_path = (
                        stored_item.path if isinstance(stored_item, FileItem) else None
                    )
                    if stored_path != file_path:
                        continue
                    is_reference = (
                        self.vm.reference_item
                        and file_item.path == self.vm.reference_item.path
                    )

                    if is_reference:
                        filename_item.setBackground(QColor("#1E90FF"))

                        status_item.setText(f"{file_item.status.value} (Reference)")
                    else:
                        status_item.setText(file_item.status.value)
                    status_item.setBackground(QColor(file_item.status.color))

                    filename_item.setData(Qt.ItemDataRole.UserRole, file_item)
                    shape_text = format_shape_by_axes(
                        file_item.shape, file_item.metadata.axes
                    )
                    shape_item.setText(shape_text)
                    dtype_item.setText(str(file_item.dtype))
                    physical_size_x_item.setText(str(file_item.metadata.PhysicalSizeX))
                    physical_size_y_item.setText(str(file_item.metadata.PhysicalSizeY))
                    alignment_channel_item.setText(
                        str(file_item.metadata.reference_channel)
                    )
                    print(f"Updated FileItem Information for {file_path}")

    def get_selected_files(self) -> List[FileItem]:
        selected_items = self.selectedItems()
        selected_files = []
        for item in selected_items:
            if item.column() == COLUMN_FILENAME:
                file_item = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(file_item, FileItem):
                    selected_files.append(file_item)

        # remove reference file if selected:
        selected_files = [f for f in selected_files]
        return selected_files

    def open_context_menu(self, position):
        index = self.indexAt(position)
        if not index.isValid():
            return

        row = index.row()
        file_item = self.item(row, 0).data(Qt.ItemDataRole.UserRole)

        menu = QMenu(self)

        is_current_reference = (
            self.vm.reference_item and file_item.path == self.vm.reference_item.path
        )

        if is_current_reference:
            clear_ref_action = QAction("Clear Reference", self)
            clear_ref_action.triggered.connect(self.vm.clear_reference)
            menu.addAction(clear_ref_action)
        else:
            set_ref_action = QAction("Set as Reference Image", self)
            set_ref_action.triggered.connect(lambda: self.vm.set_reference(file_item))
            selected_items = self.get_selected_files()
            if len(selected_items) == 1:
                menu.addAction(set_ref_action)

        selected_items = self.get_selected_files()
        export_files_action = QAction("Export As TIFF", self)

        def export_files():
            folder = QFileDialog.getExistingDirectory()
            if folder:
                self.vm.export_files(folder, selected_items)

        export_files_action.triggered.connect(export_files)
        menu.addAction(export_files_action)

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self.handle_delete_selected)
        menu.addAction(delete_action)

        menu.exec(self.viewport().mapToGlobal(position))

    def handle_delete_selected(self):
        # Get selected files before deleting rows
        selected_files = self.get_selected_files()

        # delete ui rows:
        selection_model = self.selectionModel()
        assert selection_model is not None
        rows = sorted(
            {index.row() for index in selection_model.selectedRows()}, reverse=True
        )
        my_model = self.model()
        assert my_model is not None
        selected_files = self.get_selected_files()

        # delete from vm:
        self.vm.delete_files(selected_files)

        for row in rows:
            my_model.removeRow(row)

        # delete from vm:
        self.vm.delete_files(selected_files)

        # Update button/table visibility after deletion
        self.update_button_visibility()
