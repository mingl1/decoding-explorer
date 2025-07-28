# views/main_window.py
import os
import select
import sys
import warnings
from typing import List

from PyQt6.QtCore import QEvent, QPoint, QRect, Qt, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from model.file_item import FileItem
from model.status_enum import FileStatus
from utils import is_dark_mode
from view.FileListWidget import FileTableWidget
from view.MetadataView import MetadataView
from viewmodel.file_manager_vm import FileManagerVM
from viewmodel.metadata_vm import MetadataVM

warnings.filterwarnings("ignore")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        if sys.platform == "win32":
            self.dragPos = QPoint()
            self.sideGrips = [
                SideGrip(self, Qt.Edge.LeftEdge),
                SideGrip(self, Qt.Edge.TopEdge),
                SideGrip(self, Qt.Edge.RightEdge),
                SideGrip(self, Qt.Edge.BottomEdge),
            ]
            self.cornerGrips = [QSizeGrip(self) for i in range(4)]
            self._gripSize = 8

        self.vm = FileManagerVM()

        self.file_table_widget = FileTableWidget(
            file_dropped_callback=self.handle_dropped_paths, vm=self.vm
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.metadata_vm = MetadataVM()
        self.metadata_view = MetadataView(splitter, vm=self.metadata_vm)
        splitter.addWidget(self.file_table_widget)
        splitter.addWidget(self.metadata_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        self.metadata_view.hide()
        self.load_button = QPushButton("Load Folder")
        self.load_button.clicked.connect(self.on_load_folder)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_alignment)

        self.export_progress_bar = QProgressBar()
        self.export_progress_bar.setVisible(False)

        self._setup_main_window()

        # Create a container widget for layout
        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(5, 0, 0, 5)
        layout.addWidget(self.load_button)
        progress_layout = QHBoxLayout()
        progress_layout.setContentsMargins(5, 0, 5, 0)
        progress_layout.setSpacing(5)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.cancel_button)
        layout.addLayout(progress_layout)
        layout.addWidget(self.export_progress_bar)
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.status_label.setContentsMargins(5, 0, 5, 5)

        layout.addWidget(self.status_label, stretch=0)
        layout.addWidget(splitter, stretch=1)
        container.setLayout(layout)

        # Connect ViewModel signals to UI slots
        self.vm.file_list_updated.connect(self.update_file_list)
        self.vm.file_status_updated.connect(self.update_files_view)
        self.vm.align_progress.connect(self.update_progress)
        self.vm.align_error.connect(self.show_error)
        self.vm.align_complete.connect(self.alignment_finished)
        self.vm.export_progress.connect(self.update_export_progress)
        self.vm.beads_generated.connect(self.save_beads)
        self.vm.bead_progress.connect(self.update_progress)

        self.file_table_widget.itemSelectionChanged.connect(
            self.handle_selection_change
        )
        self.metadata_vm.metadata_applied_sig.connect(self.handle_metadata_applied)
        # !TODO: Cancel shading correction if false
        self.metadata_vm.shading_correction_sig.connect(
            lambda _: self.vm.apply_shading(self.get_selected_files())
        )
        self.metadata_vm.align_channels_sig.connect(self.start_alignment)
        self.metadata_view.export_all_sig.connect(self.vm.export_files)
        self.metadata_view.generate_beads_sig.connect(self.start_bead_generation)

        # Set the container widget as central widget
        self.setCentralWidget(container)

        self.menuBarUI = MenuBarUI(self)
        self.setMenuBar(self.menuBarUI)
        if sys.platform == "win32":
            self.menuBarUI.installEventFilter(self)

    def save_beads(self, beads):
        self.status_label.setText(f"Beads generated: {len(beads)}")
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        file = QFileDialog.getSaveFileName(
            self, "Save Beads Data", "", "Excel Files (*.xlsx), CSV Files (*.csv)"
        )
        if file:
            if file[0].endswith(".xlsx"):
                beads.to_excel(file[0], index=False)
            elif file[0].endswith(".csv"):
                beads.to_csv(file[0], index=False)

    def handle_metadata_applied(self, new_metadata: dict):
        selected_items = self.file_table_widget.selectedItems()
        for item in selected_items:
            if item.column() == 0:
                file_item = item.data(Qt.ItemDataRole.UserRole)
                file_path = file_item.path if isinstance(file_item, FileItem) else None
                file_item.metadata = new_metadata.get(file_path, file_item.metadata)
                assert file_path is not None
                item.setData(Qt.ItemDataRole.UserRole, file_item)

    def get_selected_files(self) -> List[FileItem]:
        selected_items = self.file_table_widget.selectedItems()
        selected_files = []
        for item in selected_items:
            if item.column() == 0:  # Filename column
                file_item = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(file_item, FileItem):
                    selected_files.append(file_item)
        # sort by path to have consistent order
        selected_files.sort(key=lambda x: x.path)
        return selected_files

    def handle_selection_change(self):
        selected_files = self.get_selected_files()
        self.metadata_vm.update_selected_items(selected_files)
        self.vm.selected_files = selected_files
        print(f"Selected {len(selected_files)} files")

    def handle_dropped_paths(self, paths: List[str]):
        for path in paths:
            if os.path.isdir(path):
                self.vm.load_folder(path)
            elif os.path.isfile(path):
                self.vm.load_file(path)

    def on_load_folder(self):
        folder = QFileDialog.getExistingDirectory()
        if folder:
            self.vm.load_folder(folder)

    def start_alignment(self):
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.cancel_button.setVisible(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.cancel_alignment)
        self.vm.align_channels(self.get_selected_files())

    def start_bead_generation(self):
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.cancel_button.setVisible(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.cancel_bead_generation)
        self.vm.generate_beads()

    def update_progress(self, value, message):
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
            self.status_label.setVisible(True)
            self.cancel_button.setVisible(True)
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        if value >= 100:
            self.progress_bar.setVisible(False)
            self.cancel_button.setVisible(False)
            self.status_label.setVisible(False)

    def show_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)

    def alignment_finished(self, aligned_images):
        self.status_label.setText("Alignment complete!")
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        # Do something with aligned_images

    def cancel_alignment(self):
        self.vm.cancel_alignment()
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)

    def cancel_bead_generation(self):
        self.vm.cancel_bead_generation()
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)

    def update_export_progress(self, value, total):
        self.export_progress_bar.setVisible(True)
        self.export_progress_bar.setMaximum(total)
        self.export_progress_bar.setValue(value)
        if value == total:
            self.export_progress_bar.setVisible(False)

    def _setup_main_window(self):
        self.setWindowTitle("Decoding-Explorer")
        if sys.platform == "win32":
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
            )
        self.resize(1280, 800)
        self.setMinimumSize(1200, 800)

    @property
    def gripSize(self):
        return self._gripSize

    def setGripSize(self, size):
        if size == self._gripSize:
            return
        self._gripSize = max(2, size)
        self.updateGrips()

    def updateGrips(self):
        self.setContentsMargins(*[self.gripSize] * 4)

        outRect = self.rect()
        # an "inner" rect used for reference to set the geometries of size grips
        inRect = outRect.adjusted(
            self.gripSize, self.gripSize, -self.gripSize, -self.gripSize
        )

        # top left
        self.cornerGrips[0].setGeometry(QRect(outRect.topLeft(), inRect.topLeft()))
        # top right
        self.cornerGrips[1].setGeometry(
            QRect(outRect.topRight(), inRect.topRight()).normalized()
        )
        # bottom right
        self.cornerGrips[2].setGeometry(
            QRect(inRect.bottomRight(), outRect.bottomRight())
        )
        # bottom left
        self.cornerGrips[3].setGeometry(
            QRect(outRect.bottomLeft(), inRect.bottomLeft()).normalized()
        )

        # left edge
        self.sideGrips[0].setGeometry(0, inRect.top(), self.gripSize, inRect.height())
        # top edge
        self.sideGrips[1].setGeometry(inRect.left(), 0, inRect.width(), self.gripSize)
        # right edge
        self.sideGrips[2].setGeometry(
            inRect.left() + inRect.width(), inRect.top(), self.gripSize, inRect.height()
        )
        # bottom edge
        self.sideGrips[3].setGeometry(
            self.gripSize, inRect.top() + inRect.height(), inRect.width(), self.gripSize
        )

    def resizeEvent(self, event):  # type: ignore
        QMainWindow.resizeEvent(self, event)
        if sys.platform == "win32":
            self.updateGrips()

    def update_file_list(self, file_items: List[FileItem]):
        self.metadata_view.show()
        self.file_table_widget.clearSelection()
        self.file_table_widget.setSortingEnabled(False)  # Disable sorting while adding
        for file_item in file_items:
            self.file_table_widget.add_file_item(file_item)
        self.file_table_widget.setSortingEnabled(True)  # Re-enable sorting
        self.file_table_widget.resizeColumnsToContents()
        header = self.file_table_widget.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        QTimer.singleShot(
            0,
            lambda: header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive),
        )
        header.show()

    def update_files_view(self, files: List[FileItem]):
        self.file_table_widget.update_file_display(files)

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def eventFilter(self, obj, event):  # type: ignore
        if sys.platform == "win32":
            if obj == self.menuBarUI:
                if event.type() == QEvent.Type.MouseButtonPress:
                    self.dragPos = event.globalPosition().toPoint()
                    return False  # Allow the event to propagate for clicks
                elif event.type() == QEvent.Type.MouseMove:
                    if (
                        event.buttons() == Qt.MouseButton.LeftButton
                        and hasattr(self, "dragPos")
                        and self.dragPos is not None
                    ):
                        self.move(
                            self.pos() + event.globalPosition().toPoint() - self.dragPos
                        )
                        self.dragPos = event.globalPosition().toPoint()
                        return True  # Consume the event if dragging
                elif event.type() == QEvent.Type.MouseButtonRelease:
                    self.dragPos = QPoint()  # Reset dragPos
                    return False  # Allow the event to propagate
        return super().eventFilter(obj, event)


class MenuBarUI(QMenuBar):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        if sys.platform == "win32":
            # Window controls
            self.controls_widget = QWidget()
            self.controls_layout = QHBoxLayout(self.controls_widget)
            self.controls_layout.setContentsMargins(0, 10, 0, 0)
            self.controls_layout.setSpacing(0)

            self.minimize_button = QPushButton("—")
            self.maximize_button = QPushButton("☐")
            self.close_button = QPushButton("X")

            self.minimize_button.setFixedSize(30, 30)
            self.maximize_button.setFixedSize(30, 30)
            self.close_button.setFixedSize(30, 30)

            self.minimize_button.clicked.connect(parent.showMinimized)
            self.maximize_button.clicked.connect(parent.toggle_maximize)
            self.close_button.clicked.connect(parent.close)

            self.controls_layout.addWidget(self.minimize_button)
            self.controls_layout.addWidget(self.maximize_button)
            self.controls_layout.addWidget(self.close_button)

            self.setCornerWidget(self.controls_widget, Qt.Corner.TopRightCorner)

            title = "Decoding-Explorer"
            self.title_widget = QWidget()
            self.title_layout = QHBoxLayout(self.title_widget)
            self.title_layout.setContentsMargins(10, 0, 0, 0)
            self.title_layout.setSpacing(0)
            self.title_label = QLabel(title)
            self.title_layout.addWidget(self.title_label)
            self.setCornerWidget(self.title_widget, Qt.Corner.TopLeftCorner)
            self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
            if is_dark_mode():
                self.title_label.setStyleSheet(
                    "font-size: 16px; font-weight: bold; color: white;"
                )
            else:
                self.title_label.setStyleSheet(
                    "font-size: 16px; font-weight: bold; color: black;"
                )


class SideGrip(QWidget):
    def __init__(self, parent, edge):
        QWidget.__init__(self, parent)
        if edge == Qt.Edge.LeftEdge:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.resizeFunc = self.resizeLeft
        elif edge == Qt.Edge.TopEdge:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            self.resizeFunc = self.resizeTop
        elif edge == Qt.Edge.RightEdge:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            self.resizeFunc = self.resizeRight
        else:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            self.resizeFunc = self.resizeBottom
        self.mousePos = None

    def resizeLeft(self, delta):
        window = self.window()
        width = max(window.minimumWidth(), window.width() - delta.x())
        geo = window.geometry()
        geo.setLeft(geo.right() - width)
        window.setGeometry(geo)

    def resizeTop(self, delta):
        window = self.window()
        height = max(window.minimumHeight(), window.height() - delta.y())
        geo = window.geometry()
        geo.setTop(geo.bottom() - height)
        window.setGeometry(geo)

    def resizeRight(self, delta):
        window = self.window()
        width = max(window.minimumWidth(), window.width() + delta.x())
        window.resize(width, window.height())

    def resizeBottom(self, delta):
        window = self.window()
        height = max(window.minimumHeight(), window.height() + delta.y())
        window.resize(window.width(), height)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mousePos = event.pos()

    def mouseMoveEvent(self, event):
        if self.mousePos is not None:
            delta = event.pos() - self.mousePos
            self.resizeFunc(delta)

    def mouseReleaseEvent(self, event):
        self.mousePos = None
