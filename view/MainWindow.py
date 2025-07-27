# views/main_window.py
import os
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
    QPushButton,
    QSizeGrip,
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
            file_dropped_callback=self.handle_dropped_paths
        )
        self.metadata_view = MetadataView()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_table_widget)
        splitter.addWidget(self.metadata_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        self.metadata_view.hide()
        self.load_button = QPushButton("Load Folder")
        self.load_button.clicked.connect(self.on_load_folder)

        self._setup_main_window()

        # Create a container widget for layout
        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(splitter)
        container.setLayout(layout)

        # Connect ViewModel signals to UI slots
        self.vm.file_list_updated.connect(self.update_file_list)
        self.vm.file_status_updated.connect(self.update_file_status)

        # Set the container widget as central widget
        self.setCentralWidget(container)

        self.menuBarUI = MenuBarUI(self)
        self.setMenuBar(self.menuBarUI)
        if sys.platform == "win32":
            self.menuBarUI.installEventFilter(self)

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

    def update_file_status(self, file_path: str, status: FileStatus):
        self.file_table_widget.update_file_status(file_path, status)

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
