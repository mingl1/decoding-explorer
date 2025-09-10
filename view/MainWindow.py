# views/main_window.py
import os
import select
import sys
import warnings
from typing import List
import numpy as np

from pandas import DataFrame
from PyQt6.QtCore import QEvent, QPoint, QRect, Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
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
from utils import is_dark_mode, find_min_std_partition
from view.CycleAssignmentWidget import CycleAssignmentWidget
from view.FileListWidget import FileTableWidget
from view.MetadataView import MetadataView
from view.roi_inspector import ROI_Inspector
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
        self.vm.file_information_update.connect(self.update_files_view)
        self.vm.align_progress.connect(self.update_progress)
        self.vm.align_error.connect(self.show_error)
        self.vm.align_complete.connect(self.alignment_finished)
        self.vm.export_progress.connect(self.update_export_progress)
        self.vm.beads_generated.connect(self.on_beads_generated)
        self.vm.bead_progress.connect(self.update_progress)
        self.vm.inspect_beads_signal.connect(self.show_roi_inspector_window)

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
        self.metadata_vm.inspect_beads_sig.connect(self.vm.inspect_beads)
        self.metadata_view.protein_files_uploaded.connect(
            self.metadata_vm.set_protein_files
        )
        self.metadata_vm.update_overview_sig.connect(
            self.update_statistics_for_selected
        )
        self.metadata_vm.statistics_updated.connect(
            self.metadata_view.update_statistics
        )
        # Set the container widget as central widget
        self.setCentralWidget(container)

        self.menuBarUI = MenuBarUI(self)
        self.setMenuBar(self.menuBarUI)
        if sys.platform == "win32":
            self.menuBarUI.installEventFilter(self)

    def on_beads_generated(self, beads: DataFrame):
        self.save_beads(beads)
        if self.vm.reference_item:
            self.calculate_statistics_for_file(
                self.vm.reference_item, self.metadata_vm.protein_df
            )

    def calculate_statistics_for_file(
        self, file_item: FileItem, protein_profile: DataFrame
    ):
        if file_item.beads is None or file_item.beads.empty:
            return
        total_beads = len(file_item.beads)

        merge_columns = [col for col in file_item.beads.columns if col.startswith("cy")]
        

        beads_for_merge = file_item.beads.copy()
        for col in merge_columns:
            if col not in beads_for_merge.columns:
                self.show_error(
                    f"Column {col} from protein key not found in bead data. Cannot calculate statistics."
                )
                return

        merged_beads = merge_bead_data_with_protein_profile(
            beads_for_merge, protein_profile, merge_columns
        )
        counts_table = (
            merged_beads.groupby("Protein name")
            .size()
            .reset_index(name="row_count")
            .sort_values("row_count", ascending=False)
        )
        valid_proteins = counts_table[(counts_table["Protein name"] != "Invalid") & (counts_table["Protein name"] != "Filtered")]
        unique_rows = valid_proteins["row_count"].unique().mean()
        error_rate = (counts_table[counts_table["Protein name"] == "Invalid"]['row_count']/unique_rows).item()
        filtered_beads_percentage = (counts_table[counts_table["Protein name"] == "Filtered"]['row_count']/total_beads).item() 
        stats = {
            "total_beads": total_beads,
            "filtered_beads_percentage": float(filtered_beads_percentage)*100,
            "mean_rows": unique_rows,
            "error_rate": float(error_rate)*100,
            "counts_table": counts_table,
        }
        self.metadata_vm.statistics_updated.emit(stats)

    def update_statistics_for_selected(self, protein_profile: DataFrame):
        selected_files = self.get_selected_files()
        for file_item in selected_files:
            if file_item.beads is not None:
                self.calculate_statistics_for_file(file_item, protein_profile)
                break  # Process only the first selected file with beads

    def show_roi_inspector_window(
        self, bright_fields, beads_df, cycles, bboxs, labeled_image, protein_profile
    ):
        if len(labeled_image) == 0:
            labeled_image = None
        bf_image = bright_fields.get("cy0", None)
        if bf_image is None:
            self.show_error("Bright field image for cycle 0 is missing.")
            return
        data = {
            "bf_image": bf_image,
            "beads": beads_df,
            "cycles": cycles,
            # "bboxs": bboxs,
            # "labeled_image": labeled_image,
            "protein_profile": protein_profile,
            "bright_fields": bright_fields,
        }
        self.roi_inspector = ROI_Inspector(data)
        self.roi_inspector.show()

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
        return self.file_table_widget.get_selected_files()

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
                print("loading file:", path)
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
        selected_files = self.get_selected_files()
        if not selected_files:
            self.show_error("No files selected for bead generation.")
            return

        reference_item = self.vm.reference_item
        if not reference_item:
            self.show_error("Please set a reference image first.")
            return

        files_for_assignment = [
            f for f in selected_files if f.path != reference_item.path
        ]

        if not files_for_assignment:
            # If only the reference file is selected, we can proceed with one cycle.
            self.vm.generate_beads({0: reference_item})
            return

        dialog = CycleAssignmentWidget(files_for_assignment, self)
        if dialog.exec():
            assignments_from_dialog = dialog.get_assignments()
            if assignments_from_dialog is None:
                self.show_error("Each file must be assigned to exactly one cycle.")
                return

            # final_assignments = {0: reference_item}
            # final_assignments.update(assignments_from_dialog)

            self.progress_bar.setVisible(True)
            self.status_label.setVisible(True)
            self.cancel_button.setVisible(True)
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.cancel_bead_generation)
            self.vm.generate_beads(assignments_from_dialog)

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
        # popup error message
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        popup_dialog = QDialog(self)
        popup_dialog.setWindowTitle("Error")
        popup_dialog.resize(400, 200)
        layout = QVBoxLayout()
        label = QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(popup_dialog.accept)
        layout.addWidget(ok_button)
        popup_dialog.setLayout(layout)
        popup_dialog.setModal(True)
        popup_dialog.exec()

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
import pandas as pd

def merge_bead_data_with_protein_profile(bead_data, protein_profile, merge_columns):
    """
    Merge bead data with protein profile. If protein_profile is empty,
    create one by grouping combinations and partitioning counts into valid/invalid groups.
    """
    # Convert merge columns to int in both dataframes
    for col in merge_columns:
        bead_data[col] = bead_data[col].astype(int)
        if not protein_profile.empty:
            protein_profile[col] = protein_profile[col].astype(int)
    
    
    # Handle empty protein_profile case
    if protein_profile.empty:
        # Group by merge columns to get count per combination
        combination_counts = bead_data.groupby(merge_columns).size().reset_index(name='count')
        
        # Extract the counts for partitioning
        counts = combination_counts['count'].tolist()
        
        if len(counts) > 1:
            # Use find_min_std_partition to separate into two groups
            groups, min_std = find_min_std_partition(counts)
            
            # Determine which group has lower values (invalid) and higher values (valid)
            # Compare the minimum values of each group instead of means
            group1_min = min(groups[0]) if groups[0] else float('inf')
            group2_min = min(groups[1]) if groups[1] else float('inf')
            
            if group1_min <= group2_min:
                invalid_counts_set = set(groups[0])
                valid_counts_set = set(groups[1])
            else:
                invalid_counts_set = set(groups[1])
                valid_counts_set = set(groups[0])
        else:
            # If only one combination, consider it invalid
            invalid_counts_set = set(counts)
            valid_counts_set = set()
        
        # Create protein profile dataframe
        protein_profile_data = []
        valid_protein_counter = 1
        
        # Assign protein names to each combination
        for _, row in combination_counts.iterrows():
            if row['count'] in invalid_counts_set:
                protein_name = "Invalid"
            else:
                # Assign unique protein names (Protein 1, Protein 2, etc.) to valid combinations
                protein_name = f"Protein {valid_protein_counter}"
                valid_protein_counter += 1
            
            # Create row for protein profile
            profile_row = {}
            for col in merge_columns:
                profile_row[col] = row[col]
            profile_row['Protein name'] = protein_name
            protein_profile_data.append(profile_row)
        
        # Create the protein profile dataframe
        protein_profile = pd.DataFrame(protein_profile_data)
        
    
    # Merge bead_data with protein_profile
    # Create the filtered row
    # filtered_row = {col: 255 for col in merge_columns}
    # filtered_row['Protein name'] = 'Filtered'

    # protein_profile = pd.concat([protein_profile, pd.DataFrame([filtered_row])], ignore_index=True)
    
    bead_data = bead_data.merge(protein_profile, how="left", on=merge_columns)
    
    # Fill NaN values with "Invalid"
    bead_data['Protein name'].fillna("Invalid", inplace=True)
    # For all merge columns being 255
    mask_all_255 = (bead_data[merge_columns] == 255).all(axis=1)
    bead_data.loc[mask_all_255, 'Protein name'] = "Filtered"
    
    return bead_data