# views/main_window.py
import os
import sys
import warnings
from typing import List, Optional

import cv2
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import image_processing
from model.file_item import FileItem
from model.status_enum import FileStatus
from utils import find_min_std_partition, is_dark_mode
from view.alignment_preview_dialog import AlignmentPreviewDialog
from view.crop_dialog import CropDialog
from view.cycle_assignment_dialog import CycleAssignmentDialog
from view.decoding_workflow_panel import DecodingWorkflowPanel
from view.export_selection_dialog import ExportSelectionDialog
from view.file_table_widget import FileTableWidget
from view.roi_inspector import ROIInspector
from viewmodel.file_manager_vm import FileManagerVM, load_image
from viewmodel.metadata_vm import MetadataVM

warnings.filterwarnings("ignore")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        if sys.platform == "win32":
            self.drag_pos = QPoint()
            self.side_grips = [
                SideGrip(self, Qt.Edge.LeftEdge),
                SideGrip(self, Qt.Edge.TopEdge),
                SideGrip(self, Qt.Edge.RightEdge),
                SideGrip(self, Qt.Edge.BottomEdge),
            ]
            self.cornerGrips = [QSizeGrip(self) for i in range(4)]
            self._grip_size = 8

        self.vm = FileManagerVM()
        self._manual_align_assigned_files: list[FileItem] = []
        self._manual_align_layer_labels: list[str] = []

        self.file_table_widget = FileTableWidget(
            file_dropped_callback=self.handle_dropped_paths, vm=self.vm
        )
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.metadata_vm = MetadataVM()
        self.metadata_view = DecodingWorkflowPanel(splitter, vm=self.metadata_vm)
        splitter.addWidget(self.file_table_widget)
        splitter.addWidget(self.metadata_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([2000, 400])  # Give metadata view minimum initial width
        self.metadata_view.setMinimumWidth(400)
        self.metadata_view.hide()

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
        self.vm.manual_align_preview_progress.connect(self.update_progress)
        self.vm.manual_align_preview_loaded.connect(
            self._on_manual_align_preview_loaded
        )
        self.vm.manual_align_preview_error.connect(self._on_manual_align_preview_error)
        self.vm.align_error.connect(self.show_error)
        self.vm.align_complete.connect(self.alignment_finished)
        self.vm.export_progress.connect(self.update_export_progress)
        self.vm.export_complete.connect(self.on_export_complete)
        self.vm.export_error.connect(self.show_error)
        self.vm.beads_generated.connect(self.on_beads_generated)
        self.vm.bead_progress.connect(self.update_progress)
        self.vm.shading_progress.connect(self.update_progress)
        self.vm.shading_complete.connect(self.on_shading_complete)
        self.vm.folder_loaded.connect(self._on_folder_loaded)
        self.vm.folder_loading_progress.connect(self.update_progress)
        self.vm.file_loaded.connect(self._on_file_loaded)
        self.vm.file_loading_progress.connect(self.update_progress)
        self.vm.bead_upload_progress.connect(self.update_progress)
        self.vm.bead_upload_complete.connect(self.on_bead_upload_complete)
        self.vm.inspect_beads_signal.connect(self.show_roi_inspector_window)
        self.vm.dataset_assignment_changed.connect(self.on_dataset_assignment_changed)

        self.file_table_widget.itemSelectionChanged.connect(
            self.handle_selection_change
        )
        self.file_table_widget.table_emptied.connect(self.handle_table_emptied)
        self.metadata_vm.metadata_applied_sig.connect(self.handle_metadata_applied)
        self.metadata_vm.error_sig.connect(self.show_error)
        self.vm.metadata_corrected_sig.connect(self.handle_metadata_corrected)
        self.metadata_vm.align_channels_sig.connect(self.start_alignment)
        self.metadata_view.export_all_sig.connect(self.vm.export_files)
        self.metadata_view.generate_beads_sig.connect(self.start_bead_generation)
        self.metadata_view.upload_beads_sig.connect(self.upload_bead_csv)
        self.metadata_view.assign_cycles_sig.connect(self.assign_cycles)
        self.metadata_view.remove_ensemble_sig.connect(
            self.remove_ensemble_applied_changes
        )
        self.metadata_view.lower_invalid_sig.connect(self.lower_invalid_ratio)
        self.metadata_view.lower_filter_sig.connect(self.lower_filter_ratio)
        self.metadata_view.export_sig.connect(self.start_export_flow)
        self.metadata_view.manually_align_sig.connect(self.start_manual_alignment)
        self.metadata_view.crop_selected_sig.connect(self.start_crop)
        self.metadata_view.crop_beads_sig.connect(self.start_bead_crop)
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
        self.metadata_vm.file_shape_update_sig.connect(
            self.vm.file_information_update.emit
        )
        # Set the container widget as central widget
        self.setCentralWidget(container)

        self.menu_bar_ui = MenuBarUI(self)
        self.setMenuBar(self.menu_bar_ui)
        if sys.platform == "win32":
            self.menu_bar_ui.installEventFilter(self)
        self.on_dataset_assignment_changed(
            self.vm.dataset_assignment_valid, self.vm.dataset_assignment_reason
        )

    def on_beads_generated(self, beads: DataFrame):
        if self.vm.reference_item:
            self._refresh_ensemble_controls(self.vm.reference_item)
            self.calculate_statistics_for_file(
                self.vm.reference_item, self.metadata_vm.protein_df
            )
            self.metadata_view.collapse_processing_sections()

    def _refresh_ensemble_controls(self, file_item: FileItem):
        self.metadata_view.set_ensemble_sweep_stats(
            file_item.ensemble_sweep_stats,
            selected_ratio=file_item.ensemble_ratio_selected,
            applied_ratio=file_item.ensemble_ratio_applied,
        )

    def calculate_statistics_for_file(
        self, file_item: FileItem, protein_profile: DataFrame
    ):
        if file_item.beads is None or file_item.beads.empty:
            return

        beads = file_item.beads
        if file_item.bead_crop_bounds is not None:
            x1, y1, x2, y2 = file_item.bead_crop_bounds
            beads = beads[
                (beads["x"] >= x1)
                & (beads["x"] < x2)
                & (beads["y"] >= y1)
                & (beads["y"] < y2)
            ]

        total_beads = len(beads)

        merge_columns = [col for col in beads.columns if col.startswith("cy")]

        beads_for_merge = beads.copy()
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
        valid_proteins = counts_table[
            (counts_table["Protein name"] != "Invalid")
            & (counts_table["Protein name"] != "Filtered")
            & (counts_table["Protein name"].str.strip() != "")
        ]
        unique_rows = valid_proteins["row_count"].unique().mean()
        try:
            error_rate = (
                counts_table[counts_table["Protein name"] == "Invalid"]["row_count"]
                / unique_rows
            ).item()
        except:
            error_rate = 0
        try:
            filtered_beads_percentage = (
                counts_table[counts_table["Protein name"] == "Filtered"]["row_count"]
                / total_beads
            ).item()
        except:
            filtered_beads_percentage = 0
        stats = {
            "total_beads": total_beads,
            "filtered_beads_percentage": float(filtered_beads_percentage) * 100,
            "mean_rows": unique_rows,
            "error_rate": float(error_rate) * 100,
            "counts_table": valid_proteins,
        }
        self.metadata_vm.statistics_updated.emit(stats)

    def update_statistics_for_selected(self, protein_profile: DataFrame):
        reference_item = self.vm.reference_item
        if (
            reference_item is not None
            and reference_item.beads is not None
            and not reference_item.beads.empty
        ):
            self.calculate_statistics_for_file(reference_item, protein_profile)
            return
        selected_files = self.get_selected_files()
        for file_item in selected_files:
            if file_item.beads is not None:
                self.calculate_statistics_for_file(file_item, protein_profile)
                break

    def _get_table_files_in_order(self) -> list[FileItem]:
        files = []
        for row in range(self.file_table_widget.rowCount()):
            item = self.file_table_widget.item(row, 0)
            if item is None:
                continue
            file_item = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(file_item, FileItem):
                files.append(file_item)
        return files

    def on_dataset_assignment_changed(self, is_ready: bool, message: str):
        self.metadata_view.set_dataset_status(message)
        self.metadata_view.set_processing_visible(is_ready)
        if is_ready:
            dataset_files = self.vm.get_dataset_files_ordered()
            protein_file = self.vm.get_dataset_protein_file()
            if protein_file is not None and protein_file.path not in {
                file_item.path for file_item in dataset_files
            }:
                dataset_files.append(protein_file)
            self.metadata_vm.update_selected_items(dataset_files)
        else:
            self.metadata_vm.update_selected_items([])
            self.metadata_view.set_ensemble_sweep_stats(None)

    def assign_cycles(self):
        files_in_order = self._get_table_files_in_order()
        if len(files_in_order) == 0:
            self.show_error("Load at least one file before assigning cycles.")
            return

        if self.vm.is_dataset_ready():
            existing = self.vm.get_dataset_cycle_assignments() or {}
            initial_assignments = {}
            for cycle_num, file_item in existing.items():
                if cycle_num == 0:
                    initial_assignments[1] = file_item
                else:
                    initial_assignments[cycle_num + 1] = file_item
            initial_protein_file = self.vm.get_dataset_protein_file()
        else:
            initial_assignments = None
            initial_protein_file = None
        dialog = CycleAssignmentDialog(
            files_in_order,
            self,
            initial_assignments=initial_assignments,
            initial_protein_file=initial_protein_file,
            default_cycle_count=2,
        )
        if not dialog.exec():
            return

        assignments_from_dialog = dialog.get_assignments()
        if assignments_from_dialog is None:
            self.show_error("Each file must be assigned to exactly one cycle.")
            return

        reference_item = assignments_from_dialog.get(1)
        if reference_item is None:
            self.show_error("Cycle 1 must be assigned.")
            return
        if (
            self.vm.reference_item is None
            or self.vm.reference_item.path != reference_item.path
        ):
            self.vm.set_reference(reference_item)
        protein_file = dialog.get_protein_file()

        final_assignments = {0: reference_item}
        for cycle_num, file_item in assignments_from_dialog.items():
            if cycle_num == 1:
                continue
            final_assignments[cycle_num - 1] = file_item
        is_valid = self.vm.set_dataset_cycle_assignments(
            final_assignments,
            protein_file=protein_file,
        )
        if not is_valid:
            self.show_error(self.vm.dataset_assignment_reason)

    def _get_required_dataset_assignments(self) -> dict[int, FileItem] | None:
        if not self.vm.is_dataset_ready():
            self.show_error(self.vm.dataset_assignment_reason)
            return None
        assignments = self.vm.get_dataset_cycle_assignments()
        if assignments is None:
            self.show_error("Assign cycles to continue.")
            return None
        return assignments

    def show_roi_inspector_window(
        self,
        bright_fields,
        beads_df,
        cycles,
        bboxs,
        labeled_image,
        protein_profile,
        max_size,
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
            "max_size": max_size,
        }
        self.roi_inspector = ROIInspector(data)
        self.roi_inspector.show()

    def _get_cropped_beads_for_export(self, reference_item: FileItem):
        beads = reference_item.beads
        if beads is None or beads.empty:
            return None
        if reference_item.bead_crop_bounds is not None:
            x1, y1, x2, y2 = reference_item.bead_crop_bounds
            beads = beads[
                (beads["x"] >= x1)
                & (beads["x"] < x2)
                & (beads["y"] >= y1)
                & (beads["y"] < y2)
            ]
        return beads

    def recompute_ensemble_sweep(self, start: float, end: float, step: float):
        reference_item = self.vm.reference_item
        if reference_item is None:
            self.show_error("Please set a reference image first.")
            return
        try:
            self.vm.recompute_ensemble_sweep(reference_item, start, end, step)
            self._refresh_ensemble_controls(reference_item)
        except Exception as e:
            self.show_error(str(e))

    def apply_ensemble_ratio(self, ratio: float):
        reference_item = self.vm.reference_item
        if reference_item is None:
            self.show_error("Please set a reference image first.")
            return
        try:
            self.vm.apply_ensemble_ratio(reference_item, ratio)
            self._refresh_ensemble_controls(reference_item)
            if reference_item.beads is not None and not reference_item.beads.empty:
                self.calculate_statistics_for_file(
                    reference_item, self.metadata_vm.protein_df
                )
        except Exception as e:
            self.show_error(str(e))

    def _resolve_ensemble_ratio_base(self, reference_item: FileItem) -> float:
        if reference_item.ensemble_ratio_applied is not None:
            return float(reference_item.ensemble_ratio_applied)
        selected_ratio = self.metadata_view.get_selected_ensemble_ratio()
        if selected_ratio is not None:
            return float(selected_ratio)
        return float(image_processing.DEFAULT_ENSEMBLE_RATIO_START)

    def _apply_ensemble_ratio_delta(self, delta: float):
        reference_item = self.vm.reference_item
        if reference_item is None:
            self.show_error("Please set a reference image first.")
            return
        try:
            base_ratio = self._resolve_ensemble_ratio_base(reference_item)
            ratio = round(base_ratio + float(delta), 2)
            self.vm.apply_ensemble_ratio(reference_item, ratio)
            self._refresh_ensemble_controls(reference_item)
            if reference_item.beads is not None and not reference_item.beads.empty:
                self.calculate_statistics_for_file(
                    reference_item, self.metadata_vm.protein_df
                )
        except Exception as e:
            self.show_error(str(e))

    def lower_invalid_ratio(self):
        self._apply_ensemble_ratio_delta(0.05)

    def lower_filter_ratio(self):
        self._apply_ensemble_ratio_delta(-0.05)

    def _apply_pending_ensemble_for_export(self, reference_item: FileItem) -> bool:
        selected_ratio = self.metadata_view.get_selected_ensemble_ratio()
        applied_ratio = reference_item.ensemble_ratio_applied
        if (
            selected_ratio is not None
            and applied_ratio is not None
            and abs(float(selected_ratio) - float(applied_ratio)) > 1e-9
            and reference_item.pre_ensemble_beads is not None
            and reference_item.ensemble_cache is not None
        ):
            try:
                self.vm.apply_ensemble_ratio(reference_item, float(selected_ratio))
                self._refresh_ensemble_controls(reference_item)
            except Exception as e:
                self.show_error(str(e))
                return False
        return True

    def start_export_flow(self):
        assignments = self._get_required_dataset_assignments()
        if assignments is None:
            return
        reference_item = assignments.get(0)
        if reference_item is None:
            self.show_error("Cycle 1 must be assigned.")
            return
        protein_file = self.vm.get_dataset_protein_file()
        tiff_options = []
        for cycle_num in sorted(assignments.keys()):
            file_item = assignments[cycle_num]
            if cycle_num == 0:
                label = "Cycle 1 (Reference)"
            else:
                label = f"Cycle {cycle_num + 1}"
            checked = protein_file is None and cycle_num == 0
            tiff_options.append((label, file_item, checked))
        if protein_file is not None:
            tiff_options = [
                (label, file_item, False) for label, file_item, _ in tiff_options
            ]
            tiff_options.append(("Protein TIFF", protein_file, True))

        beads_available = (
            reference_item.beads is not None and not reference_item.beads.empty
        )
        dialog = ExportSelectionDialog(
            tiff_options=tiff_options,
            beads_enabled=beads_available,
            beads_checked=beads_available,
            beads_format="csv",
            parent=self,
        )
        if not dialog.exec():
            return

        selected_tiff_files = dialog.get_selected_tiff_files()
        export_beads = dialog.should_export_beads()
        if len(selected_tiff_files) == 0 and not export_beads:
            self.show_error("Select at least one export item.")
            return

        folder = QFileDialog.getExistingDirectory()
        if not folder:
            return

        beads = None
        beads_format = "csv"
        if export_beads:
            if not self._apply_pending_ensemble_for_export(reference_item):
                return
            beads = self._get_cropped_beads_for_export(reference_item)
            if beads is None or beads.empty:
                self.show_error("No beads available. Generate or upload beads first.")
                return
            beads_format = dialog.get_beads_format()

        if len(selected_tiff_files) > 0:
            self.vm.export_files(folder, selected_tiff_files)

        if export_beads and beads is not None:
            output_path = os.path.join(folder, f"beads_result.{beads_format}")
            if beads_format == "xlsx":
                beads.to_excel(output_path, index=False)
            else:
                beads.to_csv(output_path, index=False)

        if len(selected_tiff_files) == 0:
            self.on_export_complete()

    def remove_ensemble_applied_changes(self):
        reference_item = self.vm.reference_item
        if reference_item is None:
            self.show_error("Please set a reference image first.")
            return
        try:
            self.vm.remove_ensemble_applied_changes(reference_item)
            self._refresh_ensemble_controls(reference_item)
            if reference_item.beads is not None and not reference_item.beads.empty:
                self.calculate_statistics_for_file(
                    reference_item, self.metadata_vm.protein_df
                )
        except Exception as e:
            self.show_error(str(e))

    def handle_metadata_applied(self, new_metadata: dict):
        dataset_files = self.vm.get_dataset_files_ordered()
        protein_file = self.vm.get_dataset_protein_file()
        if protein_file is not None and protein_file.path not in {
            file_item.path for file_item in dataset_files
        }:
            dataset_files.append(protein_file)
        if len(dataset_files) == 0:
            dataset_files = self.get_selected_files()
        self.vm.apply_metadata(new_metadata, dataset_files)

    def handle_metadata_corrected(self, corrected_values: dict):
        for key, value in corrected_values.items():
            self.metadata_vm.update_corrected_metadata(key, value)

    def get_selected_files(self) -> List[FileItem]:
        return self.file_table_widget.get_selected_files()

    def handle_selection_change(self):
        selected_files = self.get_selected_files()
        self.vm.selected_files = selected_files
        if not self.vm.is_dataset_ready():
            self.metadata_vm.update_selected_items(selected_files)
        print(f"Selected {len(selected_files)} files")

    def handle_table_emptied(self):
        """Hide metadata view when no files remain in the table."""
        self.metadata_view.hide()

    def handle_dropped_paths(self, paths: List[str]):
        dirs = []
        files = []
        for path in paths:
            if os.path.isdir(path):
                dirs.append(path)
            elif os.path.isfile(path):
                print("loading file:", path)
                files.append(path)
        self.vm.load_folder(dirs)
        self.vm.load_file(files)

    def on_load_folder(self):
        folder = QFileDialog.getExistingDirectory()
        if folder:
            self.vm.load_folder(folder)

    def start_alignment(self):
        assignments = self._get_required_dataset_assignments()
        if assignments is None:
            return
        selected_files = [
            file_item
            for cycle_num, file_item in sorted(assignments.items(), key=lambda x: x[0])
            if cycle_num != 0
        ]
        protein_file = self.vm.get_dataset_protein_file()
        if protein_file is not None:
            if protein_file.path not in [f.path for f in selected_files]:
                selected_files.append(protein_file)
        if len(selected_files) == 0:
            self.show_error("Assign at least one cycle besides the reference.")
            return

        if self.metadata_view.apply_shading_checkbox.isChecked():
            self.progress_bar.setVisible(True)
            self.status_label.setVisible(True)
            self.cancel_button.setVisible(True)
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.cancel_shading)
            self.vm.selected_files = selected_files
            self.vm.apply_shading(selected_files)
        else:
            self.progress_bar.setVisible(True)
            self.status_label.setVisible(True)
            self.cancel_button.setVisible(True)
            self.cancel_button.clicked.disconnect()
            self.cancel_button.clicked.connect(self.cancel_alignment)
            self.vm.align_channels(selected_files)

    def cancel_shading(self):
        self.vm.cancel_shading()
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)

    def start_bead_generation(self):
        assignments = self._get_required_dataset_assignments()
        if assignments is None:
            return

        # Get StarDist settings from the metadata view
        stardist_settings = self.metadata_view.get_stardist_settings()
        use_stardist = stardist_settings["use_stardist"]
        model_name = stardist_settings["model_name"]
        use_guess_tiles = stardist_settings["use_guess_tiles"]
        stardist_n_tiles = stardist_settings["n_tiles"]
        use_stardist_bead_centers = stardist_settings["use_stardist_bead_centers"]
        area_multiplier = stardist_settings["area_multiplier"]
        ensemble_ratio_start = image_processing.DEFAULT_ENSEMBLE_RATIO_START
        ensemble_ratio_end = image_processing.DEFAULT_ENSEMBLE_RATIO_END
        ensemble_ratio_step = image_processing.DEFAULT_ENSEMBLE_RATIO_STEP

        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.cancel_button.setVisible(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.cancel_bead_generation)
        self.vm.generate_beads(
            assignments,
            use_stardist=use_stardist,
            model_name=model_name,
            stardist_use_guess_tiles=use_guess_tiles,
            stardist_n_tiles=stardist_n_tiles,
            use_stardist_bead_centers=use_stardist_bead_centers,
            area_multiplier=area_multiplier,
            ensemble_ratio_start=ensemble_ratio_start,
            ensemble_ratio_end=ensemble_ratio_end,
            ensemble_ratio_step=ensemble_ratio_step,
        )

    def upload_bead_csv(self):
        assignments = self._get_required_dataset_assignments()
        if assignments is None:
            return
        reference_item = assignments[0]

        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Bead CSV File",
            "",
            "CSV Files (*.csv)",
        )
        if not csv_path:
            return

        is_valid, error_msg, num_cycles = self.vm.validate_bead_csv(csv_path)
        if not is_valid:
            self.show_error(f"Invalid CSV file: {error_msg}")
            return
        if num_cycles != len(assignments):
            self.show_error(
                f"CSV has {num_cycles} cycle column(s), but current assignment has {len(assignments)} cycle(s). Reassign cycles."
            )
            return
        self.vm.store_uploaded_beads(csv_path, reference_item, assignments)

    def on_bead_upload_complete(self, reference_item):
        print(
            f"on_bead_upload_complete called, beads: {reference_item.beads is not None}, empty: {reference_item.beads.empty if reference_item.beads is not None else 'N/A'}"
        )
        self._refresh_ensemble_controls(reference_item)
        if reference_item.beads is not None and not reference_item.beads.empty:
            self.calculate_statistics_for_file(
                reference_item, self.metadata_vm.protein_df
            )
            self.metadata_view.collapse_processing_sections()

    def start_manual_alignment(self):
        assignments = self._get_required_dataset_assignments()
        if assignments is None:
            return
        reference_item = assignments[0]

        assigned_pairs = [
            (f"Cycle {cycle_num + 1}", file_item)
            for cycle_num, file_item in sorted(assignments.items(), key=lambda x: x[0])
            if cycle_num != 0
        ]
        protein_file = self.vm.get_dataset_protein_file()
        if protein_file is not None:
            assigned_pairs.append(("Protein", protein_file))
        if not assigned_pairs:
            self.show_error("Assign at least one cycle besides the reference.")
            return

        self._manual_align_assigned_files = [
            file_item for _, file_item in assigned_pairs
        ]
        self._manual_align_layer_labels = [label for label, _ in assigned_pairs]
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.cancel_button.setVisible(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.cancel_manual_align_preview_loading)
        self.vm.load_manual_align_preview_images(
            reference_item, self._manual_align_assigned_files
        )

    def _on_manual_align_preview_loaded(self, loaded_images: object):
        expected_count = 1 + len(self._manual_align_assigned_files)
        if not isinstance(loaded_images, list) or len(loaded_images) != expected_count:
            self._restore_cancel_to_alignment()
            self.show_error("Could not load all images for manual alignment preview.")
            self._manual_align_assigned_files = []
            self._manual_align_layer_labels = []
            return

        target_image = loaded_images[0]
        moving_images = loaded_images[1:]
        assigned_files = list(self._manual_align_assigned_files)
        layer_labels = list(self._manual_align_layer_labels)
        self._restore_cancel_to_alignment()

        preview_dialog = AlignmentPreviewDialog(
            target_image,
            moving_images,
            can_edit=True,
            can_emit=True,
            initial_preview_size=2000,
            initial_checked_indices=[0] if moving_images else [],
            layer_labels=layer_labels,
        )
        preview_dialog.assigned_files = assigned_files
        preview_dialog.transformation_matrices.connect(
            lambda matrices: self._on_manual_alignment_complete(
                matrices, assigned_files
            )
        )
        preview_dialog.exec()
        self._manual_align_assigned_files = []
        self._manual_align_layer_labels = []

    def _on_manual_align_preview_error(self, message: str):
        self._restore_cancel_to_alignment()
        self._manual_align_assigned_files = []
        self._manual_align_layer_labels = []
        if message:
            self.show_error(message)

    def _restore_cancel_to_alignment(self):
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.cancel_alignment)

    def _on_manual_alignment_complete(
        self,
        transformation_matrices: list[Optional[np.ndarray]],
        assigned_files: list[FileItem],
    ):
        """Handle completion of manual alignment."""
        if len(transformation_matrices) != len(assigned_files):
            self.show_error(
                "Mismatch between transformation matrices and assigned files."
            )
            return

        to_be_updated = []
        for i, file_item in enumerate(assigned_files):
            transf_matrix = transformation_matrices[i]
            if transf_matrix is None:
                continue
            my_f = self.vm.files.get(file_item.path)
            if not my_f:
                continue

            # Get the full multi-channel image
            full_image = (
                my_f.working_image
                if my_f.working_image is not None
                else load_image(my_f)
            )

            # Apply the transformation to all channels
            if len(full_image.shape) == 2:
                # Single channel image
                h, w = full_image.shape
                my_f.working_image = cv2.warpAffine(full_image, transf_matrix, (w, h))
            else:
                # Multi-channel image - apply transformation to all channels
                num_channels, h, w = full_image.shape
                transformed_full = np.zeros_like(full_image)

                for ch in range(num_channels):
                    transformed_full[ch] = cv2.warpAffine(
                        full_image[ch], transf_matrix, (w, h)
                    )

                my_f.working_image = transformed_full

            my_f.status = FileStatus.ALIGNED
            my_f.metadata.prefix = FileStatus.ALIGNED.name.lower()
            to_be_updated.append(my_f)

        if to_be_updated:
            self.vm.file_information_update.emit(to_be_updated)
            QMessageBox.information(
                self,
                "Alignment Complete",
                f"Successfully aligned {len(to_be_updated)} image(s).",
            )

    def start_crop(self):
        assignments = self._get_required_dataset_assignments()
        if assignments is None:
            return
        assigned_files = [
            file_item
            for _, file_item in sorted(assignments.items(), key=lambda x: x[0])
        ]
        if len(assigned_files) == 1:
            file_item = assigned_files[0]
            image = self.vm._get_brightfield_image(file_item, use_original=True)
            if image is None:
                self.show_error("Could not load image.")
                return

            dialog = CropDialog([image], self)
            dialog.crop_confirmed.connect(
                lambda x1, y1, x2, y2: self._apply_crop_single(
                    file_item, x1, y1, x2, y2
                )
            )
            dialog.exec()
            return

        images = []
        for file_item in assigned_files:
            img = self.vm._get_brightfield_image(file_item, use_original=True)
            if img is None:
                self.show_error(f"Could not load {os.path.basename(file_item.path)}")
                return
            images.append(img)

        crop_dialog = CropDialog(images, self)
        crop_dialog.crop_confirmed.connect(
            lambda x1, y1, x2, y2: self._apply_crop_multiple(
                assigned_files, x1, y1, x2, y2
            )
        )
        crop_dialog.exec()

    def start_bead_crop(self):
        """Open CropDialog to select a region for filtering beads."""
        reference_item = self.vm.reference_item
        if not reference_item:
            self.show_error("Please set a reference image first.")
            return

        if reference_item.beads is None or reference_item.beads.empty:
            self.show_error("No beads available. Generate or upload beads first.")
            return

        # Collect brightfield images from cycles
        images = []
        if reference_item.cycles:
            for key in sorted(reference_item.cycles.keys()):
                cycle_image = reference_item.cycles[key]
                if len(cycle_image.shape) == 2:
                    images.append(cycle_image)
                    continue
                if len(cycle_image.shape) == 3:
                    bf_channel = 0
                    if reference_item.cycle_files:
                        try:
                            cycle_num = int(key.replace("cy", ""))
                            cycle_file = reference_item.cycle_files.get(cycle_num)
                            if cycle_file:
                                bf_channel = int(cycle_file.metadata.reference_channel)
                        except ValueError:
                            pass
                    if bf_channel >= cycle_image.shape[0]:
                        bf_channel = 0
                    images.append(cycle_image[bf_channel])
        elif reference_item.cycle_files:
            for cycle_num in sorted(reference_item.cycle_files.keys()):
                img = self.vm._get_brightfield_image(
                    reference_item.cycle_files[cycle_num]
                )
                if img is not None:
                    images.append(img)

        if not images:
            # Fallback to reference brightfield
            img = self.vm._get_brightfield_image(reference_item)
            if img is not None:
                images.append(img)

        if not images:
            self.show_error("Could not load brightfield images for crop.")
            return

        total_before = len(reference_item.beads)
        dialog = CropDialog(images, self)
        dialog.crop_confirmed.connect(
            lambda x1, y1, x2, y2: self._apply_bead_crop(
                reference_item, x1, y1, x2, y2, total_before
            )
        )
        dialog.exec()

    def _apply_bead_crop(
        self,
        reference_item: FileItem,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        total_before: int,
    ):
        """Store bead crop bounds and recalculate statistics."""
        reference_item.bead_crop_bounds = (x1, y1, x2, y2)
        beads = reference_item.beads
        filtered = beads[
            (beads["x"] >= x1)
            & (beads["x"] < x2)
            & (beads["y"] >= y1)
            & (beads["y"] < y2)
        ]
        total_after = len(filtered)
        self.calculate_statistics_for_file(reference_item, self.metadata_vm.protein_df)
        QMessageBox.information(
            self,
            "Bead Crop Applied",
            f"Beads: {total_before} → {total_after} (cropped {total_before - total_after})",
        )

    def _refresh_metadata_for_dataset(self):
        if not self.vm.is_dataset_ready():
            return
        dataset_files = self.vm.get_dataset_files_ordered()
        protein_file = self.vm.get_dataset_protein_file()
        if protein_file is not None and protein_file.path not in {
            f.path for f in dataset_files
        }:
            dataset_files.append(protein_file)
        self.metadata_vm.update_selected_items(dataset_files)

    def _apply_crop_single(
        self, file_item: FileItem, x1: int, y1: int, x2: int, y2: int
    ):
        """Apply crop to single file."""
        self.vm.apply_crop([file_item], x1, y1, x2, y2)
        self._refresh_metadata_for_dataset()
        QMessageBox.information(self, "Crop Complete", "Image cropped successfully.")

    def _apply_crop_multiple(
        self, file_items: list[FileItem], x1: int, y1: int, x2: int, y2: int
    ):
        """Apply crop to multiple files."""
        self.vm.apply_crop(file_items, x1, y1, x2, y2)
        self._refresh_metadata_for_dataset()
        QMessageBox.information(
            self, "Crop Complete", f"Successfully cropped {len(file_items)} image(s)."
        )

    def update_progress(self, value, message):
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
            self.status_label.setVisible(True)
            self.cancel_button.setVisible(True)
        if value < 0:
            self.progress_bar.setVisible(False)
            self.cancel_button.setVisible(False)
            self.status_label.setVisible(False)
            self.show_error(message)
            return

        clamped_value = max(0, min(100, int(value)))
        self.progress_bar.setValue(clamped_value)
        self.status_label.setText(message)
        if clamped_value >= 100:
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

    def on_shading_complete(self, updated_files):
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.cancel_alignment)
        self.vm.align_channels(self.vm.selected_files)

    def cancel_alignment(self):
        self.vm.cancel_alignment()
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)

    def cancel_manual_align_preview_loading(self):
        self.vm.cancel_manual_align_preview_loading()
        self._manual_align_assigned_files = []
        self._manual_align_layer_labels = []
        self._restore_cancel_to_alignment()

    def cancel_bead_generation(self):
        self.vm.cancel_bead_generation()
        self.status_label.setVisible(False)
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(True)

    def update_export_progress(self, value: int, msg: str):
        self.export_progress_bar.setVisible(True)
        self.export_progress_bar.setMaximum(100)
        self.export_progress_bar.setValue(value)
        if value >= 100:
            self.export_progress_bar.setVisible(False)

    def on_export_complete(self):
        self.status_label.setText("Export complete!")
        self.status_label.setVisible(True)
        QTimer.singleShot(3000, lambda: self.status_label.setVisible(False))

    def _on_folder_loaded(self, file_items):
        pass

    def _on_file_loaded(self, file_item):
        pass

    def _setup_main_window(self):
        self.setWindowTitle("Decoding-Explorer")
        if sys.platform == "win32":
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.resize(1280, 800)
        self.setMinimumSize(1200, 800)

    @property
    def gripSize(self):
        return self._grip_size

    def setGripSize(self, size):
        if size == self._grip_size:
            return
        self._grip_size = max(2, size)
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
        self.side_grips[0].setGeometry(0, inRect.top(), self.gripSize, inRect.height())
        # top edge
        self.side_grips[1].setGeometry(inRect.left(), 0, inRect.width(), self.gripSize)
        # right edge
        self.side_grips[2].setGeometry(
            inRect.left() + inRect.width(), inRect.top(), self.gripSize, inRect.height()
        )
        # bottom edge
        self.side_grips[3].setGeometry(
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
        if files:
            selected_files = self.get_selected_files()
            selected_paths = {f.path for f in selected_files}
            updated_paths = {f.path for f in files}
            if selected_paths & updated_paths:
                self.handle_selection_change()

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def eventFilter(self, obj, event):  # type: ignore
        if sys.platform == "win32":
            if obj == self.menu_bar_ui:
                if event.type() == QEvent.Type.MouseButtonPress:
                    self.drag_pos = event.globalPosition().toPoint()
                    return False  # Allow the event to propagate for clicks
                elif event.type() == QEvent.Type.MouseMove:
                    if (
                        event.buttons() == Qt.MouseButton.LeftButton
                        and hasattr(self, "drag_pos")
                        and self.drag_pos is not None
                    ):
                        self.move(
                            self.pos()
                            + event.globalPosition().toPoint()
                            - self.drag_pos
                        )
                        self.drag_pos = event.globalPosition().toPoint()
                        return True  # Consume the event if dragging
                elif event.type() == QEvent.Type.MouseButtonRelease:
                    self.drag_pos = QPoint()  # Reset dragPos
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
        combination_counts = (
            bead_data.groupby(merge_columns).size().reset_index(name="count")
        )

        # Extract the counts for partitioning
        counts = combination_counts["count"].tolist()

        if len(counts) > 1:
            # Use find_min_std_partition to separate into two groups
            groups, min_std = find_min_std_partition(counts)

            # Determine which group has lower values (invalid) and higher values (valid)
            # Compare the minimum values of each group instead of means
            group1_min = min(groups[0]) if groups[0] else float("inf")
            group2_min = min(groups[1]) if groups[1] else float("inf")

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
            if row["count"] in invalid_counts_set:
                protein_name = "Invalid"
            else:
                # Assign unique protein names (Protein 1, Protein 2, etc.) to valid combinations
                protein_name = f"Protein {valid_protein_counter}"
                valid_protein_counter += 1

            # Create row for protein profile
            profile_row = {}
            for col in merge_columns:
                profile_row[col] = row[col]
            profile_row["Protein name"] = protein_name
            protein_profile_data.append(profile_row)

        # Create the protein profile dataframe
        protein_profile = pd.DataFrame(protein_profile_data)

    bead_data = bead_data.merge(protein_profile, how="left", on=merge_columns)

    # Fill NaN values with "Invalid"
    bead_data["Protein name"].fillna("Invalid", inplace=True)
    # For all merge columns being 255
    mask_all_255 = (bead_data[merge_columns] == 255).all(axis=1)
    bead_data.loc[mask_all_255, "Protein name"] = "Filtered"

    return bead_data
