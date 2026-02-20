# views/main_window.py
import os
import sys
import warnings
from typing import List

import cv2
import numpy as np
from pandas import DataFrame
from PyQt6.QtCore import QEvent, QPoint, QRect, Qt, QTimer
from PyQt6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QHeaderView,
                             QLabel, QMainWindow, QMenuBar, QMessageBox,
                             QProgressBar, QPushButton, QSizeGrip, QSizePolicy,
                             QSplitter, QVBoxLayout, QWidget)

from model.file_item import FileItem
from model.status_enum import FileStatus
from utils import find_min_std_partition, is_dark_mode
from view.alignment_preview_dialog import AlignmentPreviewDialog
from view.CropDialog import CropDialog
from view.CycleAssignmentWidget import CycleAssignmentWidget
from view.FileListWidget import FileTableWidget
from view.MetadataView import MetadataView
from view.roi_inspector import ROI_Inspector
from viewmodel.file_manager_vm import FileManagerVM, load_image
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
        splitter.setSizes([2000, 200])  # Give metadata view minimum initial width
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

        self.file_table_widget.itemSelectionChanged.connect(
            self.handle_selection_change
        )
        self.file_table_widget.table_emptied.connect(self.handle_table_emptied)
        self.metadata_vm.metadata_applied_sig.connect(self.handle_metadata_applied)
        self.vm.metadata_corrected_sig.connect(self.handle_metadata_corrected)
        self.metadata_vm.align_channels_sig.connect(self.start_alignment)
        self.metadata_view.export_all_sig.connect(self.vm.export_files)
        self.metadata_view.generate_beads_sig.connect(self.start_bead_generation)
        self.metadata_view.upload_beads_sig.connect(self.upload_bead_csv)
        self.metadata_view.recompute_ensemble_sig.connect(self.recompute_ensemble_sweep)
        self.metadata_view.apply_ensemble_sig.connect(self.apply_ensemble_ratio)
        self.metadata_view.remove_ensemble_sig.connect(self.remove_ensemble_applied_changes)
        self.metadata_view.lower_invalid_sig.connect(self.lower_invalid_ratio)
        self.metadata_view.lower_filter_sig.connect(self.lower_filter_ratio)
        self.metadata_view.save_beads_sig.connect(self.save_generated_beads)
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

        self.menuBarUI = MenuBarUI(self)
        self.setMenuBar(self.menuBarUI)
        if sys.platform == "win32":
            self.menuBarUI.installEventFilter(self)

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
            beads = beads[(beads['x'] >= x1) & (beads['x'] < x2) & (beads['y'] >= y1) & (beads['y'] < y2)]

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
        selected_files = self.get_selected_files()
        for file_item in selected_files:
            if file_item.beads is not None:
                self.calculate_statistics_for_file(file_item, protein_profile)
                break  # Process only the first selected file with beads

    def show_roi_inspector_window(
        self, bright_fields, beads_df, cycles, bboxs, labeled_image, protein_profile, max_size
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
        self.roi_inspector = ROI_Inspector(data)
        self.roi_inspector.show()

    def save_beads(self, beads):
        # Apply bead crop bounds if set on reference item
        if self.vm.reference_item and self.vm.reference_item.bead_crop_bounds is not None:
            x1, y1, x2, y2 = self.vm.reference_item.bead_crop_bounds
            beads = beads[(beads['x'] >= x1) & (beads['x'] < x2) & (beads['y'] >= y1) & (beads['y'] < y2)]

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
        start, _, _ = self.metadata_view.get_ensemble_sweep_inputs()
        return float(start)

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

    def save_generated_beads(self):
        reference_item = self.vm.reference_item
        if reference_item is None:
            self.show_error("Please set a reference image first.")
            return
        if reference_item.beads is None or reference_item.beads.empty:
            self.show_error("No beads available. Generate or upload beads first.")
            return
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
                return
        self.save_beads(reference_item.beads)

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
        selected_files = self.get_selected_files()
        self.vm.apply_metadata(new_metadata, selected_files)

    def handle_metadata_corrected(self, corrected_values: dict):
        for key, value in corrected_values.items():
            self.metadata_vm.update_corrected_metadata(key, value)

    def get_selected_files(self) -> List[FileItem]:
        return self.file_table_widget.get_selected_files()

    def handle_selection_change(self):
        selected_files = self.get_selected_files()
        self.metadata_vm.update_selected_items(selected_files)
        self.vm.selected_files = selected_files
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
        selected_files = self.get_selected_files()
        if not selected_files:
            self.show_error("No files selected for alignment.")
            return

        reference_item = self.vm.reference_item
        if not reference_item:
            self.show_error("Please set a reference image first.")
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

        # Get StarDist settings from the metadata view
        stardist_settings = self.metadata_view.get_stardist_settings()
        use_stardist = stardist_settings["use_stardist"]
        model_name = stardist_settings["model_name"]
        try:
            ensemble_ratio_start, ensemble_ratio_end, ensemble_ratio_step = (
                self.metadata_view.get_ensemble_sweep_inputs()
            )
        except ValueError:
            self.show_error("Invalid ensemble ratio sweep values.")
            return

        if not files_for_assignment:
            # If only the reference file is selected, we can proceed with one cycle.
            self.vm.generate_beads(
                {0: reference_item},
                use_stardist=use_stardist,
                model_name=model_name,
                ensemble_ratio_start=ensemble_ratio_start,
                ensemble_ratio_end=ensemble_ratio_end,
                ensemble_ratio_step=ensemble_ratio_step,
            )
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
            self.vm.generate_beads(
                assignments_from_dialog,
                use_stardist=use_stardist,
                model_name=model_name,
                ensemble_ratio_start=ensemble_ratio_start,
                ensemble_ratio_end=ensemble_ratio_end,
                ensemble_ratio_step=ensemble_ratio_step,
            )

    def upload_bead_csv(self):
        selected_files = self.get_selected_files()
        if not selected_files:
            QMessageBox.warning(
                self,
                "No Files Selected",
                "Please select files to assign to cycles."
            )
            return

        reference_item = self.vm.reference_item
        if not reference_item:
            self.show_error("Please set a reference image first.")
            return

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

        files_for_assignment = [
            f for f in selected_files if f.path != reference_item.path
        ]

        dialog = CycleAssignmentWidget(files_for_assignment, self, zero_indexed=False, start_cycle=1)
        if dialog.exec():
            assignments_from_dialog = dialog.get_assignments()
            if assignments_from_dialog is None:
                self.show_error("Each file must be assigned to exactly one cycle.")
                return

            cycle_assignments = {0: reference_item}
            for cycle_num, file_item in assignments_from_dialog.items():
                cycle_assignments[cycle_num] = file_item

            self.vm.store_uploaded_beads(csv_path, reference_item, cycle_assignments)

    def on_bead_upload_complete(self, reference_item):
        print(f"on_bead_upload_complete called, beads: {reference_item.beads is not None}, empty: {reference_item.beads.empty if reference_item.beads is not None else 'N/A'}")
        self._refresh_ensemble_controls(reference_item)
        if reference_item.beads is not None and not reference_item.beads.empty:
            self.calculate_statistics_for_file(
                reference_item, self.metadata_vm.protein_df
            )
            self.metadata_view.collapse_processing_sections()

    def start_manual_alignment(self):
        """Open manual alignment dialog for selected files."""
        selected_files = self.get_selected_files()
        if not selected_files:
            self.show_error("No files selected for manual alignment.")
            return

        reference_item = self.vm.reference_item
        if not reference_item:
            self.show_error("Please set a reference image first.")
            return

        # Get brightfield image from reference
        target_image = self.vm._get_brightfield_image(reference_item)
        if target_image is None:
            self.show_error("Could not load reference image for preview.")
            return

        # Separate reference from other files
        files_for_assignment = [
            f for f in selected_files if f.path != reference_item.path
        ]

        if not files_for_assignment:
            self.show_error("Please select at least one file besides the reference to align.")
            return

        # Use CycleAssignmentWidget to assign files
        dialog = CycleAssignmentWidget(files_for_assignment, self)
        if not dialog.exec():
            return

        assignments_from_dialog = dialog.get_assignments()
        if assignments_from_dialog is None:
            self.show_error("Each file must be assigned to exactly one cycle.")
            return

        # Get moving images from assigned files
        moving_images = []
        assigned_files = []
        for cycle_num in sorted(assignments_from_dialog.keys()):
            file_item = assignments_from_dialog[cycle_num]
            moving_img = self.vm._get_brightfield_image(file_item)
            if moving_img is None:
                self.show_error(f"Could not load image for {os.path.basename(file_item.path)}")
                return
            moving_images.append(moving_img)
            assigned_files.append(file_item)

        # Open alignment preview dialog in edit mode
        preview_dialog = AlignmentPreviewDialog(
            target_image,
            moving_images,
            can_edit=True,
            can_emit=True
        )

        # Store assigned files in dialog for later use
        preview_dialog.assigned_files = assigned_files

        # Connect signals to handle transformed results
        preview_dialog.transformation_matrices.connect(
            lambda matrices: self._on_manual_alignment_complete(
                matrices, assigned_files
            )
        )

        preview_dialog.exec()

    def _on_manual_alignment_complete(
        self,
        transformation_matrices: list[np.ndarray],
        assigned_files: list[FileItem]
    ):
        """Handle completion of manual alignment."""
        if len(transformation_matrices) != len(assigned_files):
            self.show_error("Mismatch between transformation matrices and assigned files.")
            return

        to_be_updated = []
        for i, file_item in enumerate(assigned_files):
            my_f = self.vm.files.get(file_item.path)
            if not my_f:
                continue

            # Get the full multi-channel image
            full_image = my_f.working_image if my_f.working_image is not None else load_image(my_f)
            transf_matrix = transformation_matrices[i]

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
                f"Successfully aligned {len(to_be_updated)} image(s)."
            )

    def start_crop(self):
        """Handle crop operation for single or multiple files."""
        selected_files = self.get_selected_files()
        if not selected_files:
            self.show_error("No files selected for cropping.")
            return

        # Single file mode
        if len(selected_files) == 1:
            file_item = selected_files[0]
            image = self.vm._get_brightfield_image(file_item)
            if image is None:
                self.show_error("Could not load image.")
                return

            dialog = CropDialog([image], self)
            dialog.crop_confirmed.connect(
                lambda x1, y1, x2, y2: self._apply_crop_single(file_item, x1, y1, x2, y2)
            )
            dialog.exec()

        # Multi-file mode
        else:
            reference_item = self.vm.reference_item
            if not reference_item:
                self.show_error("Please set a reference image first.")
                return

            # Check reference is in selection
            if reference_item.path not in [f.path for f in selected_files]:
                self.show_error("Reference file must be included in selection for multi-file crop.")
                return

            # Filter reference from assignment
            files_for_assignment = [
                f for f in selected_files if f.path != reference_item.path
            ]

            if not files_for_assignment:
                self.show_error("Please select at least one file besides the reference.")
                return

            # Use CycleAssignmentWidget
            dialog = CycleAssignmentWidget(files_for_assignment, self)
            if not dialog.exec():
                return

            assignments = dialog.get_assignments()
            if assignments is None:
                self.show_error("Each file must be assigned to exactly one cycle.")
                return

            # Load images for overlay
            target_image = self.vm._get_brightfield_image(reference_item)
            moving_images = []
            assigned_files = [reference_item]  # Reference first

            for cycle_num in sorted(assignments.keys()):
                file_item = assignments[cycle_num]
                img = self.vm._get_brightfield_image(file_item)
                if img is None:
                    self.show_error(f"Could not load {os.path.basename(file_item.path)}")
                    return
                moving_images.append(img)
                assigned_files.append(file_item)

            # Create overlay crop dialog
            all_images = [target_image] + moving_images
            crop_dialog = CropDialog(all_images, self)
            crop_dialog.crop_confirmed.connect(
                lambda x1, y1, x2, y2: self._apply_crop_multiple(assigned_files, x1, y1, x2, y2)
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
                img = self.vm._get_brightfield_image(reference_item.cycle_files[cycle_num])
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
            lambda x1, y1, x2, y2: self._apply_bead_crop(reference_item, x1, y1, x2, y2, total_before)
        )
        dialog.exec()

    def _apply_bead_crop(self, reference_item: FileItem, x1: int, y1: int, x2: int, y2: int, total_before: int):
        """Store bead crop bounds and recalculate statistics."""
        reference_item.bead_crop_bounds = (x1, y1, x2, y2)
        beads = reference_item.beads
        filtered = beads[(beads['x'] >= x1) & (beads['x'] < x2) & (beads['y'] >= y1) & (beads['y'] < y2)]
        total_after = len(filtered)
        self.calculate_statistics_for_file(reference_item, self.metadata_vm.protein_df)
        QMessageBox.information(
            self,
            "Bead Crop Applied",
            f"Beads: {total_before} → {total_after} (cropped {total_before - total_after})"
        )

    def _apply_crop_single(self, file_item: FileItem, x1: int, y1: int, x2: int, y2: int):
        """Apply crop to single file."""
        self.vm.apply_crop([file_item], x1, y1, x2, y2)
        QMessageBox.information(self, "Crop Complete", "Image cropped successfully.")

    def _apply_crop_multiple(self, file_items: list[FileItem], x1: int, y1: int, x2: int, y2: int):
        """Apply crop to multiple files."""
        self.vm.apply_crop(file_items, x1, y1, x2, y2)
        QMessageBox.information(
            self,
            "Crop Complete",
            f"Successfully cropped {len(file_items)} image(s)."
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
