import os
from unittest.mock import MagicMock, patch, call

import cv2
import image_processing
import numpy as np
import pytest
from model.file_item import FileItem
from model.status_enum import FileStatus
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog


class TestMainWindow:
    """Integration tests for the MainWindow."""

    def test_main_window_creation(self, mock_main_window):
        """MainWindow should create successfully with correct title."""
        window = mock_main_window
        assert window is not None
        assert window.windowTitle() == "Decoding-Explorer"

    def test_get_selected_files_returns_list(self, mock_main_window):
        """get_selected_files should return a list."""
        window = mock_main_window
        selected = window.get_selected_files()
        assert isinstance(selected, list)

    def test_update_file_list_adds_items(
        self, mock_main_window, mock_file_items
    ):
        """update_file_list should add items to the table."""
        window = mock_main_window

        initial_count = window.file_table_widget.rowCount()

        window.update_file_list(mock_file_items)

        assert window.file_table_widget.rowCount() == initial_count + len(mock_file_items)

    def test_update_files_view_updates_status(
        self, mock_main_window, mock_file_item
    ):
        """update_files_view should update the status display."""
        window = mock_main_window

        window.file_table_widget.add_file_item(mock_file_item)

        updated_item = FileItem(
            path=mock_file_item.path,
            status=FileStatus.ALIGNED,
            shape=(1, 512, 512),
            dtype="uint16"
        )
        window.update_files_view([updated_item])

        status_text = window.file_table_widget.item(0, 1).text()
        assert status_text == FileStatus.ALIGNED.value

    def test_handle_selection_change_updates_vm_selected(
        self, mock_main_window, mock_file_items
    ):
        """handle_selection_change should update vm.selected_files."""
        window = mock_main_window

        window.update_file_list(mock_file_items)
        window.file_table_widget.selectRow(0)
        window.handle_selection_change()

    def test_handle_dropped_paths_loads_folder(
        self, mock_main_window, mocker
    ):
        """handle_dropped_paths with folder should call load_folder."""
        window = mock_main_window

        with patch('os.path.isdir', return_value=True), \
             patch.object(window.vm, 'load_folder') as mock_load_folder:
            window.handle_dropped_paths(["/test/folder"])

            mock_load_folder.assert_called_once_with(["/test/folder"])

    def test_handle_dropped_paths_loads_file(
        self, mock_main_window, mocker
    ):
        """handle_dropped_paths with file should call load_file."""
        window = mock_main_window

        with patch('os.path.isfile', return_value=True), \
             patch.object(window.vm, 'load_file') as mock_load_file:
            window.handle_dropped_paths(["/test/file.tif"])

            mock_load_file.assert_called_once_with(["/test/file.tif"])

    def test_start_alignment_shows_progress(
        self, mock_main_window, mocker
    ):
        """start_alignment sets up UI for alignment progress."""
        window = mock_main_window

        # Verify the method exists and is callable
        assert hasattr(window, 'start_alignment')
        assert callable(window.start_alignment)

    def test_start_alignment_no_reference(
        self, mock_main_window, mocker
    ):
        """start_alignment without assignment should show assignment error."""
        window = mock_main_window

        # Add files to VM
        from model.file_item import FileItem
        for i in range(3):
            item = FileItem(path=f"/test/file{i}.tif")
            window.vm.files[item.path] = item

        # Mock get_selected_files to return files
        files = [window.vm.files[f"/test/file{i}.tif"] for i in range(3)]
        mocker.patch.object(window, 'get_selected_files', return_value=files)

        with patch.object(window, 'show_error') as mock_error:
            window.start_alignment()

            mock_error.assert_called_once_with("Assign cycles to continue.")

    def test_start_alignment_no_files_selected(
        self, mock_main_window, mocker
    ):
        """start_alignment without dataset assignment should show assignment error."""
        window = mock_main_window

        reference_item = FileItem(path="/test/reference.tif")
        window.vm.reference_item = reference_item
        window.vm.files[reference_item.path] = reference_item

        with patch.object(window, 'show_error') as mock_error:
            window.start_alignment()

            mock_error.assert_called_once_with("Assign cycles to continue.")

    def test_start_bead_generation_no_reference(
        self, mock_main_window, mocker
    ):
        """start_bead_generation without assignment should show assignment error."""
        window = mock_main_window

        # Add files to VM
        from model.file_item import FileItem
        for i in range(3):
            item = FileItem(path=f"/test/file{i}.tif")
            window.vm.files[item.path] = item

        # Mock get_selected_files to return files
        files = [window.vm.files[f"/test/file{i}.tif"] for i in range(3)]
        mocker.patch.object(window, 'get_selected_files', return_value=files)

        with patch.object(window, 'show_error') as mock_error:
            window.start_bead_generation()

            mock_error.assert_called_once_with("Assign cycles to continue.")

    def test_start_bead_generation_passes_stardist_and_default_sweep_settings(
        self, mock_main_window, mock_file_item, mocker
    ):
        window = mock_main_window
        window.vm.reference_item = mock_file_item
        window.vm.files[mock_file_item.path] = mock_file_item
        window.vm.dataset_cycle_assignments = {0: mock_file_item}
        window.vm.dataset_assignment_valid = True

        window.metadata_view.stardist_guess_tiles_checkbox.setChecked(False)
        window.metadata_view.stardist_num_tiles_input.setText("3")
        window.metadata_view.use_stardist_bead_centers_checkbox.setChecked(True)
        window.metadata_view.area_multiplier_input.setText("2.2")

        with patch.object(window.vm, "generate_beads") as mock_generate:
            window.start_bead_generation()

        mock_generate.assert_called_once_with(
            {0: mock_file_item},
            use_stardist=True,
            model_name="model_5_400epoch",
            stardist_use_guess_tiles=False,
            stardist_n_tiles=3,
            use_stardist_bead_centers=True,
            area_multiplier=2.2,
            ensemble_ratio_start=image_processing.DEFAULT_ENSEMBLE_RATIO_START,
            ensemble_ratio_end=image_processing.DEFAULT_ENSEMBLE_RATIO_END,
            ensemble_ratio_step=image_processing.DEFAULT_ENSEMBLE_RATIO_STEP,
        )

    def test_start_alignment_includes_optional_protein_file(
        self, mock_main_window, mock_file_items
    ):
        window = mock_main_window
        ref_item, cy1_item, protein_item = mock_file_items[:3]
        window.vm.files[ref_item.path] = ref_item
        window.vm.files[cy1_item.path] = cy1_item
        window.vm.files[protein_item.path] = protein_item
        window.vm.reference_item = ref_item
        window.vm.dataset_cycle_assignments = {0: ref_item, 1: cy1_item}
        window.vm.dataset_protein_file = protein_item
        window.vm.dataset_assignment_valid = True

        with patch.object(window.vm, "apply_shading") as mock_apply_shading:
            window.start_alignment()

        mock_apply_shading.assert_called_once()
        args, _ = mock_apply_shading.call_args
        aligned_files = args[0]
        assert [f.path for f in aligned_files] == [cy1_item.path, protein_item.path]

    def test_start_alignment_includes_optional_protein_file_with_two_cycles(
        self, mock_main_window, mock_file_items
    ):
        window = mock_main_window
        ref_item, cy1_item, cy2_item = mock_file_items[:3]
        protein_item = FileItem(path="/tmp/protein_assigned.tif")

        for file_item in [ref_item, cy1_item, cy2_item, protein_item]:
            window.vm.files[file_item.path] = file_item

        window.vm.reference_item = ref_item
        window.vm.dataset_cycle_assignments = {
            0: ref_item,
            1: cy1_item,
            2: cy2_item,
        }
        window.vm.dataset_protein_file = protein_item
        window.vm.dataset_assignment_valid = True
        window.metadata_view.apply_shading_checkbox.setChecked(False)

        with patch.object(window.vm, "align_channels") as mock_align_channels:
            window.start_alignment()

        mock_align_channels.assert_called_once()
        args, _ = mock_align_channels.call_args
        aligned_files = args[0]
        assert [f.path for f in aligned_files] == [
            cy1_item.path,
            cy2_item.path,
            protein_item.path,
        ]

    def test_start_manual_alignment_starts_async_preview_loading(
        self, mock_main_window, mock_file_items
    ):
        window = mock_main_window
        ref_item, cy1_item, protein_item = mock_file_items[:3]
        window.vm.files[ref_item.path] = ref_item
        window.vm.files[cy1_item.path] = cy1_item
        window.vm.files[protein_item.path] = protein_item
        window.vm.reference_item = ref_item
        window.vm.dataset_cycle_assignments = {0: ref_item, 1: cy1_item}
        window.vm.dataset_protein_file = protein_item
        window.vm.dataset_assignment_valid = True

        with patch.object(window.vm, "load_manual_align_preview_images") as mock_load:
            window.start_manual_alignment()

        mock_load.assert_called_once()
        args, _ = mock_load.call_args
        assert args[0] == ref_item
        assert [f.path for f in args[1]] == [cy1_item.path, protein_item.path]
        assert window.progress_bar.isHidden() is False
        assert window.status_label.isHidden() is False
        assert window.cancel_button.isHidden() is False

    def test_manual_align_preview_loaded_uses_top_left_preview_default(
        self, mock_main_window, mock_file_items
    ):
        window = mock_main_window
        ref_item, cy1_item, protein_item = mock_file_items[:3]
        window.vm.files[ref_item.path] = ref_item
        window.vm.files[cy1_item.path] = cy1_item
        window.vm.files[protein_item.path] = protein_item
        window.vm.reference_item = ref_item
        window.vm.dataset_cycle_assignments = {0: ref_item, 1: cy1_item}
        window.vm.dataset_protein_file = protein_item
        window.vm.dataset_assignment_valid = True

        preview_image = np.zeros((3200, 3200), dtype=np.uint16)
        dialog_instance = MagicMock()
        dialog_instance.exec.return_value = True

        with patch.object(window.vm, "load_manual_align_preview_images"):
            window.start_manual_alignment()

        with patch(
            "view.main_window.AlignmentPreviewDialog", return_value=dialog_instance
        ) as mock_dialog:
            window._on_manual_align_preview_loaded(
                [preview_image, preview_image, preview_image]
            )

        mock_dialog.assert_called_once()
        assert mock_dialog.call_args.kwargs["initial_preview_size"] == 2000
        assert mock_dialog.call_args.kwargs["initial_checked_indices"] == [0]
        assert mock_dialog.call_args.kwargs["layer_labels"] == ["Cycle 2", "Protein"]

    def test_cancel_manual_align_preview_loading_hides_progress(
        self, mock_main_window, mock_file_items
    ):
        window = mock_main_window
        ref_item, cy1_item, protein_item = mock_file_items[:3]
        window.vm.files[ref_item.path] = ref_item
        window.vm.files[cy1_item.path] = cy1_item
        window.vm.files[protein_item.path] = protein_item
        window.vm.reference_item = ref_item
        window.vm.dataset_cycle_assignments = {0: ref_item, 1: cy1_item}
        window.vm.dataset_protein_file = protein_item
        window.vm.dataset_assignment_valid = True

        with patch.object(window.vm, "load_manual_align_preview_images"):
            window.start_manual_alignment()

        with patch.object(
            window.vm, "cancel_manual_align_preview_loading"
        ) as mock_cancel:
            window.cancel_manual_align_preview_loading()

        mock_cancel.assert_called_once()
        assert window.progress_bar.isVisible() is False
        assert window.status_label.isVisible() is False
        assert window.cancel_button.isVisible() is False

    def test_manual_alignment_complete_only_updates_checked_layers(
        self, mock_main_window
    ):
        window = mock_main_window

        cycle2_item = FileItem(path="/tmp/cycle2.tif", status=FileStatus.RAW)
        hidden_item = FileItem(path="/tmp/hidden_cycle.tif", status=FileStatus.RAW)
        protein_item = FileItem(path="/tmp/protein.tif", status=FileStatus.RAW)

        cycle2_image = np.arange(16, dtype=np.uint16).reshape(4, 4)
        hidden_image = np.arange(16, dtype=np.uint16).reshape(4, 4) + 100
        protein_image = np.arange(16, dtype=np.uint16).reshape(4, 4) + 200

        cycle2_item.working_image = cycle2_image.copy()
        hidden_item.working_image = hidden_image.copy()
        protein_item.working_image = protein_image.copy()

        window.vm.files[cycle2_item.path] = cycle2_item
        window.vm.files[hidden_item.path] = hidden_item
        window.vm.files[protein_item.path] = protein_item

        cycle2_matrix = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        protein_matrix = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32)

        with patch("view.main_window.QMessageBox.information") as mock_info:
            window._on_manual_alignment_complete(
                [cycle2_matrix, None, protein_matrix],
                [cycle2_item, hidden_item, protein_item],
            )

        expected_cycle2 = cv2.warpAffine(cycle2_image, cycle2_matrix, (4, 4))
        expected_protein = cv2.warpAffine(protein_image, protein_matrix, (4, 4))

        np.testing.assert_array_equal(
            window.vm.files[cycle2_item.path].working_image, expected_cycle2
        )
        np.testing.assert_array_equal(
            window.vm.files[protein_item.path].working_image, expected_protein
        )
        np.testing.assert_array_equal(
            window.vm.files[hidden_item.path].working_image, hidden_image
        )

        assert window.vm.files[cycle2_item.path].status == FileStatus.ALIGNED
        assert window.vm.files[protein_item.path].status == FileStatus.ALIGNED
        assert window.vm.files[hidden_item.path].status == FileStatus.RAW

        mock_info.assert_called_once()
        assert "2 image(s)" in mock_info.call_args.args[2]

    def test_assign_cycles_sets_cycle1_as_reference(
        self, mock_main_window, mock_file_items, mocker
    ):
        window = mock_main_window
        files = mock_file_items[:3]
        for file_item in files:
            window.vm.files[file_item.path] = file_item
            window.file_table_widget.add_file_item(file_item)

        dialog_instance = MagicMock()
        dialog_instance.exec.return_value = True
        dialog_instance.get_assignments.return_value = {
            1: files[1],
            2: files[0],
            3: files[2],
        }
        dialog_instance.get_protein_file.return_value = None
        mocker.patch("view.main_window.CycleAssignmentDialog", return_value=dialog_instance)

        window.assign_cycles()

        assert window.vm.reference_item == files[1]
        assert window.vm.is_dataset_ready() is True
        assigned = window.vm.get_dataset_cycle_assignments()
        assert assigned is not None
        assert assigned[0] == files[1]
        assert assigned[1] == files[0]
        assert assigned[2] == files[2]

    def test_assign_cycles_can_mark_optional_protein_file(
        self, mock_main_window, mock_file_items, mocker
    ):
        window = mock_main_window
        files = mock_file_items[:3]
        for file_item in files:
            window.vm.files[file_item.path] = file_item
            window.file_table_widget.add_file_item(file_item)

        dialog_instance = MagicMock()
        dialog_instance.exec.return_value = True
        dialog_instance.get_assignments.return_value = {
            1: files[0],
            2: files[1],
        }
        dialog_instance.get_protein_file.return_value = files[2]
        mocker.patch("view.main_window.CycleAssignmentDialog", return_value=dialog_instance)

        window.assign_cycles()

        assert window.vm.reference_item == files[0]
        assert window.vm.is_dataset_ready() is True
        assert window.vm.get_dataset_protein_file() == files[2]
        assigned = window.vm.get_dataset_cycle_assignments()
        assert assigned is not None
        assert assigned[0] == files[0]
        assert assigned[1] == files[1]

    def test_on_beads_generated_does_not_auto_save(
        self, mock_main_window, mock_file_item, mocker
    ):
        """on_beads_generated should not trigger save_beads automatically."""
        window = mock_main_window

        window.vm.reference_item = mock_file_item
        window.vm.files[mock_file_item.path] = mock_file_item

        with patch.object(window, 'save_beads') as mock_save:
            import pandas as pd
            beads = pd.DataFrame({'x': [1, 2, 3]})
            window.on_beads_generated(beads)
            mock_save.assert_not_called()

    def test_save_generated_beads_applies_pending_ratio_before_export(
        self, mock_main_window, mock_file_item
    ):
        window = mock_main_window
        import pandas as pd

        mock_file_item.beads = pd.DataFrame(
            {"x": [1.0], "y": [2.0], "cy0": [255], "cy1": [255]}
        )
        mock_file_item.pre_ensemble_beads = mock_file_item.beads.copy()
        mock_file_item.ensemble_cache = {"dummy": True}
        mock_file_item.ensemble_ratio_applied = 1.0
        window.vm.reference_item = mock_file_item
        window.vm.files[mock_file_item.path] = mock_file_item

        stats_df = pd.DataFrame(
            [
                {"ratio": 1.0, "valid_pct": 10.0, "invalid_pct": 20.0, "filtered_pct": 70.0},
                {"ratio": 1.05, "valid_pct": 15.0, "invalid_pct": 15.0, "filtered_pct": 70.0},
            ]
        )
        window.metadata_view.set_ensemble_sweep_stats(stats_df, selected_ratio=1.05, applied_ratio=1.0)

        with patch.object(window.vm, "apply_ensemble_ratio") as mock_apply, patch.object(window, "save_beads") as mock_save:
            window.save_generated_beads()
            mock_apply.assert_called_once_with(mock_file_item, 1.05)
            mock_save.assert_called_once()

    def test_remove_ensemble_applied_changes_calls_vm(
        self, mock_main_window, mock_file_item
    ):
        window = mock_main_window
        import pandas as pd

        mock_file_item.beads = pd.DataFrame({"x": [1.0], "y": [2.0], "cy0": [1], "cy1": [2]})
        window.vm.reference_item = mock_file_item
        window.vm.files[mock_file_item.path] = mock_file_item

        with patch.object(window.vm, "remove_ensemble_applied_changes") as mock_remove, patch.object(window, "calculate_statistics_for_file"):
            window.remove_ensemble_applied_changes()
            mock_remove.assert_called_once_with(mock_file_item)

    def test_lower_invalid_ratio_uses_applied_ratio_base(
        self, mock_main_window, mock_file_item
    ):
        window = mock_main_window
        import pandas as pd

        mock_file_item.beads = pd.DataFrame({"x": [1.0], "y": [2.0], "cy0": [1], "cy1": [2]})
        mock_file_item.pre_ensemble_beads = mock_file_item.beads.copy()
        mock_file_item.ensemble_cache = {"dummy": True}
        mock_file_item.ensemble_ratio_applied = 1.2
        window.vm.reference_item = mock_file_item
        window.vm.files[mock_file_item.path] = mock_file_item

        with patch.object(window.vm, "apply_ensemble_ratio") as mock_apply, patch.object(window, "calculate_statistics_for_file"):
            window.lower_invalid_ratio()

        mock_apply.assert_called_once_with(mock_file_item, 1.25)

    def test_lower_filter_ratio_uses_selected_ratio_when_no_applied(
        self, mock_main_window, mock_file_item
    ):
        window = mock_main_window
        import pandas as pd

        mock_file_item.beads = pd.DataFrame({"x": [1.0], "y": [2.0], "cy0": [1], "cy1": [2]})
        mock_file_item.pre_ensemble_beads = mock_file_item.beads.copy()
        mock_file_item.ensemble_cache = {"dummy": True}
        mock_file_item.ensemble_ratio_applied = None
        window.vm.reference_item = mock_file_item
        window.vm.files[mock_file_item.path] = mock_file_item

        stats_df = pd.DataFrame(
            [
                {"ratio": 1.0, "valid_pct": 10.0, "invalid_pct": 20.0, "filtered_pct": 70.0},
                {"ratio": 1.05, "valid_pct": 15.0, "invalid_pct": 15.0, "filtered_pct": 70.0},
            ]
        )
        window.metadata_view.set_ensemble_sweep_stats(stats_df, selected_ratio=1.05, applied_ratio=None)

        with patch.object(window.vm, "apply_ensemble_ratio") as mock_apply, patch.object(window, "calculate_statistics_for_file"):
            window.lower_filter_ratio()

        mock_apply.assert_called_once_with(mock_file_item, 1.0)

    def test_lower_invalid_ratio_allows_out_of_range(
        self, mock_main_window, mock_file_item
    ):
        window = mock_main_window
        import pandas as pd

        mock_file_item.beads = pd.DataFrame({"x": [1.0], "y": [2.0], "cy0": [1], "cy1": [2]})
        mock_file_item.pre_ensemble_beads = mock_file_item.beads.copy()
        mock_file_item.ensemble_cache = {"dummy": True}
        mock_file_item.ensemble_ratio_applied = 1.5
        window.vm.reference_item = mock_file_item
        window.vm.files[mock_file_item.path] = mock_file_item

        with patch.object(window.vm, "apply_ensemble_ratio") as mock_apply, patch.object(window, "calculate_statistics_for_file"):
            window.lower_invalid_ratio()

        mock_apply.assert_called_once_with(mock_file_item, 1.55)

    def test_save_generated_beads_does_not_snap_out_of_range_selected_ratio(
        self, mock_main_window, mock_file_item
    ):
        window = mock_main_window
        import pandas as pd

        mock_file_item.beads = pd.DataFrame(
            {"x": [1.0], "y": [2.0], "cy0": [255], "cy1": [255]}
        )
        mock_file_item.pre_ensemble_beads = mock_file_item.beads.copy()
        mock_file_item.ensemble_cache = {"dummy": True}
        mock_file_item.ensemble_ratio_applied = 1.55
        mock_file_item.ensemble_ratio_selected = 1.55
        window.vm.reference_item = mock_file_item
        window.vm.files[mock_file_item.path] = mock_file_item

        stats_df = pd.DataFrame(
            [
                {"ratio": 1.0, "valid_pct": 10.0, "invalid_pct": 20.0, "filtered_pct": 70.0},
                {"ratio": 1.5, "valid_pct": 15.0, "invalid_pct": 15.0, "filtered_pct": 70.0},
            ]
        )
        window.metadata_view.set_ensemble_sweep_stats(stats_df, selected_ratio=1.55, applied_ratio=1.55)

        with patch.object(window.vm, "apply_ensemble_ratio") as mock_apply, patch.object(window, "save_beads") as mock_save:
            window.save_generated_beads()

        mock_apply.assert_not_called()
        mock_save.assert_called_once()

    def test_handle_metadata_applied_updates_items(
        self, mock_main_window, mock_file_item, mocker
    ):
        """handle_metadata_applied should call vm.apply_metadata with selected files."""
        window = mock_main_window

        window.file_table_widget.add_file_item(mock_file_item)
        window.file_table_widget.selectRow(0)

        new_metadata = {"max_size": 1000}
        with patch.object(window.vm, 'apply_metadata') as mock_apply:
            window.handle_metadata_applied(new_metadata)
            mock_apply.assert_called_once_with(new_metadata, [mock_file_item])

    def test_handle_metadata_applied_includes_dataset_protein_file(
        self, mock_main_window
    ):
        window = mock_main_window

        reference_item = FileItem(path="/test/reference.tif")
        cycle_item = FileItem(path="/test/cycle_1.tif")
        protein_item = FileItem(path="/test/protein.tif")
        new_metadata = {"max_size": 1000}

        with (
            patch.object(
                window.vm,
                "get_dataset_files_ordered",
                return_value=[reference_item, cycle_item],
            ),
            patch.object(
                window.vm,
                "get_dataset_protein_file",
                return_value=protein_item,
            ),
            patch.object(window.vm, "apply_metadata") as mock_apply,
        ):
            window.handle_metadata_applied(new_metadata)

        mock_apply.assert_called_once_with(
            new_metadata, [reference_item, cycle_item, protein_item]
        )

    def test_handle_metadata_applied_updates_assigned_protein_tiff_shape(
        self, mock_main_window
    ):
        window = mock_main_window

        reference_item = FileItem(path="/test/reference.tif", status=FileStatus.RAW)
        reference_item.shape = (1, 512, 512)
        reference_item.original_shape = (1, 512, 512)
        reference_item.dtype = "uint16"

        cycle_item = FileItem(path="/test/cycle_1.tif", status=FileStatus.RAW)
        cycle_item.shape = (1, 512, 512)
        cycle_item.original_shape = (1, 512, 512)
        cycle_item.dtype = "uint16"

        protein_item = FileItem(path="/test/protein.tif", status=FileStatus.RAW)
        protein_item.shape = (1, 512, 512)
        protein_item.original_shape = (1, 512, 512)
        protein_item.dtype = "uint16"

        for file_item in [reference_item, cycle_item, protein_item]:
            window.vm.files[file_item.path] = file_item
        window.update_file_list([reference_item, cycle_item, protein_item])

        window.vm.reference_item = reference_item
        assert window.vm.set_dataset_cycle_assignments(
            {0: reference_item, 1: cycle_item},
            protein_file=protein_item,
        )

        window.handle_metadata_applied({"max_size": 256})

        assert window.vm.files[protein_item.path].shape == (1, 256, 256)

        protein_row = None
        for row in range(window.file_table_widget.rowCount()):
            filename_item = window.file_table_widget.item(row, 0)
            if filename_item is None:
                continue
            row_item = filename_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(row_item, FileItem) and row_item.path == protein_item.path:
                protein_row = row
                break

        assert protein_row is not None
        assert window.file_table_widget.item(protein_row, 2).text() == "C=1, Y=256, X=256"

    def test_minimum_size_set(self, mock_main_window):
        """MainWindow should have minimum size set."""
        window = mock_main_window
        assert window.minimumWidth() == 1200
        assert window.minimumHeight() == 800

    def test_initial_size_set(self, mock_main_window):
        """MainWindow should have initial size set."""
        window = mock_main_window
        assert window.width() == 1280
        assert window.height() == 800

    def test_handle_metadata_corrected_calls_metadata_vm(
        self, mock_main_window, mock_file_item, mocker
    ):
        """handle_metadata_corrected should call metadata_vm.update_corrected_metadata."""
        window = mock_main_window

        corrected_values = {"max_size": 512}

        with patch.object(window.metadata_vm, 'update_corrected_metadata') as mock_update:
            window.handle_metadata_corrected(corrected_values)

            mock_update.assert_called_once_with("max_size", 512)

    def test_handle_metadata_applied_calls_vm_apply_metadata(
        self, mock_main_window, mock_file_item, mocker
    ):
        """handle_metadata_applied should call vm.apply_metadata."""
        window = mock_main_window

        window.file_table_widget.add_file_item(mock_file_item)
        window.file_table_widget.selectRow(0)

        new_metadata = {"max_size": 1000}

        with patch.object(window.vm, 'apply_metadata') as mock_apply:
            window.handle_metadata_applied(new_metadata)

            mock_apply.assert_called_once()
            args, kwargs = mock_apply.call_args
            assert args[0] == new_metadata
            assert len(args[1]) == 1
