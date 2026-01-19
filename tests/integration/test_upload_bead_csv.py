"""
Integration test for the Upload Bead CSV feature.

Feature: Add functionality to upload a pre-generated bead CSV file to populate
the statistics table and enable ROI viewer inspection. The workflow should:
1. Validate the CSV format (x, y, cy0, cy1, etc. columns)
2. Allow user to assign cycle files to each cycle in the CSV
3. Store the bead data in a FileItem
4. Extract cycle images for the ROI viewer
5. Enable statistics table population

This test verifies the core behavior:
- FileManagerVM.validate_bead_csv() validates CSV structure
- FileManagerVM.store_uploaded_beads() stores beads and updates FileItem status
- FileStatus changes to BEADS_GENERATED
- file_information_update signal is emitted
"""

import pandas as pd
import pytest

from model.file_item import FileItem
from model.status_enum import FileStatus


class TestUploadBeadCSV:
    """Integration test for uploading pre-generated bead CSV files."""

    def test_validate_bead_csv_with_valid_format(
        self, mock_file_manager_vm, tmp_path
    ):
        """Validate that a correctly formatted CSV passes validation.

        This test verifies:
        1. FileManagerVM has a validate_bead_csv() method
        2. Method returns True for valid CSV with required columns
        3. Valid format includes: x, y, cy0, cy1, ... columns
        """
        vm = mock_file_manager_vm

        # Create a valid bead CSV file
        csv_path = tmp_path / "beads.csv"
        beads_data = {
            'x': [10.5, 20.3, 30.7, 40.2],
            'y': [15.2, 25.8, 35.1, 45.9],
            'cy0': [1.0, 2.0, 0.0, 1.0],
            'cy1': [0.0, 1.0, 2.0, 0.0],
            'cy2': [2.0, 0.0, 1.0, 255.0],  # 255.0 = unassigned
        }
        df = pd.DataFrame(beads_data)
        df.to_csv(csv_path, index=False)

        # ----------------------------------------------------------------
        # ASSERTION 1: validate_bead_csv method must exist
        # ----------------------------------------------------------------
        assert hasattr(vm, 'validate_bead_csv'), (
            "FileManagerVM should have a 'validate_bead_csv' method"
        )

        # ----------------------------------------------------------------
        # ASSERTION 2: Method should return validation result
        # ----------------------------------------------------------------
        is_valid, error_msg, num_cycles = vm.validate_bead_csv(str(csv_path))

        assert is_valid is True, (
            f"Valid CSV should pass validation, but got error: {error_msg}"
        )
        assert error_msg is None or error_msg == "", (
            "Valid CSV should not return an error message"
        )
        assert num_cycles == 3, (
            "CSV with cy0, cy1, cy2 should detect 3 cycles"
        )

    def test_validate_bead_csv_rejects_missing_required_columns(
        self, mock_file_manager_vm, tmp_path
    ):
        """Validate that CSV without required columns fails validation.

        This test verifies:
        1. Missing 'x' or 'y' columns should fail validation
        2. Missing cycle columns should fail validation
        3. Error message explains what's missing
        """
        vm = mock_file_manager_vm

        # Create CSV without required columns
        csv_path = tmp_path / "invalid_beads.csv"
        invalid_data = {
            'cy0': [1.0, 2.0, 0.0],
            'cy1': [0.0, 1.0, 2.0],
            # Missing 'x' and 'y' columns
        }
        df = pd.DataFrame(invalid_data)
        df.to_csv(csv_path, index=False)

        # ----------------------------------------------------------------
        # ASSERTION: Invalid CSV should fail validation
        # ----------------------------------------------------------------
        is_valid, error_msg, num_cycles = vm.validate_bead_csv(str(csv_path))

        assert is_valid is False, (
            "CSV without 'x' and 'y' columns should fail validation"
        )
        assert error_msg is not None and len(error_msg) > 0, (
            "Failed validation should provide an error message"
        )
        assert 'x' in error_msg.lower() or 'y' in error_msg.lower(), (
            "Error message should mention missing 'x' or 'y' columns"
        )

    def test_store_uploaded_beads_updates_file_item(
        self, mock_file_manager_vm, tmp_path, tmp_tiff_path, signal_recorder
    ):
        """Store uploaded beads in FileItem and verify status update.

        This test verifies:
        1. FileManagerVM has store_uploaded_beads() method
        2. Beads DataFrame is stored in FileItem.beads
        3. Cycles dict is populated with cycle images
        4. FileStatus changes to BEADS_GENERATED
        5. file_information_update signal is emitted
        """
        vm = mock_file_manager_vm

        # Create valid bead CSV
        csv_path = tmp_path / "beads.csv"
        beads_data = {
            'x': [10.5, 20.3, 30.7],
            'y': [15.2, 25.8, 35.1],
            'cy0': [1.0, 2.0, 0.0],
            'cy1': [0.0, 1.0, 2.0],
        }
        df = pd.DataFrame(beads_data)
        df.to_csv(csv_path, index=False)

        # Create FileItems for reference and cycle files
        reference_path = tmp_tiff_path("reference.tif", shape=(3, 512, 512))
        cycle1_path = tmp_tiff_path("cycle1.tif", shape=(3, 512, 512))

        reference_item = FileItem(path=reference_path, status=FileStatus.ALIGNED)
        reference_item.shape = (3, 512, 512)
        reference_item.original_shape = (3, 512, 512)
        reference_item.dtype = "uint16"
        reference_item.metadata.reference_channel = 0
        reference_item.metadata.max_size = 512

        cycle1_item = FileItem(path=cycle1_path, status=FileStatus.ALIGNED)
        cycle1_item.shape = (3, 512, 512)
        cycle1_item.original_shape = (3, 512, 512)
        cycle1_item.dtype = "uint16"
        cycle1_item.metadata.reference_channel = 0
        cycle1_item.metadata.max_size = 512

        # Add items to VM
        vm.files[reference_path] = reference_item
        vm.files[cycle1_path] = cycle1_item
        vm.reference_item = reference_item

        # Create cycle assignments (cy0 -> reference, cy1 -> cycle1)
        cycle_assignments = {
            0: reference_item,
            1: cycle1_item,
        }

        # Connect signal recorder
        signal_recorder.connect(vm.file_information_update)

        # ----------------------------------------------------------------
        # ASSERTION 1: store_uploaded_beads method must exist
        # ----------------------------------------------------------------
        assert hasattr(vm, 'store_uploaded_beads'), (
            "FileManagerVM should have a 'store_uploaded_beads' method"
        )

        # ----------------------------------------------------------------
        # ACTION: Store uploaded beads
        # ----------------------------------------------------------------
        vm.store_uploaded_beads(
            str(csv_path),
            reference_item,
            cycle_assignments
        )

        # ----------------------------------------------------------------
        # ASSERTION 2: Beads DataFrame should be stored
        # ----------------------------------------------------------------
        stored_file = vm.files[reference_path]
        assert stored_file.beads is not None, (
            "FileItem.beads should be populated after storing uploaded beads"
        )
        assert isinstance(stored_file.beads, pd.DataFrame), (
            "FileItem.beads should be a pandas DataFrame"
        )
        assert len(stored_file.beads) == 3, (
            "FileItem.beads should contain all 3 bead rows from CSV"
        )
        assert 'x' in stored_file.beads.columns, (
            "Beads DataFrame should contain 'x' column"
        )
        assert 'y' in stored_file.beads.columns, (
            "Beads DataFrame should contain 'y' column"
        )

        # ----------------------------------------------------------------
        # ASSERTION 3: Cycles dict should be populated
        # ----------------------------------------------------------------
        assert stored_file.cycles is not None, (
            "FileItem.cycles should be populated with cycle images"
        )
        assert isinstance(stored_file.cycles, dict), (
            "FileItem.cycles should be a dictionary"
        )
        assert 'cy0' in stored_file.cycles, (
            "Cycles dict should contain 'cy0' key"
        )
        assert 'cy1' in stored_file.cycles, (
            "Cycles dict should contain 'cy1' key"
        )

        # ----------------------------------------------------------------
        # ASSERTION 4: FileStatus should change to BEADS_GENERATED
        # ----------------------------------------------------------------
        assert stored_file.status == FileStatus.BEADS_GENERATED, (
            "FileStatus should be BEADS_GENERATED after storing beads"
        )

        # ----------------------------------------------------------------
        # ASSERTION 5: file_information_update signal should be emitted
        # ----------------------------------------------------------------
        assert signal_recorder.get_call_count() > 0, (
            "file_information_update signal should be emitted"
        )
        last_signal_args = signal_recorder.get_last_args()
        assert last_signal_args is not None, (
            "Signal should have been emitted with arguments"
        )
        emitted_files = last_signal_args[0][0]  # First positional arg is the list
        assert reference_item in emitted_files, (
            "Emitted signal should include the updated reference item"
        )
