"""Integration test for inspect_beads respecting max_size metadata field."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from model.file_item import FileItem
from model.status_enum import FileStatus
from viewmodel.file_manager_vm import FileManagerVM


class TestInspectBeadsMaxSize:
    """Test that inspect_beads properly uses max_size from metadata."""

    def test_inspect_beads_signal_includes_max_size(
        self, qapp, mock_file_manager_vm, tmp_tiff_path
    ):
        """Verify that inspect_beads emits max_size in the signal."""
        from PyQt6.QtCore import QCoreApplication

        file_path = tmp_tiff_path(shape=(1, 1024, 1024))
        file_item = FileItem(path=file_path, status=FileStatus.BEADS_GENERATED)
        file_item.shape = (1, 1024, 1024)
        file_item.original_shape = (1, 1024, 1024)
        file_item.dtype = "uint16"
        file_item.metadata.max_size = 512
        file_item.metadata.reference_channel = 0

        mock_file_manager_vm.files[file_item.path] = file_item

        beads_df = pd.DataFrame({"x": [100, 200], "y": [100, 200]})
        file_item.beads = beads_df

        cycles = {
            "cy0": [np.zeros((512, 512)), np.zeros((512, 512))],
            "cy1": [np.zeros((512, 512)), np.zeros((512, 512))],
        }
        file_item.cycles = cycles

        signal_args = {}

        def capture_signal(*args, **kwargs):
            signal_args["args"] = args
            signal_args["kwargs"] = kwargs

        mock_file_manager_vm.inspect_beads_signal.connect(capture_signal)

        protein_profile = pd.DataFrame()

        mock_file_manager_vm.inspect_beads(file_item, protein_profile)

        assert "args" in signal_args, "Signal was not emitted"
        args = signal_args["args"]

        assert len(args) >= 7, f"Expected at least 7 signal arguments, got {len(args)}"

        assert args[6] == 512, (
            f"Expected max_size=512 in signal, but got {args[6] if len(args) > 6 else 'not present'}"
        )

    def test_inspect_beads_respects_different_max_sizes(
        self, qapp, tmp_tiff_path
    ):
        """Verify that different max_size values are correctly passed through."""

        for max_size_val in [256, 512, 1024]:
            mock_file_manager_vm = FileManagerVM()

            file_path = tmp_tiff_path(shape=(1, 2048, 2048))
            file_item = FileItem(path=file_path, status=FileStatus.BEADS_GENERATED)
            file_item.shape = (1, 2048, 2048)
            file_item.original_shape = (1, 2048, 2048)
            file_item.dtype = "uint16"
            file_item.metadata.max_size = max_size_val
            file_item.metadata.reference_channel = 0

            mock_file_manager_vm.files[file_item.path] = file_item

            beads_df = pd.DataFrame({"x": [100], "y": [100]})
            file_item.beads = beads_df

            cycles = {"cy0": [np.zeros((max_size_val, max_size_val))]}
            file_item.cycles = cycles

            signal_args = {}

            def capture_signal(*args, **kwargs):
                signal_args["args"] = args
                signal_args["kwargs"] = kwargs

            mock_file_manager_vm.inspect_beads_signal.connect(capture_signal)

            protein_profile = pd.DataFrame()

            mock_file_manager_vm.inspect_beads(file_item, protein_profile)

            assert "args" in signal_args, f"Signal was not emitted for max_size={max_size_val}"
            args = signal_args["args"]

            assert len(args) >= 7, f"Expected at least 7 signal arguments for max_size={max_size_val}"
            assert args[6] == max_size_val, (
                f"Expected max_size={max_size_val} in signal, got {args[6] if len(args) > 6 else 'not present'}"
            )
