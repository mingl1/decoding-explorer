from unittest.mock import MagicMock

import pandas as pd


class TestMetadataVM:
    """Unit tests for the MetadataVM ViewModel."""

    def test_update_corrected_metadata_emits_signal(
        self, mock_metadata_vm, signal_recorder
    ):
        """update_corrected_metadata should emit metadata_corrected_sig."""
        vm = mock_metadata_vm

        signal_recorder.connect(vm.metadata_corrected_sig)

        vm.update_corrected_metadata("max_size", 512)

        assert signal_recorder.get_call_count() == 1
        call_args = signal_recorder.get_last_args()
        assert call_args is not None
        args, kwargs = call_args
        corrected_values = args[0]
        assert "max_size" in corrected_values
        assert corrected_values["max_size"] == 512

    def test_apply_metadata_emits_metadata_applied_sig(
        self, mock_metadata_vm, mock_file_item, signal_recorder
    ):
        """apply_metadata should emit metadata_applied_sig."""
        vm = mock_metadata_vm
        vm.selected_files = [mock_file_item]

        signal_recorder.connect(vm.metadata_applied_sig)

        vm.apply_metadata({"max_size": 1000})

        assert signal_recorder.get_call_count() == 1

    def test_apply_metadata_with_use_status_as_prefix(
        self, mock_metadata_vm, mock_file_item, signal_recorder
    ):
        """apply_metadata with use_status_as_prefix should emit the changes dict."""
        vm = mock_metadata_vm
        mock_file_item.status = MagicMock(value="ALIGNED")
        vm.selected_files = [mock_file_item]

        signal_recorder.connect(vm.metadata_applied_sig)

        vm.apply_metadata({"use_status_as_prefix": True})

        assert signal_recorder.get_call_count() == 1
        call_args = signal_recorder.get_last_args()
        args, kwargs = call_args
        result = args[0]
        assert "use_status_as_prefix" in result
        assert result["use_status_as_prefix"] == True

    def test_set_protein_files_reads_utf16_csv(
        self, mock_metadata_vm, signal_recorder, tmp_path, capsys
    ):
        vm = mock_metadata_vm
        signal_recorder.connect(vm.update_overview_sig)
        source_df = pd.DataFrame({"Protein name": ["A"], "Cycle 1": [1]})
        protein_path = tmp_path / "protein_utf16.csv"
        source_df.to_csv(protein_path, index=False, encoding="utf-16")

        vm.set_protein_files([str(protein_path)])
        captured = capsys.readouterr().out

        assert signal_recorder.get_call_count() == 1
        assert list(vm.protein_df.columns) == ["Protein name", "cy0"]
        assert vm.protein_df.iloc[0]["Protein name"] == "A"
        assert vm.protein_df.iloc[0]["cy0"] == 1
        assert "Uploaded protein dataframe:" in captured
        assert "Protein name" in captured

    def test_set_protein_files_reads_utf16le_csv_without_bom(
        self, mock_metadata_vm, signal_recorder, tmp_path
    ):
        vm = mock_metadata_vm
        signal_recorder.connect(vm.update_overview_sig)
        source_df = pd.DataFrame({"Protein name": ["B"], "Cycle 1": [2]})
        protein_path = tmp_path / "protein_utf16le.csv"
        source_df.to_csv(protein_path, index=False, encoding="utf-16le")

        vm.set_protein_files([str(protein_path)])

        assert signal_recorder.get_call_count() == 1
        assert list(vm.protein_df.columns) == ["Protein name", "cy0"]
        assert vm.protein_df.iloc[0]["Protein name"] == "B"
        assert vm.protein_df.iloc[0]["cy0"] == 2

    def test_set_protein_files_emits_error_on_unreadable_input(
        self, mock_metadata_vm, signal_recorder, tmp_path
    ):
        vm = mock_metadata_vm
        signal_recorder.connect(vm.error_sig)
        unreadable_path = tmp_path / "not_a_file.csv"
        unreadable_path.mkdir()

        vm.set_protein_files([str(unreadable_path)])

        assert signal_recorder.get_call_count() == 1
        args, kwargs = signal_recorder.get_last_args()
        assert "Failed to load protein key files" in args[0]

    def test_set_protein_files_normalizes_misaligned_name_column(
        self, mock_metadata_vm, signal_recorder, tmp_path
    ):
        vm = mock_metadata_vm
        signal_recorder.connect(vm.update_overview_sig)

        first_df = pd.DataFrame(
            {"Protein name": ["Seq 16-55"], "Cycle 1": [3], "Cycle 2": [3]}
        )
        second_df = pd.DataFrame(
            {"Cycle 1": [3], "Cycle 2": [1], "Protein name ": ["Yap-1"]}
        )

        first_path = tmp_path / "protein_a.csv"
        second_path = tmp_path / "protein_b.csv"
        first_df.to_csv(first_path, index=False)
        second_df.to_csv(second_path, index=False)

        vm.set_protein_files([str(first_path), str(second_path)])

        assert signal_recorder.get_call_count() == 1
        assert list(vm.protein_df.columns) == ["Protein name", "cy0", "cy1"]
        assert "cy2" not in vm.protein_df.columns

        row = vm.protein_df[vm.protein_df["Protein name"] == "Yap-1"].iloc[0]
        assert row["cy0"] == 3
        assert row["cy1"] == 1
