import pytest
from unittest.mock import MagicMock, patch


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
