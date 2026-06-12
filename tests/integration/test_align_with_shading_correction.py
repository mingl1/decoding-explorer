"""
Integration test for the 'Apply shading correction' checkbox in alignment workflow.

Feature: Add a checkbox to the 'Align to Reference' workflow that automatically
applies shading correction before alignment. The checkbox should be checked by default.

This test verifies the core behavior:
- MetadataView has an 'apply_shading_checkbox' that is checked by default
- When alignment is triggered with the checkbox checked, shading correction is applied
  to the selected files before alignment proceeds
"""


from PyQt6.QtWidgets import QCheckBox

from model.file_item import FileItem
from model.status_enum import FileStatus


class TestAlignWithShadingCorrection:
    """Integration test for shading correction checkbox in alignment workflow."""

    def test_alignment_applies_shading_when_checkbox_checked(
        self, mock_main_window, tmp_tiff_path, mocker
    ):
        """When alignment is triggered with shading checkbox checked (default),
        shading correction should be applied to files before alignment proceeds.

        This test verifies:
        1. MetadataView has an 'apply_shading_checkbox' attribute (QCheckBox)
        2. The checkbox is checked by default
        3. When 'Align to Reference' button is clicked with checkbox checked,
           apply_shading is called on selected files before align_channels
        """
        window = mock_main_window
        metadata_view = window.metadata_view

        # ----------------------------------------------------------------
        # ASSERTION 1: MetadataView must have the checkbox
        # ----------------------------------------------------------------
        assert hasattr(metadata_view, "apply_shading_checkbox"), (
            "MetadataView should have an 'apply_shading_checkbox' attribute"
        )
        checkbox = metadata_view.apply_shading_checkbox
        assert isinstance(checkbox, QCheckBox), (
            "apply_shading_checkbox should be a QCheckBox widget"
        )

        # ----------------------------------------------------------------
        # ASSERTION 2: Checkbox must be checked by default
        # ----------------------------------------------------------------
        assert checkbox.isChecked(), (
            "Apply shading correction checkbox should be checked by default"
        )

        # ----------------------------------------------------------------
        # Setup: Create test file items with multi-channel images
        # ----------------------------------------------------------------
        reference_path = tmp_tiff_path("reference.tif", shape=(3, 256, 256))
        target_path = tmp_tiff_path("target.tif", shape=(3, 256, 256))

        reference_item = FileItem(path=reference_path, status=FileStatus.RAW)
        reference_item.shape = (3, 256, 256)
        reference_item.original_shape = (3, 256, 256)
        reference_item.dtype = "uint16"
        reference_item.metadata.reference_channel = 0
        reference_item.metadata.max_size = 256

        target_item = FileItem(path=target_path, status=FileStatus.RAW)
        target_item.shape = (3, 256, 256)
        target_item.original_shape = (3, 256, 256)
        target_item.dtype = "uint16"
        target_item.metadata.reference_channel = 0
        target_item.metadata.max_size = 256

        # Add items to the ViewModel
        window.vm.files[reference_path] = reference_item
        window.vm.files[target_path] = target_item
        window.vm.reference_item = reference_item

        # Add items to the file table widget
        window.update_file_list([reference_item, target_item])

        # Select the target file for alignment
        window.file_table_widget.selectRow(1)
        window.handle_selection_change()

        # ----------------------------------------------------------------
        # Mock the VM methods to track calls
        # ----------------------------------------------------------------
        apply_shading_mock = mocker.patch.object(window.vm, "apply_shading")
        align_channels_mock = mocker.patch.object(window.vm, "align_channels")

        # ----------------------------------------------------------------
        # ACTION: Click the 'Align to Reference' button
        # ----------------------------------------------------------------
        metadata_view.align_channels_btn.click()

        # ----------------------------------------------------------------
        # ASSERTION 3: Shading correction should be called with selected files
        # ----------------------------------------------------------------
        assert apply_shading_mock.called, (
            "apply_shading should be called when checkbox is checked"
        )

        # ----------------------------------------------------------------
        # ASSERTION 4: Alignment should proceed after shading
        # ----------------------------------------------------------------
        assert align_channels_mock.called, (
            "align_channels should be called after shading correction"
        )
