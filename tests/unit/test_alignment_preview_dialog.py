import numpy as np
from unittest.mock import patch

from view.alignment_preview_dialog import AlignmentPreviewDialog
from PyQt6.QtWidgets import QMessageBox


class TestAlignmentPreviewDialog:
    def test_manual_dialog_defaults_to_top_left_2000_crop(self, qapp):
        target = np.zeros((3200, 3200), dtype=np.uint16)
        moving = [np.zeros((3200, 3200), dtype=np.uint16)]

        dialog = AlignmentPreviewDialog(
            target,
            moving,
            can_edit=True,
            initial_preview_size=2000,
        )

        assert dialog.preview_size == 2000
        assert dialog.target_image.shape == (2000, 2000)
        assert dialog.moving_images[0].shape == (2000, 2000)
        assert "top-left" in dialog.preview_label.text().lower()

    def test_manual_dialog_preview_size_is_adjustable_up_to_max(self, qapp):
        target = np.zeros((3200, 3200), dtype=np.uint16)
        moving = [np.zeros((3200, 3200), dtype=np.uint16)]

        dialog = AlignmentPreviewDialog(
            target,
            moving,
            can_edit=True,
            initial_preview_size=2000,
        )
        dialog.preview_size_input.setText("2500")
        dialog.apply_preview_size()

        assert dialog.preview_size == 2500
        assert dialog.target_image.shape == (2500, 2500)
        assert dialog.moving_images[0].shape == (2500, 2500)

    def test_accept_alignment_emits_none_for_hidden_layers(self, qapp):
        target = np.zeros((64, 64), dtype=np.uint16)
        moving = [
            np.ones((64, 64), dtype=np.uint16),
            np.full((64, 64), 2, dtype=np.uint16),
        ]

        dialog = AlignmentPreviewDialog(
            target,
            moving,
            can_edit=True,
            can_emit=True,
            initial_preview_size=64,
        )
        received = []
        dialog.transformation_matrices.connect(lambda matrices: received.append(matrices))

        dialog.visibility_checkboxes[1].setChecked(False)
        dialog.accept_alignment()

        assert len(received) == 1
        assert len(received[0]) == 2
        assert isinstance(received[0][0], np.ndarray)
        assert received[0][1] is None

    def test_translation_applies_only_to_checked_layers(self, qapp):
        target = np.zeros((64, 64), dtype=np.uint16)
        moving = [
            np.ones((64, 64), dtype=np.uint16),
            np.full((64, 64), 2, dtype=np.uint16),
        ]

        dialog = AlignmentPreviewDialog(
            target,
            moving,
            can_edit=True,
            initial_preview_size=64,
        )
        dialog.visibility_checkboxes[1].setChecked(False)
        dialog.dx_input.setText("3")
        dialog.dy_input.setText("2")
        dialog.apply_manual_translation()

        first = dialog.image_view.moving_items[0].transform()
        second = dialog.image_view.moving_items[1].transform()

        assert int(round(first.dx())) == 3
        assert int(round(first.dy())) == 2
        assert int(round(second.dx())) == 0
        assert int(round(second.dy())) == 0

    def test_manual_dialog_initial_checked_indices(self, qapp):
        target = np.zeros((64, 64), dtype=np.uint16)
        moving = [
            np.ones((64, 64), dtype=np.uint16),
            np.full((64, 64), 2, dtype=np.uint16),
        ]

        dialog = AlignmentPreviewDialog(
            target,
            moving,
            can_edit=True,
            initial_checked_indices=[0],
            initial_preview_size=64,
        )

        assert dialog.visibility_checkboxes[0].isChecked() is True
        assert dialog.visibility_checkboxes[1].isChecked() is False

    def test_accept_alignment_warns_when_unchecked_layer_has_edits(self, qapp):
        target = np.zeros((64, 64), dtype=np.uint16)
        moving = [
            np.ones((64, 64), dtype=np.uint16),
            np.full((64, 64), 2, dtype=np.uint16),
        ]

        dialog = AlignmentPreviewDialog(
            target,
            moving,
            can_edit=True,
            can_emit=True,
            initial_checked_indices=[0, 1],
            initial_preview_size=64,
        )
        emitted = []
        dialog.transformation_matrices.connect(lambda matrices: emitted.append(matrices))

        dialog.dx_input.setText("1")
        dialog.dy_input.setText("0")
        dialog.apply_manual_translation()
        dialog.visibility_checkboxes[1].setChecked(False)

        with patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.No
        ) as mock_question:
            dialog.accept_alignment()

        mock_question.assert_called_once()
        assert emitted == []
        assert dialog.result_accepted is False
