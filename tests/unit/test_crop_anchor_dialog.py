from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crop_anchor_finder import CropAnchorFinder
from model.file_item import FileItem, MetaData
from model.status_enum import FileStatus
from view.crop_anchor_dialog import CropAnchorDialog


def test_crop_anchor_dialog_background_candidates_different_page(qapp):
    ref_img = np.zeros((100, 100), dtype=np.uint8)
    mov_img1 = np.zeros((100, 100), dtype=np.uint8)
    mov_img2 = np.zeros((100, 100), dtype=np.uint8)

    file_item1 = FileItem(
        path="path1.tif",
        shape=(1, 100, 100),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=100),
    )
    file_item2 = FileItem(
        path="path2.tif",
        shape=(1, 100, 100),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=100),
    )

    moving_files = [
        {"label": "Mov 1", "file_item": file_item1, "image": mov_img1},
        {"label": "Mov 2", "file_item": file_item2, "image": mov_img2},
    ]

    dialog = CropAnchorDialog(
        reference_img=ref_img,
        moving_files=moving_files,
        ref_shape=(100, 100),
        pad=1.0,
    )

    assert dialog.current_page_idx == 0
    assert len(dialog.pages_state) == 2

    mock_finder = MagicMock(spec=CropAnchorFinder)
    mock_finder.isRunning.return_value = False

    with patch("view.crop_anchor_dialog.CropAnchorFinder", return_value=mock_finder):
        dialog._start_find()

    progress_slot = None
    candidates_ready_slot = None

    for call in mock_finder.progress.connect.call_args_list:
        progress_slot = call[0][0]
    for call in mock_finder.candidates_ready.connect.call_args_list:
        candidates_ready_slot = call[0][0]

    assert progress_slot is not None
    assert candidates_ready_slot is not None

    dialog._load_page(1)
    assert dialog.current_page_idx == 1

    progress_slot(50, "Finding...")
    assert dialog.progress.value() == 0
    assert dialog.pages_state[0]["progress_value"] == 50

    dummy_candidates = [
        {"score": 0.9, "angle": 10.0, "anchor": (10, 10), "T": np.eye(3)[:2]}
    ]
    candidates_ready_slot(dummy_candidates)

    assert len(dialog.pages_state[0]["candidates"]) == 1
    assert len(dialog.pages_state[1]["candidates"]) == 0
    assert dialog.list_ncc.count() == 0

    dialog._load_page(0)
    assert dialog.current_page_idx == 0
    assert dialog.list_ncc.count() == 1
    assert dialog.pages_state[0]["selected_index"] == 0
    assert len(dialog.pages_state[0]["box_items"]) > 0
    assert len(dialog.overview_scene.items()) > 1

    dialog._load_page(1)
    assert len(dialog.overview_scene.items()) == 1



def test_crop_anchor_dialog_concurrent_searches(qapp):
    ref_img = np.zeros((100, 100), dtype=np.uint8)
    mov_img1 = np.zeros((100, 100), dtype=np.uint8)
    mov_img2 = np.zeros((100, 100), dtype=np.uint8)

    file_item1 = FileItem(
        path="path1.tif",
        shape=(1, 100, 100),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=100),
    )
    file_item2 = FileItem(
        path="path2.tif",
        shape=(1, 100, 100),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=100),
    )

    moving_files = [
        {"label": "Mov 1", "file_item": file_item1, "image": mov_img1},
        {"label": "Mov 2", "file_item": file_item2, "image": mov_img2},
    ]

    dialog = CropAnchorDialog(
        reference_img=ref_img,
        moving_files=moving_files,
        ref_shape=(100, 100),
        pad=1.0,
    )

    assert dialog.current_page_idx == 0
    assert dialog.find_button.isEnabled() is True

    mock_finder1 = MagicMock(spec=CropAnchorFinder)
    mock_finder1.isRunning.return_value = True

    with patch("view.crop_anchor_dialog.CropAnchorFinder", return_value=mock_finder1):
        dialog._start_find()

    assert dialog.find_button.isEnabled() is False
    assert dialog.pages_state[0]["finder"] is mock_finder1

    dialog._load_page(1)
    assert dialog.current_page_idx == 1
    assert dialog.find_button.isEnabled() is True

    mock_finder2 = MagicMock(spec=CropAnchorFinder)
    mock_finder2.isRunning.return_value = True

    with patch("view.crop_anchor_dialog.CropAnchorFinder", return_value=mock_finder2):
        dialog._start_find()

    assert dialog.find_button.isEnabled() is False
    assert dialog.pages_state[1]["finder"] is mock_finder2

    finished_slot_finder1 = None
    for call in mock_finder1.finished.connect.call_args_list:
        finished_slot_finder1 = call[0][0]
    assert finished_slot_finder1 is not None

    mock_finder1.isRunning.return_value = False
    finished_slot_finder1()

    assert dialog.find_button.isEnabled() is False

    finished_slot_finder2 = None
    for call in mock_finder2.finished.connect.call_args_list:
        finished_slot_finder2 = call[0][0]
    assert finished_slot_finder2 is not None

    mock_finder2.isRunning.return_value = False
    finished_slot_finder2()

    assert dialog.find_button.isEnabled() is True
