from unittest.mock import patch

import cv2
import numpy as np

from crop_anchor_finder import CropAnchorFinder, infer_tile_size
from model.file_item import FileItem, MetaData
from viewmodel.file_manager_vm import FileManagerVM


def test_crop_anchor_finder_template_matching():
    ref_img = np.zeros((200, 200), dtype=np.uint8)
    mov_img = np.zeros((500, 500), dtype=np.uint8)

    ref_img[20:60, 20:60] = 255
    mov_img[120:160, 120:160] = 255

    finder = CropAnchorFinder(
        ref_img,
        mov_img,
        patch_size=200,
        num_candidates=3,
    )

    mock_res = np.zeros((38, 38), dtype=np.float32)
    mock_res[15, 15] = 0.95

    with patch("cv2.matchTemplate", return_value=mock_res) as mock_match_template:
        candidates = finder.find_candidates()

        mock_match_template.assert_called_once()

        assert len(candidates) > 0
        best_cand = candidates[0]
        assert "anchor" in best_cand
        assert "score" in best_cand
        assert "flow_inliers" in best_cand
        assert "T" in best_cand


def test_crop_anchor_finder_dynamic_tile_size():
    x = np.linspace(0, 10 * np.pi, 5000)
    profile = np.sin(x) * 100 + 128
    dummy_img = np.tile(profile, (200, 1)).astype(np.uint8)

    inferred_size = infer_tile_size(dummy_img)
    assert abs(inferred_size - 1000) < 50


def test_crop_anchor_finder_translation_only():
    ref_img = np.zeros((200, 200), dtype=np.uint8)
    mov_img = np.zeros((500, 500), dtype=np.uint8)

    finder = CropAnchorFinder(
        ref_img,
        mov_img,
        patch_size=200,
        assume_no_transform=True,
    )

    mock_res = np.zeros((38, 38), dtype=np.float32)
    mock_res[15, 15] = 0.95

    mock_flow_M = np.array([[0.98, 0.17, 10.5], [-0.17, 0.98, 5.2]], dtype=np.float64)

    with (
        patch("cv2.matchTemplate", return_value=mock_res),
        patch(
            "crop_anchor_finder.try_optical_flow_alignment",
            return_value=(mock_flow_M, 150),
        ),
    ):
        candidates = finder.find_candidates()

        assert len(candidates) > 0
        best_cand = candidates[0]

        T = best_cand["T"]
        assert T[0, 0] == 1.0
        assert T[0, 1] == 0.0
        assert T[1, 0] == 0.0
        assert T[1, 1] == 1.0

        assert best_cand["angle"] == 0.0


def test_apply_crop_with_transform(tmp_path):
    from model.status_enum import FileStatus

    test_file_path = str(tmp_path / "test_img.tif")
    img_data = np.zeros((2, 100, 100), dtype=np.uint16)
    cv2.imwrite(test_file_path, img_data[0])

    file_item = FileItem(
        path=test_file_path,
        shape=(2, 100, 100),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=100),
    )

    vm = FileManagerVM()
    vm.files[test_file_path] = file_item

    transform = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float64)

    with patch("viewmodel.file_manager_vm.load_image", return_value=img_data):
        vm.apply_crop_with_transform(file_item, transform, 50, 60)

    updated_file = vm.files[test_file_path]
    assert updated_file.working_image.shape == (2, 60, 50)
    assert updated_file.status == FileStatus.CROPPED
    assert updated_file.metadata.prefix == "cropped"
    assert updated_file.metadata.crop_bounds == (0, 0, 50, 60)


def test_crop_updates_shape_and_max_size_metadata(tmp_path):
    from model.status_enum import FileStatus

    test_file_path = str(tmp_path / "test_img.tif")
    img_data = np.zeros((2, 100, 100), dtype=np.uint16)
    cv2.imwrite(test_file_path, img_data[0])

    file_item = FileItem(
        path=test_file_path,
        shape=(2, 100, 100),
        original_shape=(2, 100, 100),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=100),
    )

    vm = FileManagerVM()
    vm.files[test_file_path] = file_item

    with patch("viewmodel.file_manager_vm.load_image", return_value=img_data):
        vm.apply_crop([file_item], 10, 20, 60, 80)

    updated_file = vm.files[test_file_path]
    assert updated_file.working_image.shape == (2, 60, 50)
    assert updated_file.shape == (2, 60, 50)
    assert updated_file.metadata.max_size == 50
    assert updated_file.status == FileStatus.CROPPED

    vm.apply_metadata({"max_size": 30}, [updated_file])
    assert updated_file.metadata.max_size == 30
    assert updated_file.shape == (2, 30, 30)
    assert updated_file.working_image is not None
    assert updated_file.working_image.shape == (2, 60, 50)

    vm.apply_metadata({"max_size": 80}, [updated_file])
    assert updated_file.metadata.max_size == 50
    assert updated_file.shape == (2, 50, 50)
    assert updated_file.working_image is not None
    assert updated_file.working_image.shape == (2, 60, 50)


def test_crop_anchor_preserves_translated_content_after_max_size_change(tmp_path):
    from model.status_enum import FileStatus

    test_file_path = str(tmp_path / "test_img.tif")
    img_data = np.zeros((2, 100, 100), dtype=np.uint16)
    img_data[:, 0:10, 0:10] = 10
    img_data[:, 20:30, 10:20] = 99

    cv2.imwrite(test_file_path, img_data[0])

    file_item = FileItem(
        path=test_file_path,
        shape=(2, 100, 100),
        original_shape=(2, 100, 100),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=100),
    )

    vm = FileManagerVM()
    vm.files[test_file_path] = file_item

    transform = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float64)

    with patch("viewmodel.file_manager_vm.load_image", return_value=img_data):
        vm.apply_crop_with_transform(file_item, transform, 50, 60)

    updated_file = vm.files[test_file_path]
    assert updated_file.working_image[0, 0, 0] == 99

    vm.apply_metadata({"max_size": 30}, [updated_file])

    bf_img = vm._get_brightfield_image(updated_file)
    assert bf_img is not None
    assert bf_img[0, 0] == 99

    from viewmodel.file_io import load_and_constrain_image

    constrained_img = load_and_constrain_image(updated_file, 30)
    assert constrained_img.shape == (2, 30, 30)
    assert constrained_img[0, 0, 0] == 99


def test_register_respects_cropped_working_image(tmp_path):
    from align_arrays import Register
    from model.status_enum import FileStatus

    ref_path = str(tmp_path / "ref.tif")
    mov_path = str(tmp_path / "mov.tif")

    ref_file = FileItem(
        path=ref_path,
        shape=(2, 50, 50),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=50, reference_channel=0),
    )
    mov_file = FileItem(
        path=mov_path,
        shape=(2, 50, 50),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=50, reference_channel=0),
    )

    ref_working = np.zeros((2, 50, 50), dtype=np.uint16)
    ref_working[:, 0, 0] = 777
    ref_file.working_image = ref_working

    mov_working = np.zeros((2, 50, 50), dtype=np.uint16)
    mov_working[:, 0, 0] = 888
    mov_file.working_image = mov_working

    files_dict = {ref_path: ref_file, mov_path: mov_file}

    ref_params = {
        "max_size": 30,
        "alignment_layer": 0,
        "overlap": 10,
        "num_tiles": 1,
        "threshold": 0.5,
        "file_path": ref_path,
    }

    reg = Register(
        reference_file_item=ref_file,
        reference_params=ref_params,
        to_be_aligned_files=[mov_file],
        files_dict=files_dict,
    )
    reg._is_running = True

    with patch(
        "align_arrays.load_image_from_path",
        side_effect=AssertionError(
            "Should not load from disk when working_image is present"
        ),
    ):
        with patch.object(
            reg,
            "align_two_img_robust",
            return_value=((None, None), 0, 0, 10, 0, 0, 0.99),
        ):
            reg.run()

    assert reg.tifs[0]["image"][0, 0, 0] == 888


def test_register_handles_2d_shading_corrected_working_image(tmp_path):
    from align_arrays import Register
    from model.status_enum import FileStatus

    ref_path = str(tmp_path / "ref.tif")
    mov_path = str(tmp_path / "mov.tif")

    ref_file = FileItem(
        path=ref_path,
        shape=(2, 50, 50),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=50, reference_channel=0),
    )
    mov_file = FileItem(
        path=mov_path,
        shape=(2, 50, 50),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=50, reference_channel=0),
    )

    # 2D working image (only brightfield channel corrected)
    ref_working = np.zeros((50, 50), dtype=np.uint16)
    ref_working[0, 0] = 777
    ref_file.working_image = ref_working

    mov_working = np.zeros((50, 50), dtype=np.uint16)
    mov_working[0, 0] = 888
    mov_file.working_image = mov_working

    files_dict = {ref_path: ref_file, mov_path: mov_file}

    ref_params = {
        "max_size": 30,
        "alignment_layer": 0,
        "overlap": 10,
        "num_tiles": 1,
        "threshold": 0.5,
        "file_path": ref_path,
    }

    reg = Register(
        reference_file_item=ref_file,
        reference_params=ref_params,
        to_be_aligned_files=[mov_file],
        files_dict=files_dict,
    )
    reg._is_running = True

    dummy_disk_ref = np.ones((2, 50, 50), dtype=np.uint16) * 111
    dummy_disk_mov = np.ones((2, 50, 50), dtype=np.uint16) * 222

    def mock_load_image(path, shape, max_size=None):
        if "ref.tif" in path:
            return dummy_disk_ref[:, :max_size, :max_size]
        else:
            return dummy_disk_mov[:, :max_size, :max_size]

    with patch("align_arrays.load_image_from_path", side_effect=mock_load_image):
        with patch.object(
            reg,
            "align_two_img_robust",
            return_value=((None, None), 0, 0, 10, 0, 0, 0.99),
        ):
            reg.run()

    # The moving image channel 0 should be overwritten by mov_working (888)
    assert reg.tifs[0]["image"][0, 0, 0] == 888
    # The moving image channel 1 should come from disk load (222)
    assert reg.tifs[0]["image"][1, 0, 0] == 222


def test_shading_correction_task_respects_working_image(tmp_path):
    import threading

    from model.status_enum import FileStatus
    from viewmodel.tasks import shading_correction_task

    test_path = str(tmp_path / "test.tif")
    file_item = FileItem(
        path=test_path,
        shape=(2, 50, 50),
        status=FileStatus.RAW,
        metadata=MetaData(max_size=30, reference_channel=0),
    )

    # 3D working image
    working = np.ones((2, 50, 50), dtype=np.uint16) * 100
    working[0, 0, 0] = 500
    file_item.working_image = working

    files = {test_path: file_item}
    stop_event = threading.Event()

    with (
        patch("utils.shading_correction", side_effect=lambda x: x * 2),
        patch(
            "viewmodel.tasks.load_image",
            side_effect=AssertionError(
                "Should not read from disk when working_image is present"
            ),
        ),
    ):
        list(shading_correction_task([file_item], files, stop_event))

    assert file_item.working_image[0, 0, 0] == 1000
    assert file_item.working_image[1, 0, 0] == 100
