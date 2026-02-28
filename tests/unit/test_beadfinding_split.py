import numpy as np

import image_processing


def _make_touching_disks(shape=(96, 96), c1=(42, 48), c2=(54, 48), radius=11):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    disk1 = (xx - c1[0]) ** 2 + (yy - c1[1]) ** 2 <= radius**2
    disk2 = (xx - c2[0]) ** 2 + (yy - c2[1]) ** 2 <= radius**2
    return disk1 | disk2


def test_split_large_components_splits_touching_blobs():
    merged_mask = _make_touching_disks()
    label_image = np.zeros(merged_mask.shape, dtype=np.int32)
    label_image[merged_mask] = 1
    areas = np.array([int(merged_mask.sum())], dtype=np.int64)

    split = image_processing._split_large_components(
        label_image=label_image,
        areas=areas,
        area_multiplier=1.0,
        min_distance=5,
        peak_rel_height=0.2,
    )

    assert split is not None
    assert int(split.max()) >= 2


def test_split_large_components_keeps_single_blob_when_only_one_peak():
    yy, xx = np.ogrid[:64, :64]
    disk = (xx - 32) ** 2 + (yy - 32) ** 2 <= 12**2
    label_image = np.zeros((64, 64), dtype=np.int32)
    label_image[disk] = 1
    areas = np.array([int(disk.sum())], dtype=np.int64)

    split = image_processing._split_large_components(
        label_image=label_image,
        areas=areas,
        area_multiplier=1.0,
        min_distance=6,
        peak_rel_height=0.4,
    )

    assert split is not None
    assert np.array_equal(split, label_image)
