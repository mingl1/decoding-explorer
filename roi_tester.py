import sys

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from PyQt6.QtWidgets import QApplication

from view import roi_inspector

if __name__ == "__main__":
    # Example usage: load a CSV file and print its contents

    #
    # df = pd.read_csv("./test_outputs/max_length_roi_tile_based_output.csv")
    # df = pd.read_csv("./test_outputs/fixed_radius.csv")
    # df = pd.read_csv("better_batching.csv")
    df = pd.read_csv("./new_test/new_beads2.csv")
    bboxs = df.pop("bbox")
    print(df.head())
    print(f"Loaded {len(df)} beads from CSV.")

    # Load a TIFF file and print its shape
    # cycle1 = tiff.imread(
    #     "./test_outputs/changed__SP13 16111 Fibrosis 0% Decoding Cycle 1.ome.tif"
    # )[:, :10000, :10000]
    cycle1 = tiff.imread("./new_test/changed__cycle 1.ome-001.tif")[:, :10000, :10000]
    bf_image = np.array(cycle1)[0]
    cycle1 = np.array(cycle1)[1:]
    cycle2 = tiff.imread(
        "./test_outputs/aligned_SP13 16111 Fibrosis 0% Decoding Cycle 2.ome.tif"
    )[:, :10000, :10000]
    cycle2 = tiff.imread("./new_test/aligned_cropped_cycle 2.ome.tif")[
        :, :10000, :10000
    ]
    cycle2 = np.array(cycle2)[1:]
    cycle3 = tiff.imread("./new_test/aligned_cropped_cycle 3.ome.tif")[
        :, :10000, :10000
    ]
    print(f"Loaded BF image with shape {bf_image.shape}.")
    app = QApplication([])
    # labeled_image = Image.open("labeled_image.png").convert("RGB")
    # labeled_image = np.array(labeled_image)
    # print(labeled_image.shape)
    labeled_image = None
    inspector = roi_inspector.ROI_Inspector(
        {
            "bf_image": bf_image,
            "beads": df,
            "cycles": {"cy0": cycle1, "cy1": cycle2, "cy2": cycle3},
            "bboxs": bboxs,
            "labeled_image": labeled_image,
        }
    )
    inspector.show()
    sys.exit(app.exec())
