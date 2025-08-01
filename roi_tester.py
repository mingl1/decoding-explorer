import sys

import numpy as np
import pandas as pd
import tifffile as tiff
from PyQt6.QtWidgets import QApplication

from view import roi_inspector

if __name__ == "__main__":
    # Example usage: load a CSV file and print its contents

    #
    df = pd.read_csv("test_outputs/efficient_test.csv")
    print(df.head())
    print(f"Loaded {len(df)} beads from CSV.")

    # Load a TIFF file and print its shape
    cycle1 = tiff.imread(
        "./test_outputs/changed__SP13 16111 Fibrosis 0% Decoding Cycle 1.ome.tif"
    )[:, :10000, :10000]
    bf_image = np.array(cycle1)[0]
    cycle1 = np.array(cycle1)[1:]
    cycle2 = tiff.imread(
        "./test_outputs/aligned_SP13 16111 Fibrosis 0% Decoding Cycle 2.ome.tif"
    )[:, :10000, :10000]
    cycle2 = np.array(cycle2)[1:]
    print(f"Loaded BF image with shape {bf_image.shape}.")
    app = QApplication([])
    inspector = roi_inspector.ROI_Inspector(
        {"bf_image": bf_image, "beads": df, "cycles": {"cy0": cycle1, "cy1": cycle2}}
    )
    inspector.show()
    sys.exit(app.exec())
