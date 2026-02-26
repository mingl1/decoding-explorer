import sys

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from PyQt6.QtWidgets import QApplication
from scipy.signal import correlate2d
from skimage.exposure import match_histograms

from utils import adjust_contrast, to_uint8
from view import roi_inspector
from skimage.exposure import adjust_sigmoid
from skimage.morphology import erosion, isotropic_erosion

def best_split_df_max_avg_gap(df, count_col):
    # Sort by counts
    df_sorted = df.sort_values(by=count_col).reset_index(drop=True)

    best_gap = float("-inf")
    best_split = None

    for i in range(1, len(df_sorted)):
        left = df_sorted.iloc[:i]
        right = df_sorted.iloc[i:]
        avg_left = left[count_col].mean()
        avg_right = right[count_col].mean()
        gap = avg_right - avg_left

        if gap > best_gap:
            best_gap = gap
            best_split = (left, right)

    return best_split, best_gap


def merge_bead_data_with_protein_profile(bead_data, protein_profile):
    bead_data = bead_data.merge(
        protein_profile, left_on=["cy0", "cy1"], right_on=["cy0", "cy1"], how="left"
    )
    bead_data.fillna("Invalid", inplace=True)
    mean_rows_per_protein = bead_data.groupby("Protein name")
    return bead_data, mean_rows_per_protein


def calculate_template_match(roi):
    gaussian_9x9 = np.array(
        [
            [1, 4, 7, 11, 14, 11, 7, 4, 1],
            [4, 16, 26, 41, 53, 41, 26, 16, 4],
            [7, 26, 42, 67, 87, 67, 42, 26, 7],
            [11, 41, 67, 107, 139, 107, 67, 41, 11],
            [14, 53, 87, 139, 181, 139, 87, 53, 14],
            [11, 41, 67, 107, 139, 107, 67, 41, 11],
            [7, 26, 42, 67, 87, 67, 42, 26, 7],
            [4, 16, 26, 41, 53, 41, 26, 16, 4],
            [1, 4, 7, 11, 14, 11, 7, 4, 1],
        ],
        dtype=np.float32,
    )
    # remove corners
    gaussian_9x9[0, 0] = 0
    gaussian_9x9[0, 1] = 0
    gaussian_9x9[1, 0] = 0
    gaussian_9x9[0, 7] = 0
    gaussian_9x9[0, 8] = 0
    gaussian_9x9[1, 8] = 0
    gaussian_9x9[7, 0] = 0
    gaussian_9x9[8, 0] = 0
    gaussian_9x9[8, 1] = 0
    gaussian_9x9[7, 8] = 0
    gaussian_9x9[8, 7] = 0
    gaussian_9x9[8, 8] = 0

    gaussian_9x9 /= np.sum(gaussian_9x9)
    roi = adjust_contrast(roi.astype(np.float32), 10, 90)
    score = correlate2d(roi.astype(np.float32), gaussian_9x9, mode="same")
    return np.max(score)


if __name__ == "__main__":
    # Example usage: load a CSV file and print its contents

    #
    # df = pd.read_csv("./test_outputs/max_length_roi_tile_based_output.csv")
    df = pd.read_csv("example2.csv")
    # df = pd.read_csv("better_batching.csv")
    # df = pd.read_csv("./new_test/new_beads2.csv")
    # round first two columns to int
    df.iloc[:, 0] = df.iloc[:, 0].round().astype(int)
    df.iloc[:, 1] = df.iloc[:, 1].round().astype(int)
    try:
        bboxs = df.pop("bbox")
    except KeyError:
        bboxs = None
    print(df.head())
    print(f"Loaded {len(df)} beads from CSV.")

    # Load a TIFF file and print its shape
    cycle1 = tiff.imread(
        "./test_outputs/changed__SP13 16111 Fibrosis 0% Decoding Cycle 1.ome.tif"
    )[:, :10000, :10000]
    bf1 = cycle1[0]
    # cycle1 = tiff.imread("./new_test/changed__cycle 1.ome-001.tif")[:, :10000, :10000]
    bf_image = np.array(cycle1)[0]
    cycle1 = np.array(cycle1)[1:]
    cycle2 = tiff.imread(
        "./test_outputs/aligned_SP13 16111 Fibrosis 0% Decoding Cycle 2.ome.tif"
    )[:, :10000, :10000]
    bf2 = cycle2[0]
    # cycle2 = tiff.imread("./new_test/aligned_cropped_cycle 2.ome.tif")[
    #     :, :10000, :10000
    # ]
    cycle2 = np.array(cycle2)[1:]
    # cycle3 = tiff.imread("./new_test/aligned_cropped_cycle 3.ome.tif")[
    #     :, :10000, :10000
    # ]
    # cycle3 = np.array(cycle3)[1:]
    print(f"Loaded BF image with shape {bf_image.shape}.")
    app = QApplication([])
    # labeled_image = Image.open("labeled_image.png").convert("RGB")
    # labeled_image = np.array(labeled_image)
    # print(labeled_image.shape)
    labeled_image = None
    protein_profile_paths = [
        "KDIG Channel Protein Decoding.csv",
        "Updated Biotin Decoding Scheme.csv",
    ]
    protein_profiles = [pd.read_csv(path) for path in protein_profile_paths]
    protein_profile = (
        pd.concat(protein_profiles).drop_duplicates().reset_index(drop=True)
    )
    cols = protein_profile.columns.tolist()
    renamed = {}
    for i in range(1, len(cols)):
        renamed[cols[i]] = "cy" + str(i - 1)
    protein_profile.rename(columns=renamed, inplace=True)
    bright_fields = {}
    bright_fields["cy0"] = bf1
    bright_fields["cy1"] = bf2
    num_layers = cycle1.shape[0]
    # match each cycle's histograms to their first layer then equalize them
    for i, cycle in enumerate([cycle1, cycle2]):
        reference_layer = cycle[0]  # First layer as reference
        matched_and_equalized = []
        for j in range(num_layers-1, -1,-1):
            img16 = cycle[j]
            # Match histogram to reference layer
            matched_img = match_histograms(img16, reference_layer)
            # Convert to 8-bit and equalize
            # img8 = np.zeros_like(img16, dtype=np.uint8)
            # img8 = cv2.normalize(matched_img, img8, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # img_eq = cv2.equalizeHist(img8)
            img_eq = adjust_sigmoid(cycle[j],0.2)
            img_eq = erosion(img_eq)
            matched_and_equalized.insert(0,img_eq)
        if i == 0:
            cycle1 = np.vstack((cycle1, np.array(matched_and_equalized)))
        else:
            cycle2 = np.vstack((cycle2, np.array(matched_and_equalized)))
    # # get low probability cycle combinations from dataframe
    # low_prob_cycles = df.groupby(["cy0", "cy1"]).size().reset_index(name="counts")
    # (invalid_beads, valid_beads), best_gap = best_split_df_max_avg_gap(
    #     low_prob_cycles, "counts"
    # )

    # print(len(invalid_beads), len(valid_beads), best_gap)
    # recalculate cycle for invalid beads
    inspector = roi_inspector.ROIInspector(
        {
            "bf_image": bf_image,
            "beads": df,
            "cycles": {
                "cy0": cycle1,
                "cy1": cycle2,
            },
            # "bboxs": None if bboxs is None else bboxs,
            # "labeled_image": labeled_image,
            "protein_profile": protein_profile,
            "bright_fields": bright_fields,
        }
    )
    print(df.columns)
    print(protein_profile.columns)
    inspector.show()
    sys.exit(app.exec())
