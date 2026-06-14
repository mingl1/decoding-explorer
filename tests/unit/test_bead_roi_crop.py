import numpy as np
import pandas as pd
from PyQt6.QtCore import QPointF
from model.file_item import FileItem
from view.main_window import filter_beads_by_rois, point_in_polygon
from view.crop_dialog import CropDialog, CropRectROI, CropCircleROI, CropPolyROI
from view.roi_inspector import ROIInspector


def test_point_in_polygon():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon(5, 5, poly) is True
    assert point_in_polygon(15, 5, poly) is False


def test_filter_beads_by_rois():
    beads = pd.DataFrame({
        "x": [5.0, 15.0, 25.0, 50.0],
        "y": [5.0, 5.0, 25.0, 50.0]
    })
    
    rois = [
        {"type": "rect", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        {"type": "circle", "cx": 25.0, "cy": 25.0, "r": 2.0},
        {"type": "polygon", "points": [(45.0, 45.0), (55.0, 45.0), (55.0, 55.0), (45.0, 55.0)]}
    ]
    
    filtered_include = filter_beads_by_rois(beads, rois, "include")
    assert len(filtered_include) == 3
    assert list(filtered_include["x"].values) == [5.0, 25.0, 50.0]

    filtered_exclude = filter_beads_by_rois(beads, rois, "exclude")
    assert len(filtered_exclude) == 1
    assert list(filtered_exclude["x"].values) == [15.0]


def test_crop_dialog_bead_mode(qapp):
    images = [np.zeros((100, 100), dtype=np.uint8)]
    beads = pd.DataFrame({
        "x": [10.0, 50.0],
        "y": [10.0, 50.0]
    })
    
    dialog = CropDialog(images, mode="bead", beads=beads)
    assert dialog.mode == "bead"
    assert dialog.beads is beads
    assert len(dialog.created_rois) == 0
    
    rect_roi = CropRectROI([0, 0], [20, 20])
    dialog._on_roi_added(rect_roi)
    assert len(dialog.created_rois) == 1
    assert dialog.created_rois[0]["type"] == "rect"
    
    assert "Selected Beads: 1 / 2" in dialog.beads_count_label.text()
    
    defs = dialog._get_roi_definitions()
    assert len(defs) == 1
    assert defs[0]["type"] == "rect"
    assert defs[0]["x1"] == 0.0
    assert defs[0]["y2"] == 20.0
    
    dialog._clear_all_rois()
    assert len(dialog.created_rois) == 0


def test_roi_inspector_tabs_and_crop(qapp):
    snapshot_data = {
        "bf_image": np.zeros((100, 100), dtype=np.uint8),
        "beads": pd.DataFrame({"x": [10.0, 50.0], "y": [10.0, 50.0]}),
        "cycles": {},
        "bright_fields": {"cy0": np.zeros((100, 100), dtype=np.uint8)}
    }
    
    file_item = FileItem(path="dummy.tif", beads=snapshot_data["beads"])
    file_item.bead_crop_rois = [
        {"type": "rect", "x1": 0.0, "y1": 0.0, "x2": 20.0, "y2": 20.0}
    ]
    file_item.bead_crop_mode = "exclude"
    
    inspector = ROIInspector(snapshot_data, file_item=file_item, initial_tab="crop")
    assert inspector.initial_tab == "crop"
    assert inspector.sidebar_tabs.currentIndex() == 1
    
    assert inspector.exclude_radio.isChecked() is True
    assert len(inspector.created_rois) == 1
    assert inspector.roi_list_widget.count() == 1
    
    assert "Selected Beads: 1 / 2" in inspector.beads_count_label.text()
    
    inspector.include_radio.setChecked(True)
    assert "Selected Beads: 1 / 2" in inspector.beads_count_label.text()
    
    inspector._clear_all_rois()
    assert len(inspector.created_rois) == 0


def test_roi_inspector_bead_color_baking(qapp):
    snapshot_data = {
        "bf_image": np.zeros((100, 100), dtype=np.uint8),
        "beads": pd.DataFrame({"x": [10.0, 50.0], "y": [10.0, 50.0]}),
        "cycles": {},
        "bright_fields": {"cy0": np.zeros((100, 100), dtype=np.uint8)}
    }
    file_item = FileItem(path="dummy.tif", beads=snapshot_data["beads"])
    inspector = ROIInspector(snapshot_data, file_item=file_item, initial_tab="inspect")
    
    # Verify cached images: normal contrast is populated on startup
    assert inspector.bg_images[False] is not None
    assert inspector.bg_images[True] is None
    
    # Retrieve background image
    bg_inspect = inspector._get_background_image(contrast=False)
    assert bg_inspect is not None
    assert inspector.bg_images[False] is bg_inspect
    
    # Verify that minimap is generated when show_minimap is called
    inspector._update_minimap()
    assert inspector.minimap_label.pixmap() is not None
    
    # Verify coordinate clipping when calling inspect_roi near borders
    from unittest.mock import patch
    with patch("view.roi_inspector.ROIGridDisplay") as mock_grid:
        inspector.radius_input.setText("5")
        inspector.scale_input.setText("1.0")
        # Coordinates inside target_image
        inspector.x_input.setText("5")
        inspector.y_input.setText("5")
        inspector.inspect_roi(None)
        mock_grid.assert_called_once()

