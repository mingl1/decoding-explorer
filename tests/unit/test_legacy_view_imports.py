def test_legacy_main_window_module_import():
    from view.main_window import MainWindow as CanonicalMainWindow
    from view.MainWindow import MainWindow as LegacyMainWindow

    assert LegacyMainWindow is CanonicalMainWindow


def test_legacy_file_table_module_import():
    from view.file_table_widget import FileTableWidget as CanonicalFileTableWidget
    from view.FileListWidget import FileTableWidget as LegacyFileTableWidget

    assert LegacyFileTableWidget is CanonicalFileTableWidget


def test_legacy_workflow_panel_alias():
    from view.decoding_workflow_panel import DecodingWorkflowPanel
    from view.MetadataView import MetadataView as LegacyMetadataView

    assert LegacyMetadataView is DecodingWorkflowPanel


def test_legacy_cycle_assignment_alias():
    from view.cycle_assignment_dialog import CycleAssignmentDialog
    from view.CycleAssignmentWidget import (
        CycleAssignmentWidget as LegacyCycleAssignmentWidget,
    )

    assert LegacyCycleAssignmentWidget is CycleAssignmentDialog


def test_legacy_roi_aliases():
    from view.roi_inspector import (
        ROI_Grid_Display,
        ROI_Inspector,
        ROIGridDisplay,
        ROIInspector,
    )

    assert ROI_Inspector is ROIInspector
    assert ROI_Grid_Display is ROIGridDisplay
