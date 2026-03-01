from model.file_item import FileItem
from view.cycle_assignment_dialog import CycleAssignmentDialog


def _make_files(n: int) -> list[FileItem]:
    return [FileItem(path=f"/tmp/file_{i}.tif") for i in range(n)]


class TestCycleAssignmentDialog:
    def test_defaults_to_two_cycle_rows(self, qapp):
        files = _make_files(3)
        dialog = CycleAssignmentDialog(files, default_cycle_count=2)

        assert dialog.table.rowCount() == 2

    def test_assign_all_files_expands_rows(self, qapp):
        files = _make_files(4)
        dialog = CycleAssignmentDialog(files, default_cycle_count=2)

        dialog._assign_all_files()

        assert dialog.table.rowCount() == 4

    def test_delete_selected_cycle_row(self, qapp):
        files = _make_files(4)
        dialog = CycleAssignmentDialog(files, default_cycle_count=2)
        dialog._assign_all_files()
        dialog.table.selectRow(2)

        dialog._delete_selected_cycle_row()

        assert dialog.table.rowCount() == 3
        assert dialog.table.item(0, 0).text() == "Cycle 1"
        assert dialog.table.item(1, 0).text() == "Cycle 2"
        assert dialog.table.item(2, 0).text() == "Cycle 3"

    def test_get_assignments_allows_two_cycles_plus_protein(self, qapp):
        files = _make_files(3)
        dialog = CycleAssignmentDialog(files, default_cycle_count=2)

        dialog.comboboxes[0].setCurrentText("file_0.tif")
        dialog.comboboxes[1].setCurrentText("file_1.tif")
        dialog.protein_combo.setCurrentText("file_2.tif")

        assignments = dialog.get_assignments()

        assert assignments is not None
        assert set(assignments.keys()) == {1, 2}
        assert assignments[1].path.endswith("file_0.tif")
        assert assignments[2].path.endswith("file_1.tif")

    def test_auto_assign_prioritizes_protein_name_over_cycle_name(self, qapp):
        files = [
            FileItem(path="/tmp/sample_cycle_1.tif"),
            FileItem(path="/tmp/sample_cycle_2.tif"),
            FileItem(path="/tmp/sample_cycle_1_protein.tif"),
        ]

        dialog = CycleAssignmentDialog(files, default_cycle_count=2)
        assignments = dialog.get_assignments()
        protein_file = dialog.get_protein_file()

        assert assignments is not None
        assert assignments[1].path.endswith("sample_cycle_1.tif")
        assert assignments[2].path.endswith("sample_cycle_2.tif")
        assert protein_file is not None
        assert protein_file.path.endswith("sample_cycle_1_protein.tif")

    def test_auto_assign_fills_remaining_cycle_slots_after_name_matches(self, qapp):
        files = [
            FileItem(path="/tmp/untagged_reference.tif"),
            FileItem(path="/tmp/round_cycle_2.tif"),
            FileItem(path="/tmp/panel_protein.tif"),
        ]

        dialog = CycleAssignmentDialog(files, default_cycle_count=2)
        assignments = dialog.get_assignments()
        protein_file = dialog.get_protein_file()

        assert assignments is not None
        assert assignments[1].path.endswith("untagged_reference.tif")
        assert assignments[2].path.endswith("round_cycle_2.tif")
        assert protein_file is not None
        assert protein_file.path.endswith("panel_protein.tif")

    def test_auto_assign_matches_names_case_and_space_insensitive(self, qapp):
        files = [
            FileItem(path="/tmp/sample c Y c l e 1.tif"),
            FileItem(path="/tmp/sample CyClE 2.tif"),
            FileItem(path="/tmp/Panel PRO TEIN Key.tif"),
        ]

        dialog = CycleAssignmentDialog(files, default_cycle_count=2)
        assignments = dialog.get_assignments()
        protein_file = dialog.get_protein_file()

        assert assignments is not None
        assert assignments[1].path.endswith("sample c Y c l e 1.tif")
        assert assignments[2].path.endswith("sample CyClE 2.tif")
        assert protein_file is not None
        assert protein_file.path.endswith("Panel PRO TEIN Key.tif")
