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
