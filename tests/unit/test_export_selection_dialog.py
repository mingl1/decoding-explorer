from model.file_item import FileItem
from view.export_selection_dialog import ExportSelectionDialog


class TestExportSelectionDialog:
    def test_defaults_with_protein_assigned(self, qapp):
        reference_item = FileItem(path="/tmp/reference.tif")
        cycle_item = FileItem(path="/tmp/cycle2.tif")
        protein_item = FileItem(path="/tmp/protein.tif")
        dialog = ExportSelectionDialog(
            tiff_options=[
                ("Cycle 1 (Reference)", reference_item, False),
                ("Cycle 2", cycle_item, False),
                ("Protein TIFF", protein_item, True),
            ],
            beads_enabled=True,
            beads_checked=True,
            beads_format="csv",
        )

        states = {cb.text(): cb.isChecked() for cb, _ in dialog._tiff_checkboxes}
        assert states["Cycle 1 (Reference)"] is False
        assert states["Cycle 2"] is False
        assert states["Protein TIFF"] is True

    def test_defaults_without_protein_selects_reference_cycle(self, qapp):
        reference_item = FileItem(path="/tmp/reference.tif")
        cycle_item = FileItem(path="/tmp/cycle2.tif")
        dialog = ExportSelectionDialog(
            tiff_options=[
                ("Cycle 1 (Reference)", reference_item, True),
                ("Cycle 2", cycle_item, False),
            ],
            beads_enabled=False,
            beads_checked=False,
            beads_format="csv",
        )

        states = {cb.text(): cb.isChecked() for cb, _ in dialog._tiff_checkboxes}
        assert states["Cycle 1 (Reference)"] is True
        assert states["Cycle 2"] is False

    def test_beads_default_checked_when_available(self, qapp):
        reference_item = FileItem(path="/tmp/reference.tif")
        dialog = ExportSelectionDialog(
            tiff_options=[("Cycle 1 (Reference)", reference_item, True)],
            beads_enabled=True,
            beads_checked=True,
            beads_format="csv",
        )

        assert dialog.export_beads_checkbox.isEnabled()
        assert dialog.export_beads_checkbox.isChecked()

    def test_beads_format_defaults_to_csv(self, qapp):
        reference_item = FileItem(path="/tmp/reference.tif")
        dialog = ExportSelectionDialog(
            tiff_options=[("Cycle 1 (Reference)", reference_item, True)],
            beads_enabled=True,
            beads_checked=True,
            beads_format="csv",
        )

        assert dialog.get_beads_format() == "csv"
        assert dialog.beads_format_combo.currentText() == "CSV"

    def test_beads_format_enabled_state_follows_beads_checkbox(self, qapp):
        reference_item = FileItem(path="/tmp/reference.tif")
        dialog = ExportSelectionDialog(
            tiff_options=[("Cycle 1 (Reference)", reference_item, True)],
            beads_enabled=True,
            beads_checked=True,
            beads_format="csv",
        )

        assert dialog.beads_format_combo.isEnabled()
        dialog.export_beads_checkbox.setChecked(False)
        assert not dialog.beads_format_combo.isEnabled()
        dialog.export_beads_checkbox.setChecked(True)
        assert dialog.beads_format_combo.isEnabled()
