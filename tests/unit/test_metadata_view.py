from unittest.mock import patch

class TestMetadataView:
    """Unit tests for the MetadataView."""

    def test_on_metadata_corrected_updates_max_size_input(self, qapp, tmp_tiff_path):
        """on_metadata_corrected should update max_size_input with corrected value."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)

        view.max_size_input.setText("10000")

        corrected_values = {"max_size": 512}
        view.on_metadata_corrected(corrected_values)

        assert view.max_size_input.text() == "512"

    def test_on_metadata_corrected_handles_empty_values(self, qapp):
        """on_metadata_corrected should handle empty corrected values."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)

        original_text = view.max_size_input.text()
        view.on_metadata_corrected({})

        assert view.max_size_input.text() == original_text

    def test_on_metadata_corrected_ignores_other_keys(self, qapp):
        """on_metadata_corrected should ignore keys other than max_size."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)

        view.max_size_input.setText("10000")

        corrected_values = {"other_key": 512}
        view.on_metadata_corrected(corrected_values)

        assert view.max_size_input.text() == "10000"

    def test_text_input_saves_to_file_item_on_change(self, qapp, tmp_tiff_path):
        """Changing text input should save to FileItem immediately."""
        from model.file_item import FileItem
        from model.status_enum import FileStatus
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)

        file_path = tmp_tiff_path()
        item = FileItem(path=file_path, status=FileStatus.RAW)
        item.shape = (1, 512, 512)
        item.original_shape = (1, 512, 512)
        item.dtype = "uint16"
        vm.selected_files = [item]

        view.prefix_input.setText("test_prefix")

        assert item.metadata.prefix == "test_prefix"

    def test_max_size_input_saves_to_file_item_on_change(self, qapp, tmp_tiff_path):
        """Changing max_size_input should save to FileItem immediately (using max_viable_size)."""
        from model.file_item import FileItem
        from model.status_enum import FileStatus
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)

        file_path = tmp_tiff_path()
        item = FileItem(path=file_path, status=FileStatus.RAW)
        item.shape = (1, 512, 512)
        item.original_shape = (1, 512, 512)
        item.dtype = "uint16"
        vm.selected_files = [item]

        view.max_size_input.setText("1024")

        assert item.metadata.max_size == 512
        assert item.shape == (1, 512, 512)

    def test_max_size_change_updates_shape(self, qapp, tmp_tiff_path, mocker):
        """Changing max_size should update shape and emit shape update signal."""
        from model.file_item import FileItem
        from model.status_enum import FileStatus
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)

        file_path = tmp_tiff_path()
        item = FileItem(path=file_path, status=FileStatus.RAW)
        item.shape = (1, 512, 512)
        item.original_shape = (1, 512, 512)
        item.dtype = "uint16"
        vm.selected_files = [item]

        with patch.object(vm, 'file_shape_update_sig') as mock_sig:
            view.max_size_input.setText("256")

            assert item.metadata.max_size == 256
            assert item.shape == (1, 256, 256)
            mock_sig.emit.assert_called_once_with([item])

    def test_max_size_larger_than_original_uses_original(self, qapp, tmp_tiff_path, mocker):
        """When max_size is larger than original, should use original dimensions."""
        from model.file_item import FileItem
        from model.status_enum import FileStatus
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)

        file_path = tmp_tiff_path()
        item = FileItem(path=file_path, status=FileStatus.RAW)
        item.shape = (1, 512, 512)
        item.original_shape = (1, 512, 512)
        item.dtype = "uint16"
        vm.selected_files = [item]

        with patch.object(vm, 'file_shape_update_sig') as mock_sig:
            view.max_size_input.setText("10000")

            assert item.metadata.max_size == 512
            assert item.shape == (1, 512, 512)
            mock_sig.emit.assert_called_once_with([item])

    def test_metadata_section_toggle_hides_content_on_click(self, qapp):
        """Clicking on Metadata section title should hide its content widgets."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        assert view.prefix_input.isVisible()

        view._toggle_section("metadata")
        assert not view.prefix_input.isVisible()

    def test_metadata_section_toggle_shows_content_on_second_click(self, qapp):
        """Clicking on Metadata section title twice should show content again."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        view._toggle_section("metadata")
        view._toggle_section("metadata")
        assert view.prefix_input.isVisible()

    def test_align_arrays_section_toggle_hides_content(self, qapp):
        """Clicking on Align Arrays section title should hide its content widgets."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        assert view.num_tiles_input.isVisible()

        view._toggle_section("align_arrays")
        assert not view.num_tiles_input.isVisible()

    def test_bead_generation_section_toggle_hides_content(self, qapp):
        """Clicking on Bead Generation section title should hide its content widgets."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        assert view.generate_beads_btn.isVisible()

        view._toggle_section("bead_generation")
        assert not view.generate_beads_btn.isVisible()

    def test_statistics_section_toggle_hides_content(self, qapp):
        """Clicking on Statistics section title should hide its content widgets."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        assert view.total_beads_label.isVisible()

        view._toggle_section("statistics")
        assert not view.total_beads_label.isVisible()

    def test_all_sections_can_be_toggled_independently(self, qapp):
        """Each section should be toggleable independently of others."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        view._toggle_section("metadata")
        view._toggle_section("bead_generation")

        assert not view.prefix_input.isVisible()
        assert not view.generate_beads_btn.isVisible()
        assert view.num_tiles_input.isVisible()
        assert view.total_beads_label.isVisible()

    def test_metadata_section_toggle_hides_labels_and_inputs(self, qapp):
        """Clicking Metadata section should hide both labels and input widgets."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        assert view.prefix_label.isVisible()
        assert view.prefix_input.isVisible()

        view._toggle_section("metadata")

        assert not view.prefix_label.isVisible()
        assert not view.prefix_input.isVisible()

    def test_align_arrays_section_toggle_hides_labels_and_inputs(self, qapp):
        """Clicking Align Arrays section should hide both labels and input widgets."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        assert view.num_tiles_label.isVisible()
        assert view.num_tiles_input.isVisible()

        view._toggle_section("align_arrays")

        assert not view.num_tiles_label.isVisible()
        assert not view.num_tiles_input.isVisible()

    def test_collapse_processing_sections_closes_metadata_align_bead_gen(self, qapp):
        """collapse_processing_sections should hide metadata, align_arrays, and bead_generation sections."""
        from view.MetadataView import MetadataView
        from viewmodel.metadata_vm import MetadataVM

        vm = MetadataVM()
        view = MetadataView(None, vm=vm)
        view.show()

        assert view.prefix_input.isVisible()
        assert view.num_tiles_input.isVisible()
        assert view.generate_beads_btn.isVisible()

        view.collapse_processing_sections()

        assert not view.prefix_input.isVisible()
        assert not view.num_tiles_input.isVisible()
        assert not view.generate_beads_btn.isVisible()
        assert view.total_beads_label.isVisible()
