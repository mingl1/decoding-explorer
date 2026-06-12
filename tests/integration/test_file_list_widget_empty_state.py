"""Integration tests for FileTableWidget empty state visual improvements.

These tests verify the empty state styling requirements:
1. SVG icon size should be 96x96 pixels (currently 64x64)
2. Text font should be non-italic (currently italic)
3. Text font size should be 14pt (currently 12pt)
4. Text color should be #888888 (currently rgb(150,150,150))
5. Widget should have browse_files_button and browse_folder_button attributes
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QPushButton


class TestFileTableWidgetEmptyState:
    """Integration tests for the FileTableWidget empty state appearance.

    These tests inspect the paintEvent source code to verify the styling
    requirements are met. This approach avoids Qt painting issues in test
    environments while still verifying the implementation.
    """

    def test_svg_icon_size_is_96_pixels(self, mock_file_table_widget):
        """Empty state SVG icon should be 96x96 pixels, not 64x64.

        The paintEvent method should use svg_size = 96 when drawing the
        empty state icon.
        """
        widget = mock_file_table_widget

        # Widget should have no rows to show empty state
        assert widget.rowCount() == 0

        # Examine the paintEvent source code to find the svg_size value
        source = inspect.getsource(widget.paintEvent)

        # The current implementation uses svg_size = 64
        # After the fix, it should use svg_size = 96
        assert "svg_size = 96" in source, (
            "paintEvent should use svg_size = 96. Current implementation uses 'svg_size = 64'."
        )

    def test_empty_state_text_is_not_italic(self, mock_file_table_widget):
        """Empty state text font should NOT be italic.

        The paintEvent method should use font.setItalic(False) or not call
        setItalic at all when rendering the empty state text.
        """
        widget = mock_file_table_widget

        assert widget.rowCount() == 0

        # Examine the paintEvent source code
        source = inspect.getsource(widget.paintEvent)

        # The current implementation uses font.setItalic(True)
        # After the fix, it should either use setItalic(False) or not set italic
        has_italic_true = "setItalic(True)" in source

        assert not has_italic_true, (
            "paintEvent should NOT use font.setItalic(True). Font should be non-italic."
        )

    def test_empty_state_text_font_size_is_14pt(self, mock_file_table_widget):
        """Empty state text font size should be 14pt, not 12pt.

        The paintEvent method should use font.setPointSize(14) when
        rendering the empty state text.
        """
        widget = mock_file_table_widget

        assert widget.rowCount() == 0

        # Examine the paintEvent source code
        source = inspect.getsource(widget.paintEvent)

        # The current implementation uses font.setPointSize(12)
        # After the fix, it should use font.setPointSize(14)
        assert "setPointSize(14)" in source, (
            "paintEvent should use font.setPointSize(14). Current implementation uses setPointSize(12)."
        )

    def test_empty_state_text_color_is_888888(self, mock_file_table_widget):
        """Empty state text color should be #888888, not rgb(150,150,150).

        The paintEvent method should use QColor('#888888') or QColor(136, 136, 136)
        when setting the pen color for the empty state text.
        """
        widget = mock_file_table_widget

        assert widget.rowCount() == 0

        # Examine the paintEvent source code
        source = inspect.getsource(widget.paintEvent)

        # The current implementation uses QColor(150, 150, 150)
        # After the fix, it should use #888888 which is rgb(136, 136, 136)
        uses_new_hex_color = '"#888888"' in source or "'#888888'" in source
        uses_new_rgb_color = "QColor(136, 136, 136)" in source

        assert uses_new_hex_color or uses_new_rgb_color, (
            "paintEvent should use QColor('#888888') or QColor(136, 136, 136). "
            "Current implementation uses QColor(150, 150, 150)."
        )

    def test_widget_has_browse_files_button_attribute(self, mock_file_table_widget):
        """FileTableWidget should have a browse_files_button attribute.

        The widget should expose a browse_files_button for programmatic access
        to trigger file browsing functionality.
        """
        widget = mock_file_table_widget

        assert hasattr(widget, "browse_files_button"), (
            "FileTableWidget should have browse_files_button attribute"
        )
        assert isinstance(widget.browse_files_button, QPushButton), (
            "browse_files_button should be a QPushButton instance"
        )

    def test_widget_has_browse_folder_button_attribute(self, mock_file_table_widget):
        """FileTableWidget should have a browse_folder_button attribute.

        The widget should expose a browse_folder_button for programmatic access
        to trigger folder browsing functionality.
        """
        widget = mock_file_table_widget

        assert hasattr(widget, "browse_folder_button"), (
            "FileTableWidget should have browse_folder_button attribute"
        )
        assert isinstance(widget.browse_folder_button, QPushButton), (
            "browse_folder_button should be a QPushButton instance"
        )


class TestFileTableWidgetEmptyStateIntegration:
    """Integration tests for empty state with full widget interaction."""

    def test_empty_state_visible_when_no_files(self, mock_file_table_widget):
        """Empty state should be visible when widget has no files."""
        widget = mock_file_table_widget

        # Initially empty
        assert widget.rowCount() == 0

        # Verify the widget is configured to show empty state
        # The paintEvent should draw the empty state elements
        assert widget.svg_renderer.isValid(), "SVG renderer should be valid"

    def test_empty_state_not_rendered_when_files_present(
        self, mock_file_table_widget, mock_file_item
    ):
        """Empty state logic should not execute when widget has files.

        When files are added to the widget, the rowCount() > 0 check in
        paintEvent should prevent empty state rendering.
        """
        widget = mock_file_table_widget

        # Add a file
        widget.add_file_item(mock_file_item)

        # Widget should have a row now
        assert widget.rowCount() == 1

        # Verify the condition in paintEvent that controls empty state
        # The paintEvent only renders empty state when rowCount() == 0
        source = inspect.getsource(widget.paintEvent)
        assert "if self.rowCount() == 0:" in source, (
            "paintEvent should check rowCount() == 0 before rendering empty state"
        )

    def test_empty_state_returns_after_deleting_all_files(
        self, mock_file_table_widget, mock_file_item, mock_file_manager_vm
    ):
        """Empty state should return after all files are deleted."""
        widget = mock_file_table_widget
        vm = mock_file_manager_vm

        # Add and then remove a file
        widget.add_file_item(mock_file_item)
        vm.files[mock_file_item.path] = mock_file_item

        assert widget.rowCount() == 1

        # Delete the file
        widget.selectRow(0)
        with patch.object(vm, "delete_files"):
            widget.handle_delete_selected()

        # Widget should be empty again
        assert widget.rowCount() == 0, (
            "Widget should have 0 rows after deleting all files"
        )


class TestFileTableWidgetEmptyStateColorValues:
    """Tests that verify the exact color values used in the implementation."""

    def test_current_color_is_rgb_150_150_150(self, mock_file_table_widget):
        """Verify current implementation uses the old color value.

        This test documents the current (incorrect) behavior that uses
        rgb(150, 150, 150) instead of the required #888888.
        """
        widget = mock_file_table_widget
        source = inspect.getsource(widget.paintEvent)

        # Current implementation should have the old color
        # This test will pass now but helps document what needs to change
        uses_old_color = "QColor(150, 150, 150)" in source

        # We expect the new color to be used
        expected_color = QColor("#888888")

        # Verify what #888888 means in RGB
        assert expected_color.red() == 136, "Expected red = 136 for #888888"
        assert expected_color.green() == 136, "Expected green = 136 for #888888"
        assert expected_color.blue() == 136, "Expected blue = 136 for #888888"

        # The test that should fail - we want the new color
        uses_new_hex_color = '"#888888"' in source or "'#888888'" in source
        uses_new_rgb_color = "QColor(136, 136, 136)" in source

        assert uses_new_hex_color or uses_new_rgb_color, (
            "Expected QColor('#888888') or QColor(136, 136, 136), "
            "but found QColor(150, 150, 150) which is rgb(150,150,150)"
        )


class TestFileTableWidgetEmptyStateFontValues:
    """Tests that verify the exact font values used in the implementation."""

    def test_font_point_size_value(self, mock_file_table_widget):
        """Verify the font point size is set to 14, not 12."""
        widget = mock_file_table_widget
        source = inspect.getsource(widget.paintEvent)

        # Check for the correct point size
        has_size_14 = "setPointSize(14)" in source
        has_size_12 = "setPointSize(12)" in source

        assert has_size_14 and not has_size_12, (
            "Font should use setPointSize(14), not setPointSize(12)"
        )

    def test_font_italic_setting(self, mock_file_table_widget):
        """Verify the font is NOT set to italic."""
        widget = mock_file_table_widget
        source = inspect.getsource(widget.paintEvent)

        # Should NOT have setItalic(True)
        has_italic_true = "setItalic(True)" in source

        # May have setItalic(False) or no setItalic call at all
        assert not has_italic_true, (
            "Font should not use setItalic(True). Remove the italic setting or use setItalic(False)."
        )


class TestFileTableWidgetOrDivider:
    """Tests for the 'or' divider feature between drop text and browse buttons.

    The empty state should display:
    - Drop instruction text
    - A divider with horizontal lines and 'or' text (--- or ---)
    - Browse buttons

    The divider should use:
    - 10pt font size for 'or' text
    - #888888 color (same as other empty state text)
    - Horizontal lines on each side of 'or'
    """

    def test_paint_event_draws_or_divider_text(self, mock_file_table_widget):
        """paintEvent should draw 'or' text as a divider.

        The paintEvent method should contain code that draws the text 'or'
        between the drop instruction and the browse buttons.
        """
        widget = mock_file_table_widget
        source = inspect.getsource(widget.paintEvent)

        # The paintEvent should draw "or" text
        # Look for drawText with "or" string
        has_or_text = "drawText" in source and ('"or"' in source or "'or'" in source)

        assert has_or_text, (
            "paintEvent should contain code to draw 'or' text as a divider. "
            "Expected drawText call with 'or' string."
        )

    def test_paint_event_draws_horizontal_lines_for_divider(
        self, mock_file_table_widget
    ):
        """paintEvent should draw horizontal lines on each side of 'or'.

        The divider should look like: --- or ---
        This requires drawLine calls to render the horizontal lines.
        """
        widget = mock_file_table_widget
        source = inspect.getsource(widget.paintEvent)

        # The paintEvent should contain drawLine calls for the horizontal lines
        has_draw_line = "drawLine" in source

        assert has_draw_line, (
            "paintEvent should contain drawLine calls to draw horizontal lines "
            "on each side of the 'or' divider (--- or ---)."
        )

    def test_or_divider_uses_smaller_font_size(self, mock_file_table_widget):
        """The 'or' divider text should use a smaller font (10pt).

        The 'or' text should be rendered with setPointSize(10) to make it
        visually distinct from the main instruction text (14pt).
        """
        widget = mock_file_table_widget
        source = inspect.getsource(widget.paintEvent)

        # The paintEvent should set font size to 10 for the 'or' text
        has_10pt_font = "setPointSize(10)" in source

        assert has_10pt_font, (
            "paintEvent should use setPointSize(10) for the 'or' divider text. "
            "The 'or' text should be smaller (10pt) than the main text (14pt)."
        )

    def test_or_divider_uses_correct_color(self, mock_file_table_widget):
        """The 'or' divider should use the same gray color (#888888).

        Both the 'or' text and horizontal lines should use the #888888 color
        to maintain visual consistency with the empty state design.
        """
        widget = mock_file_table_widget
        source = inspect.getsource(widget.paintEvent)

        # Verify the color is used (already checked in other tests, but
        # ensuring the divider section also uses it)
        uses_gray_color = (
            '"#888888"' in source
            or "'#888888'" in source
            or "QColor(136, 136, 136)" in source
        )

        # Additionally, verify there's a pen set before drawLine
        # The drawLine should use the same gray color
        has_draw_line = "drawLine" in source

        assert uses_gray_color and has_draw_line, (
            "paintEvent should use #888888 color for the 'or' divider lines. "
            "Both the 'or' text and horizontal lines should use this gray color."
        )


class TestFileTableWidgetBrowseFilesButton:
    """Integration tests for the Browse Files button functionality.

    These tests verify:
    1. FileTableWidget has a browse_files method
    2. The browse_files_button.clicked signal is connected
    3. The browse_files method calls QFileDialog.getOpenFileNames with TIFF filter
    4. Selected files are passed to file_dropped_callback
    """

    def test_file_table_widget_has_browse_files_method(self, mock_file_table_widget):
        """FileTableWidget should have a browse_files method.

        The widget should have a browse_files (or on_browse_files) method
        that handles the browse files button click.
        """
        widget = mock_file_table_widget

        has_browse_files = hasattr(widget, "browse_files")
        has_on_browse_files = hasattr(widget, "on_browse_files")

        assert has_browse_files or has_on_browse_files, (
            "FileTableWidget should have a 'browse_files' or 'on_browse_files' method"
        )

    def test_browse_files_button_clicked_signal_is_connected(
        self, mock_file_table_widget
    ):
        """The browse_files_button.clicked signal should be connected.

        When browse_files_button is clicked, it should trigger the browse_files
        (or on_browse_files) method.
        """
        widget = mock_file_table_widget

        # Check that the button has receivers connected to its clicked signal
        button = widget.browse_files_button
        receivers_count = button.receivers(button.clicked)

        assert receivers_count > 0, (
            "browse_files_button.clicked signal should be connected to a slot. "
            "Expected at least 1 receiver, got 0."
        )

    def test_browse_files_method_calls_qfiledialog_with_tiff_filter(
        self, mock_file_table_widget
    ):
        """browse_files method should call QFileDialog.getOpenFileNames with TIFF filter.

        When browse_files is called, it should open a file dialog filtered
        to show only TIFF files (*.tif, *.tiff).
        """
        widget = mock_file_table_widget

        with patch(
            "view.file_table_widget.QFileDialog.getOpenFileNames"
        ) as mock_dialog:
            mock_dialog.return_value = ([], "")  # No files selected

            # Call the browse_files method (or on_browse_files)
            if hasattr(widget, "browse_files"):
                widget.browse_files()
            elif hasattr(widget, "on_browse_files"):
                widget.on_browse_files()
            else:
                pytest.fail("Widget should have browse_files or on_browse_files method")

            # Verify QFileDialog was called
            mock_dialog.assert_called_once()

            # Verify the filter includes TIFF files
            call_args = mock_dialog.call_args
            # Check positional or keyword arguments for the filter
            args, kwargs = call_args
            filter_arg = kwargs.get("filter") or (args[3] if len(args) > 3 else None)

            assert filter_arg is not None, (
                "QFileDialog.getOpenFileNames should be called with a filter argument"
            )
            assert "tif" in filter_arg.lower() or "tiff" in filter_arg.lower(), (
                f"Filter should include TIFF files. Got: {filter_arg}"
            )

    def test_browse_files_passes_selected_files_to_callback(
        self, mock_file_table_widget
    ):
        """Selected files from the dialog should be passed to file_dropped_callback.

        When the user selects files in the dialog, those files should be passed
        to the file_dropped_callback function.
        """
        widget = mock_file_table_widget
        callback_mock = MagicMock()
        widget.file_dropped_callback = callback_mock

        test_files = ["/path/to/file1.tiff", "/path/to/file2.tif"]

        with patch(
            "view.file_table_widget.QFileDialog.getOpenFileNames"
        ) as mock_dialog:
            mock_dialog.return_value = (test_files, "TIFF Files (*.tif *.tiff)")

            # Call the browse_files method
            if hasattr(widget, "browse_files"):
                widget.browse_files()
            elif hasattr(widget, "on_browse_files"):
                widget.on_browse_files()
            else:
                pytest.fail("Widget should have browse_files or on_browse_files method")

            # Verify callback was called with the selected files
            callback_mock.assert_called_once_with(test_files)

    def test_browse_files_does_not_call_callback_when_no_files_selected(
        self, mock_file_table_widget
    ):
        """Callback should NOT be called when user cancels the dialog.

        When the user cancels the file dialog (no files selected),
        the file_dropped_callback should not be invoked.
        """
        widget = mock_file_table_widget
        callback_mock = MagicMock()
        widget.file_dropped_callback = callback_mock

        with patch(
            "view.file_table_widget.QFileDialog.getOpenFileNames"
        ) as mock_dialog:
            mock_dialog.return_value = ([], "")  # User canceled - no files

            # Call the browse_files method
            if hasattr(widget, "browse_files"):
                widget.browse_files()
            elif hasattr(widget, "on_browse_files"):
                widget.on_browse_files()
            else:
                pytest.fail("Widget should have browse_files or on_browse_files method")

            # Verify callback was NOT called
            callback_mock.assert_not_called()

    def test_browse_files_button_click_triggers_file_dialog(
        self, mock_file_table_widget
    ):
        """Clicking browse_files_button should open the file dialog.

        This is an end-to-end test verifying that clicking the button
        actually triggers the file dialog through the connected slot.
        """
        widget = mock_file_table_widget

        with patch(
            "view.file_table_widget.QFileDialog.getOpenFileNames"
        ) as mock_dialog:
            mock_dialog.return_value = ([], "")

            # Simulate button click
            widget.browse_files_button.click()

            # Verify the dialog was opened
            (
                mock_dialog.assert_called_once(),
                "Clicking browse_files_button should open QFileDialog.getOpenFileNames",
            )


class TestFileTableWidgetBrowseFolderButton:
    """Integration tests for the Browse Folder button functionality.

    These tests verify:
    1. FileTableWidget has a browse_folder method (or on_browse_folder)
    2. The browse_folder_button.clicked signal is connected
    3. The browse_folder method calls QFileDialog.getExistingDirectory
    4. Selected folder is passed to file_dropped_callback as a list
    5. When user cancels (empty folder), callback should NOT be called
    """

    def test_file_table_widget_has_browse_folder_method(self, mock_file_table_widget):
        """FileTableWidget should have a browse_folder method.

        The widget should have a browse_folder (or on_browse_folder) method
        that handles the browse folder button click.
        """
        widget = mock_file_table_widget

        has_browse_folder = hasattr(widget, "browse_folder")
        has_on_browse_folder = hasattr(widget, "on_browse_folder")

        assert has_browse_folder or has_on_browse_folder, (
            "FileTableWidget should have a 'browse_folder' or 'on_browse_folder' method"
        )

    def test_browse_folder_button_clicked_signal_is_connected(
        self, mock_file_table_widget
    ):
        """The browse_folder_button.clicked signal should be connected.

        When browse_folder_button is clicked, it should trigger the browse_folder
        (or on_browse_folder) method.
        """
        widget = mock_file_table_widget

        # Check that the button has receivers connected to its clicked signal
        button = widget.browse_folder_button
        receivers_count = button.receivers(button.clicked)

        assert receivers_count > 0, (
            "browse_folder_button.clicked signal should be connected to a slot. "
            "Expected at least 1 receiver, got 0."
        )

    def test_browse_folder_method_calls_qfiledialog_getexistingdirectory(
        self, mock_file_table_widget
    ):
        """browse_folder method should call QFileDialog.getExistingDirectory.

        When browse_folder is called, it should open a directory selection dialog.
        """
        widget = mock_file_table_widget

        with patch(
            "view.file_table_widget.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = ""  # No folder selected

            # Call the browse_folder method (or on_browse_folder)
            if hasattr(widget, "browse_folder"):
                widget.browse_folder()
            elif hasattr(widget, "on_browse_folder"):
                widget.on_browse_folder()
            else:
                pytest.fail(
                    "Widget should have browse_folder or on_browse_folder method"
                )

            # Verify QFileDialog.getExistingDirectory was called
            mock_dialog.assert_called_once()

    def test_browse_folder_passes_selected_folder_to_callback_as_list(
        self, mock_file_table_widget
    ):
        """Selected folder from the dialog should be passed to file_dropped_callback as a list.

        When the user selects a folder in the dialog, that folder should be passed
        to the file_dropped_callback function as a list containing the folder path.
        """
        widget = mock_file_table_widget
        callback_mock = MagicMock()
        widget.file_dropped_callback = callback_mock

        test_folder = "/path/to/selected/folder"

        with patch(
            "view.file_table_widget.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = test_folder

            # Call the browse_folder method
            if hasattr(widget, "browse_folder"):
                widget.browse_folder()
            elif hasattr(widget, "on_browse_folder"):
                widget.on_browse_folder()
            else:
                pytest.fail(
                    "Widget should have browse_folder or on_browse_folder method"
                )

            # Verify callback was called with the folder as a list
            callback_mock.assert_called_once_with([test_folder])

    def test_browse_folder_does_not_call_callback_when_user_cancels(
        self, mock_file_table_widget
    ):
        """Callback should NOT be called when user cancels the dialog.

        When the user cancels the folder dialog (empty string returned),
        the file_dropped_callback should not be invoked.
        """
        widget = mock_file_table_widget
        callback_mock = MagicMock()
        widget.file_dropped_callback = callback_mock

        with patch(
            "view.file_table_widget.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = ""  # User canceled - empty string

            # Call the browse_folder method
            if hasattr(widget, "browse_folder"):
                widget.browse_folder()
            elif hasattr(widget, "on_browse_folder"):
                widget.on_browse_folder()
            else:
                pytest.fail(
                    "Widget should have browse_folder or on_browse_folder method"
                )

            # Verify callback was NOT called
            callback_mock.assert_not_called()

    def test_browse_folder_button_click_triggers_directory_dialog(
        self, mock_file_table_widget
    ):
        """Clicking browse_folder_button should open the directory dialog.

        This is an end-to-end test verifying that clicking the button
        actually triggers the directory dialog through the connected slot.
        """
        widget = mock_file_table_widget

        with patch(
            "view.file_table_widget.QFileDialog.getExistingDirectory"
        ) as mock_dialog:
            mock_dialog.return_value = ""

            # Simulate button click
            widget.browse_folder_button.click()

            # Verify the dialog was opened
            (
                mock_dialog.assert_called_once(),
                "Clicking browse_folder_button should open QFileDialog.getExistingDirectory",
            )


class TestFileTableWidgetButtonVisibility:
    """Integration tests for button visibility management based on table state.

    These tests verify:
    1. Buttons are visible when table is empty (rowCount() == 0)
    2. Buttons become hidden when files are present (rowCount() > 0)
    3. Widget has update_button_visibility method
    4. The add_file_item method triggers visibility update
    """

    def test_buttons_visible_when_table_is_empty(self, mock_file_table_widget):
        """Browse buttons should be visible when the table has no files.

        When the table is empty (rowCount() == 0), both browse_files_button
        and browse_folder_button should be visible to allow users to add files.
        """
        widget = mock_file_table_widget

        # Verify table is empty
        assert widget.rowCount() == 0, "Table should be empty for this test"

        # Buttons should be visible when empty
        assert widget.browse_files_button.isVisible(), (
            "browse_files_button should be visible when table is empty"
        )
        assert widget.browse_folder_button.isVisible(), (
            "browse_folder_button should be visible when table is empty"
        )

    def test_buttons_hidden_after_adding_file_item(
        self, mock_file_table_widget, mock_file_item
    ):
        """Browse buttons should be hidden after adding a file to the table.

        When files are present (rowCount() > 0), both browse_files_button
        and browse_folder_button should be hidden since the empty state
        is no longer shown.
        """
        widget = mock_file_table_widget

        # Initially empty - buttons should be visible
        assert widget.rowCount() == 0
        assert widget.browse_files_button.isVisible()
        assert widget.browse_folder_button.isVisible()

        # Add a file
        widget.add_file_item(mock_file_item)

        # Verify file was added
        assert widget.rowCount() == 1, "Table should have 1 row after adding file"

        # Buttons should now be hidden
        assert not widget.browse_files_button.isVisible(), (
            "browse_files_button should be hidden when table has files"
        )
        assert not widget.browse_folder_button.isVisible(), (
            "browse_folder_button should be hidden when table has files"
        )

    def test_widget_has_update_button_visibility_method(self, mock_file_table_widget):
        """FileTableWidget should have an update_button_visibility method.

        The widget should expose a method to update button visibility based
        on the current table state. This method can be named either
        'update_button_visibility' or '_update_button_visibility'.
        """
        widget = mock_file_table_widget

        has_public_method = hasattr(widget, "update_button_visibility")
        has_private_method = hasattr(widget, "_update_button_visibility")

        assert has_public_method or has_private_method, (
            "FileTableWidget should have 'update_button_visibility' or "
            "'_update_button_visibility' method"
        )

    def test_update_button_visibility_shows_buttons_when_empty(
        self, mock_file_table_widget
    ):
        """update_button_visibility should show buttons when table is empty.

        Calling update_button_visibility (or _update_button_visibility) when
        the table has no rows should make both buttons visible.
        """
        widget = mock_file_table_widget

        # Ensure table is empty
        assert widget.rowCount() == 0

        # Call the visibility update method
        if hasattr(widget, "update_button_visibility"):
            widget.update_button_visibility()
        elif hasattr(widget, "_update_button_visibility"):
            widget._update_button_visibility()
        else:
            pytest.fail(
                "Widget should have 'update_button_visibility' or "
                "'_update_button_visibility' method"
            )

        # Buttons should be visible
        assert widget.browse_files_button.isVisible(), (
            "browse_files_button should be visible after update when table is empty"
        )
        assert widget.browse_folder_button.isVisible(), (
            "browse_folder_button should be visible after update when table is empty"
        )

    def test_update_button_visibility_hides_buttons_when_files_present(
        self, mock_file_table_widget, mock_file_item
    ):
        """update_button_visibility should hide buttons when table has files.

        Calling update_button_visibility (or _update_button_visibility) when
        the table has rows should hide both buttons.
        """
        widget = mock_file_table_widget

        # Add a file to the table
        widget.add_file_item(mock_file_item)
        assert widget.rowCount() == 1

        # Call the visibility update method
        if hasattr(widget, "update_button_visibility"):
            widget.update_button_visibility()
        elif hasattr(widget, "_update_button_visibility"):
            widget._update_button_visibility()
        else:
            pytest.fail(
                "Widget should have 'update_button_visibility' or "
                "'_update_button_visibility' method"
            )

        # Buttons should be hidden
        assert not widget.browse_files_button.isVisible(), (
            "browse_files_button should be hidden after update when table has files"
        )
        assert not widget.browse_folder_button.isVisible(), (
            "browse_folder_button should be hidden after update when table has files"
        )

    def test_add_file_item_calls_visibility_update(
        self, mock_file_table_widget, mock_file_item
    ):
        """add_file_item should trigger button visibility update.

        When add_file_item is called, it should also call the visibility
        update method to ensure buttons are hidden after a file is added.
        """
        widget = mock_file_table_widget

        # Patch the visibility update method to track if it's called
        visibility_method_name = None
        if hasattr(widget, "update_button_visibility"):
            visibility_method_name = "update_button_visibility"
        elif hasattr(widget, "_update_button_visibility"):
            visibility_method_name = "_update_button_visibility"

        if visibility_method_name is None:
            pytest.fail(
                "Widget should have 'update_button_visibility' or "
                "'_update_button_visibility' method"
            )

        with patch.object(
            widget,
            visibility_method_name,
            wraps=getattr(widget, visibility_method_name),
        ) as mock_visibility:
            widget.add_file_item(mock_file_item)

            # Verify the visibility update was called
            (
                mock_visibility.assert_called(),
                f"add_file_item should call {visibility_method_name}",
            )

    def test_buttons_become_visible_again_after_removing_all_files(
        self, mock_file_table_widget, mock_file_item, mock_file_manager_vm
    ):
        """Buttons should become visible again when all files are removed.

        After all files are deleted from the table, the buttons should
        become visible again to allow users to add more files.
        """
        widget = mock_file_table_widget
        vm = mock_file_manager_vm

        # Add a file
        widget.add_file_item(mock_file_item)
        vm.files[mock_file_item.path] = mock_file_item
        assert widget.rowCount() == 1

        # Buttons should be hidden with files present
        # (This relies on add_file_item triggering visibility update)
        # For now, manually check if method exists and call it
        if hasattr(widget, "update_button_visibility"):
            widget.update_button_visibility()
        elif hasattr(widget, "_update_button_visibility"):
            widget._update_button_visibility()

        # Delete all files
        widget.selectRow(0)
        with patch.object(vm, "delete_files"):
            widget.handle_delete_selected()

        # Table should be empty
        assert widget.rowCount() == 0, "Table should be empty after deleting all files"

        # Update visibility (this should be called by handle_delete_selected ideally)
        if hasattr(widget, "update_button_visibility"):
            widget.update_button_visibility()
        elif hasattr(widget, "_update_button_visibility"):
            widget._update_button_visibility()

        # Buttons should be visible again
        assert widget.browse_files_button.isVisible(), (
            "browse_files_button should be visible after removing all files"
        )
        assert widget.browse_folder_button.isVisible(), (
            "browse_folder_button should be visible after removing all files"
        )
