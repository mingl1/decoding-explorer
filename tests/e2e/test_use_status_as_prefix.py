import os
import numpy as np
import pytest
import tifffile
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import Qt
from unittest.mock import patch
from view.main_window import MainWindow
from model.status_enum import FileStatus


@pytest.fixture
def tiff_folder(tmp_path):
    """Generates a folder with dummy TIFF files."""
    folder = tmp_path / "test_images"
    folder.mkdir()

    for i in range(3):
        data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        file_path = folder / f"image_{i}.tif"
        tifffile.imwrite(file_path, data)

    return folder


def test_use_status_as_prefix_checkbox_enables_disables_prefix_input(qtbot, tiff_folder):
    """
    Test that the checkbox enables/disables the prefix input field.
    When checkbox is checked, prefix input should be disabled.
    When checkbox is unchecked, prefix input should be enabled.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    assert window.metadata_view.prefix_checkbox.isChecked() == True, "Checkbox should be checked when prefix matches status"

    window.metadata_view.prefix_checkbox.setChecked(False)
    qtbot.wait(100)

    assert window.metadata_view.prefix_input.isEnabled() == True, "Prefix input should be enabled when checkbox is unchecked"

    window.metadata_view.prefix_checkbox.setChecked(True)
    qtbot.wait(100)
    window.metadata_view.on_prefix_checkbox_changed(window.metadata_view.prefix_checkbox.checkState())

    assert window.metadata_view.prefix_input.isEnabled() == False, "Prefix input should be disabled when checkbox is checked"


def test_use_status_as_prefix_applies_status_to_prefix(qtbot, tiff_folder):
    """
    Test that when 'use_status_as_prefix' is checked and Apply is clicked,
    each file's prefix is set to its status value (lowercase).
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    qtbot.mouseClick(window.metadata_view.prefix_checkbox, Qt.MouseButton.LeftButton)
    qtbot.wait(100)

    qtbot.mouseClick(window.metadata_view.apply_btn, Qt.MouseButton.LeftButton)
    qtbot.wait(500)

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    file_item = window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert file_item.metadata.prefix == file_item.status.value.lower(), (
        f"Prefix should be '{file_item.status.value.lower()}' but got '{file_item.metadata.prefix}'"
    )


def test_use_status_as_prefix_with_manual_prefix_unchecked(qtbot, tiff_folder):
    """
    Test that when checkbox is unchecked and a manual prefix is entered,
    the manual prefix is applied instead of status-based prefix.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    assert window.metadata_view.prefix_checkbox.isChecked() == True, "Checkbox should be checked initially"

    window.metadata_view.prefix_checkbox.setChecked(False)
    qtbot.wait(100)
    assert window.metadata_view.prefix_checkbox.isChecked() == False, "Checkbox should be unchecked after setChecked(False)"

    manual_prefix = "my_custom_prefix"
    window.metadata_view.prefix_input.clear()
    qtbot.keyClicks(window.metadata_view.prefix_input, manual_prefix)

    qtbot.mouseClick(window.metadata_view.apply_btn, Qt.MouseButton.LeftButton)
    qtbot.wait(500)

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    file_item = window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert file_item.metadata.prefix == manual_prefix, (
        f"Prefix should be '{manual_prefix}' but got '{file_item.metadata.prefix}'"
    )


def test_use_status_as_prefix_checkbox_state_after_selection(qtbot, tiff_folder):
    """
    Test that checkbox state is correctly shown when selecting files.
    If all selected files have matching status-based prefixes, checkbox should be checked.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    assert window.metadata_view.prefix_checkbox.isChecked() == True, (
        "Checkbox should be checked when all files have status-based prefixes"
    )

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    assert window.metadata_view.prefix_checkbox.isChecked() == True, (
        "Checkbox should still be checked when single file has status-based prefix"
    )


def test_use_status_as_prefix_checkbox_unchecked_when_prefixes_mismatch(qtbot, tiff_folder):
    """
    Test that checkbox is unchecked when selected files have different prefixes
    that don't match their statuses.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    window.metadata_view.prefix_checkbox.setChecked(False)
    qtbot.wait(100)

    window.metadata_view.prefix_input.clear()
    qtbot.keyClicks(window.metadata_view.prefix_input, "custom_prefix")
    qtbot.mouseClick(window.metadata_view.apply_btn, Qt.MouseButton.LeftButton)
    qtbot.wait(500)

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectAll()
    qtbot.wait(200)

    assert window.metadata_view.prefix_checkbox.isChecked() == False, (
        "Checkbox should be unchecked when prefixes don't match statuses"
    )


def test_use_status_as_prefix_disabled_persists_after_reselection(qtbot, tiff_folder):
    """
    Test that when checkbox is checked, prefix input remains disabled
    after user unselects and reselects files.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    assert window.metadata_view.prefix_checkbox.isChecked() == True
    assert window.metadata_view.prefix_input.isEnabled() == False

    window.file_table_widget.clearSelection()
    qtbot.wait(200)

    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    assert window.metadata_view.prefix_input.isEnabled() == False, (
        "Prefix input should remain disabled after reselection when checkbox is checked"
    )

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectAll()
    qtbot.wait(200)

    assert window.metadata_view.prefix_input.isEnabled() == False, (
        "Prefix input should remain disabled after selecting all files when checkbox is checked"
    )


def test_use_status_as_prefix_updates_on_status_change(qtbot, tiff_folder):
    """
    Test that when user changes file status through the file list table,
    the prefix is updated to match the new status if 'use_status_as_prefix' is checked.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    assert window.metadata_view.prefix_checkbox.isChecked() == True

    initial_status = window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole).status
    assert window.metadata_view.prefix_input.text() == initial_status.value.lower()

    window.vm.set_status(window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole), FileStatus.SHADE_CORRECTED)
    qtbot.wait(200)

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    new_status = FileStatus.SHADE_CORRECTED
    assert window.metadata_view.prefix_input.text() == new_status.value.lower(), (
        f"Prefix should be '{new_status.value.lower()}' after status change, "
        f"but got '{window.metadata_view.prefix_input.text()}'"
    )


def test_use_status_as_prefix_shows_current_status_in_input(qtbot, tiff_folder):
    """
    Test that when 'use_status_as_prefix' is checked, the prefix input field
    displays the current status value (not editable).
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    assert window.metadata_view.prefix_checkbox.isChecked() == True

    file_item = window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert window.metadata_view.prefix_input.text() == file_item.status.value.lower(), (
        f"Prefix input should show '{file_item.status.value.lower()}' (status value), "
        f"but got '{window.metadata_view.prefix_input.text()}'"
    )

    assert window.metadata_view.prefix_input.isEnabled() == False, (
        "Prefix input should be disabled when checkbox is checked"
    )

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(1)
    qtbot.wait(200)

    file_item = window.file_table_widget.item(1, 0).data(Qt.ItemDataRole.UserRole)
    assert window.metadata_view.prefix_input.text() == file_item.status.value.lower(), (
        f"Prefix input should update to '{file_item.status.value.lower()}' for row 1"
    )


def test_use_status_as_prefix_unchecked_enables_input_for_editing(qtbot, tiff_folder):
    """
    Test that when checkbox is unchecked, prefix input becomes enabled
    and user can manually edit the prefix.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    assert window.metadata_view.prefix_checkbox.isChecked() == True
    assert window.metadata_view.prefix_input.isEnabled() == False

    window.metadata_view.prefix_checkbox.setChecked(False)
    qtbot.wait(100)

    assert window.metadata_view.prefix_input.isEnabled() == True, (
        "Prefix input should be enabled when checkbox is unchecked"
    )

    initial_text = window.metadata_view.prefix_input.text()

    window.metadata_view.prefix_input.clear()
    qtbot.keyClicks(window.metadata_view.prefix_input, "custom_")

    assert window.metadata_view.prefix_input.text() == "custom_", (
        "Prefix input should be editable when checkbox is unchecked"
    )

    qtbot.mouseClick(window.metadata_view.apply_btn, Qt.MouseButton.LeftButton)
    qtbot.wait(500)

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    file_item = window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert file_item.metadata.prefix == "custom_", (
        f"Custom prefix 'custom_' should be saved, but got '{file_item.metadata.prefix}'"
    )


def test_use_status_as_prefix_unchecked_persists_after_reselection(qtbot, tiff_folder):
    """
    Test that after unchecking use_status_as_prefix and applying,
    the checkbox remains unchecked when reselecting files.
    Should NOT auto-check based on prefix matching status.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectAll()
    qtbot.wait(500)

    assert window.metadata_view.prefix_checkbox.isChecked() == True

    window.metadata_view.prefix_checkbox.setChecked(False)
    qtbot.wait(100)

    window.metadata_view.prefix_input.clear()
    qtbot.keyClicks(window.metadata_view.prefix_input, "custom_prefix")

    qtbot.mouseClick(window.metadata_view.apply_btn, Qt.MouseButton.LeftButton)
    qtbot.wait(500)

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    assert window.metadata_view.prefix_checkbox.isChecked() == False, (
        "Checkbox should remain unchecked after reselection, "
        "not auto-check based on prefix matching status"
    )

    assert window.metadata_view.prefix_input.text() == "custom_prefix", (
        "Custom prefix should be preserved"
    )


def test_use_status_as_prefix_updates_prefix_immediately_on_status_change(qtbot, tiff_folder):
    """
    Test that when status is changed in file list table, the prefix input
    updates immediately, not just after reselection.
    """
    window = MainWindow()
    window.show()
    qtbot.addWidget(window)

    with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(tiff_folder)):
        qtbot.mouseClick(window.load_button, Qt.MouseButton.LeftButton)

    def check_files_loaded():
        assert window.file_table_widget.rowCount() == 3
    qtbot.waitUntil(check_files_loaded, timeout=5000)

    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    initial_status = window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole).status
    assert window.metadata_view.prefix_input.text() == initial_status.value.lower()

    file_item = window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole)
    window.vm.set_status(file_item, FileStatus.SHADE_CORRECTED)
    qtbot.wait(200)

    current_prefix = window.metadata_view.prefix_input.text()
    assert current_prefix == "corrected", (
        f"Prefix should update immediately to 'corrected' after status change, "
        f"but got '{current_prefix}'"
    )

    window.metadata_view.prefix_checkbox.setChecked(False)
    qtbot.wait(100)

    initial_text = window.metadata_view.prefix_input.text()

    window.metadata_view.prefix_input.clear()
    qtbot.keyClicks(window.metadata_view.prefix_input, "custom_")

    assert window.metadata_view.prefix_input.text() == "custom_", (
        "Prefix input should be editable when checkbox is unchecked"
    )

    qtbot.mouseClick(window.metadata_view.apply_btn, Qt.MouseButton.LeftButton)
    qtbot.wait(500)

    window.file_table_widget.clearSelection()
    window.file_table_widget.selectRow(0)
    qtbot.wait(200)

    file_item = window.file_table_widget.item(0, 0).data(Qt.ItemDataRole.UserRole)
    assert file_item.metadata.prefix == "custom_", (
        f"Custom prefix 'custom_' should be saved, but got '{file_item.metadata.prefix}'"
    )
