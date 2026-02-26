import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from model.file_item import FileItem, MetaData
from model.status_enum import FileStatus
from PyQt6.QtWidgets import QTableWidgetItem
from view.file_table_widget import FileTableWidget


class TestMetadataTableDisplay:
    """Integration tests for metadata display in the FileTableWidget."""

    def test_shape_column_header_is_simple_shape(self, mock_file_table_widget, mock_file_item):
        """Shape column header should be 'Shape', not 'Shape (CYX)'."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        header_labels = [widget.horizontalHeaderItem(col).text() for col in range(widget.columnCount())]
        assert "Shape" in header_labels
        assert "Shape (CYX)" not in header_labels

    def test_shape_column_shows_axis_value_format(self, mock_file_table_widget):
        """Shape column should show axis=value format like 'C=3, Y=10000, X=10000'."""
        widget = mock_file_table_widget

        file_item = FileItem(
            path="/test/image.tif",
            status=FileStatus.RAW,
            metadata=MetaData(axes="CYX")
        )
        file_item.shape = (3, 10000, 10000)
        file_item.original_shape = (3, 10000, 10000)
        file_item.dtype = "uint16"

        widget.add_file_item(file_item)

        shape_text = widget.item(0, 2).text()
        assert shape_text == "C=3, Y=10000, X=10000"

    def test_shape_column_with_different_axes(self, mock_file_table_widget):
        """Shape column should handle different axis orders."""
        widget = mock_file_table_widget

        file_item = FileItem(
            path="/test/image.tif",
            status=FileStatus.RAW,
            metadata=MetaData(axes="YX")
        )
        file_item.shape = (2048, 2048)
        file_item.original_shape = (2048, 2048)
        file_item.dtype = "uint16"

        widget.add_file_item(file_item)

        shape_text = widget.item(0, 2).text()
        assert shape_text == "Y=2048, X=2048"

    def test_metadata_columns_exist(self, mock_file_table_widget, mock_file_item):
        """Table should have columns for PhysicalSizeX, PhysicalSizeY, and AlignmentChannel."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        header_labels = [widget.horizontalHeaderItem(col).text() for col in range(widget.columnCount())]
        assert "PhysicalSizeX" in header_labels
        assert "PhysicalSizeY" in header_labels
        assert "AlignmentChannel" in header_labels

    def test_physicalsize_x_displayed(self, mock_file_table_widget):
        """PhysicalSizeX column should show the metadata value."""
        widget = mock_file_table_widget

        file_item = FileItem(
            path="/test/image.tif",
            status=FileStatus.RAW,
            metadata=MetaData(PhysicalSizeX=0.5)
        )
        file_item.shape = (1, 512, 512)
        file_item.original_shape = (1, 512, 512)
        file_item.dtype = "uint16"

        widget.add_file_item(file_item)

        physical_size_x_col = None
        for col in range(widget.columnCount()):
            if widget.horizontalHeaderItem(col).text() == "PhysicalSizeX":
                physical_size_x_col = col
                break

        assert physical_size_x_col is not None
        assert widget.item(0, physical_size_x_col).text() == "0.5"

    def test_physicalsize_y_displayed(self, mock_file_table_widget):
        """PhysicalSizeY column should show the metadata value."""
        widget = mock_file_table_widget

        file_item = FileItem(
            path="/test/image.tif",
            status=FileStatus.RAW,
            metadata=MetaData(PhysicalSizeY=0.75)
        )
        file_item.shape = (1, 512, 512)
        file_item.original_shape = (1, 512, 512)
        file_item.dtype = "uint16"

        widget.add_file_item(file_item)

        physical_size_y_col = None
        for col in range(widget.columnCount()):
            if widget.horizontalHeaderItem(col).text() == "PhysicalSizeY":
                physical_size_y_col = col
                break

        assert physical_size_y_col is not None
        assert widget.item(0, physical_size_y_col).text() == "0.75"

    def test_alignment_channel_displayed(self, mock_file_table_widget):
        """AlignmentChannel column should show the reference_channel metadata value."""
        widget = mock_file_table_widget

        file_item = FileItem(
            path="/test/image.tif",
            status=FileStatus.RAW,
            metadata=MetaData(reference_channel=2)
        )
        file_item.shape = (3, 512, 512)
        file_item.original_shape = (3, 512, 512)
        file_item.dtype = "uint16"

        widget.add_file_item(file_item)

        alignment_channel_col = None
        for col in range(widget.columnCount()):
            if widget.horizontalHeaderItem(col).text() == "AlignmentChannel":
                alignment_channel_col = col
                break

        assert alignment_channel_col is not None
        assert widget.item(0, alignment_channel_col).text() == "2"

    def test_update_file_display_updates_all_metadata_columns(
        self, mock_file_table_widget, mock_file_item
    ):
        """update_file_display should update all metadata columns."""
        widget = mock_file_table_widget

        widget.add_file_item(mock_file_item)

        updated_item = FileItem(
            path=mock_file_item.path,
            status=FileStatus.ALIGNED,
            metadata=MetaData(
                axes="CYX",
                PhysicalSizeX=0.5,
                PhysicalSizeY=0.5,
                reference_channel=1
            )
        )
        updated_item.shape = (3, 1024, 1024)
        updated_item.original_shape = (3, 1024, 1024)
        updated_item.dtype = "uint16"

        widget.update_file_display([updated_item])

        shape_text = widget.item(0, 2).text()
        assert shape_text == "C=3, Y=1024, X=1024"

        physical_size_x_col = None
        physical_size_y_col = None
        alignment_channel_col = None
        for col in range(widget.columnCount()):
            header_text = widget.horizontalHeaderItem(col).text()
            if header_text == "PhysicalSizeX":
                physical_size_x_col = col
            elif header_text == "PhysicalSizeY":
                physical_size_y_col = col
            elif header_text == "AlignmentChannel":
                alignment_channel_col = col

        assert widget.item(0, physical_size_x_col).text() == "0.5"
        assert widget.item(0, physical_size_y_col).text() == "0.5"
        assert widget.item(0, alignment_channel_col).text() == "1"
