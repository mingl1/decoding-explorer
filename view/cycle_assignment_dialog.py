import os
from typing import Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QAbstractItemView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from model.file_item import FileItem


class CycleAssignmentDialog(QDialog):
    def __init__(
        self,
        files: list[FileItem],
        parent=None,
        zero_indexed=False,
        start_cycle=1,
        cycle_numbers: Optional[list[int]] = None,
        initial_assignments: Optional[dict[int, FileItem]] = None,
        initial_protein_file: Optional[FileItem] = None,
        default_cycle_count: int = 2,
    ):
        super().__init__(parent)
        self.setWindowTitle("Assign Cycles")
        self.files = files
        self.default_cycle_count = max(1, int(default_cycle_count))
        self._option_to_file: dict[str, FileItem] = {}
        self._path_to_option: dict[str, str] = {}
        self._file_options = self._build_file_options(files)
        self.layout = QVBoxLayout(self)

        if cycle_numbers is None:
            max_initial_cycle = 0
            if initial_assignments:
                max_initial_cycle = max(initial_assignments.keys())
            num_cycles = max(self.default_cycle_count, max_initial_cycle)
            num_cycles = min(max(1, len(files)), num_cycles)
        else:
            cycle_list = [int(cycle_num) for cycle_num in cycle_numbers]
            num_cycles = max(1, len(cycle_list))

        self.table = QTableWidget(num_cycles, 2)
        self.table.setHorizontalHeaderLabels(["Cycle", "File"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.comboboxes = []
        self._rebuild_table_rows(
            row_count=num_cycles,
            initial_assignments=initial_assignments,
        )

        self.layout.addWidget(self.table)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.assign_all_button = QPushButton("Assign All Files")
        self.assign_all_button.clicked.connect(self._assign_all_files)
        self.delete_cycle_button = QPushButton("Delete Selected Cycle")
        self.delete_cycle_button.clicked.connect(self._delete_selected_cycle_row)
        button_layout.addWidget(self.assign_all_button)
        button_layout.addWidget(self.delete_cycle_button)
        self.layout.addWidget(button_row)

        protein_row = QWidget()
        protein_layout = QHBoxLayout(protein_row)
        protein_layout.setContentsMargins(0, 0, 0, 0)
        protein_layout.addWidget(QLabel("Protein File (align only):"))
        self.protein_combo = QComboBox()
        self.protein_combo.addItem("None")
        self.protein_combo.addItems(self._option_to_file.keys())
        if initial_protein_file is not None:
            selected_option = self._path_to_option.get(initial_protein_file.path)
            if selected_option:
                self.protein_combo.setCurrentText(selected_option)
        protein_layout.addWidget(self.protein_combo)
        self.layout.addWidget(protein_row)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def get_assignments(self):
        assignments = {}
        selected_files = set()
        protein_selection = self.protein_combo.currentText()
        if protein_selection != "None":
            selected_files.add(protein_selection)
        for i, combo in enumerate(self.comboboxes):
            selected_file_name = combo.currentText()
            if selected_file_name != "Select a file":
                if selected_file_name in selected_files:
                    return None
                selected_files.add(selected_file_name)

                file_item = self._option_to_file[selected_file_name]
                cycle_num = i + 1
                assignments[cycle_num] = file_item

        required_cycles = len(self.comboboxes)
        if protein_selection != "None":
            required_cycles -= 1
        if len(assignments) != required_cycles:
            return None

        return assignments

    def get_protein_file(self) -> Optional[FileItem]:
        selected = self.protein_combo.currentText()
        if selected == "None":
            return None
        return self._option_to_file.get(selected)

    def _assign_all_files(self):
        self._rebuild_table_rows(
            row_count=max(1, len(self.files)),
            initial_assignments={
                index + 1: file_item for index, file_item in enumerate(self.files)
            },
        )

    def _delete_selected_cycle_row(self):
        current_row = self.table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "Delete Cycle", "Select a cycle row to delete.")
            return
        if len(self.comboboxes) <= 1:
            QMessageBox.warning(
                self,
                "Delete Cycle",
                "At least one cycle row is required.",
            )
            return
        assignments = self.get_assignments() or {}
        next_assignments = {}
        for cycle_num, file_item in assignments.items():
            if cycle_num == current_row + 1:
                continue
            next_cycle_num = cycle_num
            if cycle_num > current_row + 1:
                next_cycle_num = cycle_num - 1
            next_assignments[next_cycle_num] = file_item
        self._rebuild_table_rows(
            row_count=len(self.comboboxes) - 1,
            initial_assignments=next_assignments,
        )

    def _build_file_options(self, files: list[FileItem]) -> list[str]:
        seen_names: dict[str, int] = {}
        options = []
        for file_item in files:
            base_name = os.path.basename(file_item.path)
            count = seen_names.get(base_name, 0)
            seen_names[base_name] = count + 1
            option_name = base_name if count == 0 else f"{base_name} ({count + 1})"
            self._option_to_file[option_name] = file_item
            self._path_to_option[file_item.path] = option_name
            options.append(option_name)
        return options

    def _rebuild_table_rows(
        self,
        row_count: int,
        initial_assignments: Optional[dict[int, FileItem]] = None,
    ):
        row_count = max(1, int(row_count))
        self.table.setRowCount(row_count)
        self.comboboxes = []
        file_names = ["Select a file"] + self._file_options
        for row in range(row_count):
            self.table.setItem(row, 0, QTableWidgetItem(f"Cycle {row + 1}"))
            combo = QComboBox()
            combo.addItems(file_names)
            if initial_assignments and (row + 1) in initial_assignments:
                assigned_item = initial_assignments[row + 1]
                assigned_option = self._path_to_option.get(assigned_item.path)
                if assigned_option:
                    combo.setCurrentText(assigned_option)
            self.table.setCellWidget(row, 1, combo)
            self.comboboxes.append(combo)


CycleAssignmentWidget = CycleAssignmentDialog
