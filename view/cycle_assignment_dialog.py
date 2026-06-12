import os
import re
from typing import Optional

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
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

        if initial_assignments is None or initial_protein_file is None:
            (
                suggested_assignments,
                suggested_protein_file,
            ) = self._suggest_initial_selection()
            if initial_assignments is None:
                initial_assignments = suggested_assignments
            if initial_protein_file is None:
                initial_protein_file = suggested_protein_file

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

    def _suggest_initial_selection(
        self,
    ) -> tuple[dict[int, FileItem], Optional[FileItem]]:
        if len(self.files) == 0:
            return {}, None

        protein_file = None
        non_protein_files: list[FileItem] = []
        for file_item in self.files:
            normalized_name = self._normalize_name(file_item.path)
            if protein_file is None and "protein" in normalized_name:
                protein_file = file_item
                continue
            non_protein_files.append(file_item)

        if len(non_protein_files) == 0:
            return {}, protein_file

        parsed_cycles: dict[str, int] = {}
        max_named_cycle = 0
        for file_item in non_protein_files:
            cycle_number = self._extract_cycle_number(file_item.path)
            if cycle_number is None:
                continue
            parsed_cycles[file_item.path] = cycle_number
            if cycle_number > max_named_cycle:
                max_named_cycle = cycle_number

        cycle_slots = max(self.default_cycle_count, max_named_cycle)
        cycle_slots = min(len(non_protein_files), cycle_slots)
        cycle_slots = max(1, cycle_slots)

        assignments: dict[int, FileItem] = {}
        assigned_paths = set()
        for file_item in non_protein_files:
            cycle_number = parsed_cycles.get(file_item.path)
            if cycle_number is None:
                continue
            if cycle_number < 1 or cycle_number > cycle_slots:
                continue
            if cycle_number in assignments:
                continue
            assignments[cycle_number] = file_item
            assigned_paths.add(file_item.path)

        remaining = [
            file_item
            for file_item in non_protein_files
            if file_item.path not in assigned_paths
        ]
        for cycle_num in range(1, cycle_slots + 1):
            if cycle_num in assignments:
                continue
            if len(remaining) == 0:
                break
            assignments[cycle_num] = remaining.pop(0)

        return assignments, protein_file

    def _normalize_name(self, file_path: str) -> str:
        base_name = os.path.basename(file_path).lower()
        return re.sub(r"[^a-z0-9]", "", base_name)

    def _extract_cycle_number(self, file_path: str) -> Optional[int]:
        normalized_name = self._normalize_name(file_path)
        patterns = [
            r"cycle(\d+)",
            r"cy(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_name)
            if match is not None:
                return int(match.group(1))
        return None

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

        if len(assignments) != len(self.comboboxes):
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
