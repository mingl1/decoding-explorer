import os

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from model.file_item import FileItem


class CycleAssignmentWidget(QDialog):
    def __init__(
        self, files: list[FileItem], parent=None, zero_indexed=False, start_cycle=1
    ):
        super().__init__(parent)
        self.setWindowTitle("Assign Cycles")
        self.files = files
        self.file_map = {os.path.basename(f.path): f for f in files}
        self.layout = QVBoxLayout(self)

        num_cycles = len(files)
        self.table = QTableWidget(num_cycles, 2)
        self.table.setHorizontalHeaderLabels(["Cycle", "File"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.comboboxes = []
        file_names = ["Select a file"] + list(self.file_map.keys())

        for i in range(num_cycles):
            self.table.setItem(
                i,
                0,
                QTableWidgetItem(
                    f"Cycle {i + (0 if zero_indexed else 1) + start_cycle}"
                ),
            )
            combo = QComboBox()
            combo.addItems(file_names)
            self.table.setCellWidget(i, 1, combo)
            self.comboboxes.append(combo)

        self.layout.addWidget(self.table)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def get_assignments(self):
        assignments = {}
        selected_files = set()
        for i, combo in enumerate(self.comboboxes):
            selected_file_name = combo.currentText()
            if selected_file_name != "Select a file":
                if selected_file_name in selected_files:
                    # Duplicate assignment
                    return None  # Indicate error
                selected_files.add(selected_file_name)

                file_item = self.file_map[selected_file_name]
                cycle_num = i + 1
                assignments[cycle_num] = file_item

        if len(assignments) != len(self.files):
            # Not all files assigned
            return None  # Indicate error

        return assignments
