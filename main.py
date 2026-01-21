"""
Main script for starting the MIST-Explorer application.
"""

import io
import sys
import traceback

try:
    import pyi_splash
except ImportError:
    class DummySplash:
        def close(self):
            pass
    pyi_splash = DummySplash()
from PyQt6.QtWidgets import (QApplication, QDialog, QLabel, QPushButton,
                             QTextEdit, QVBoxLayout)

if sys.stdout is None:
    sys.stdout = io.StringIO()

if sys.stderr is None:
    sys.stderr = io.StringIO()
import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore

from view.MainWindow import MainWindow


def show_error_dialog(etype, value, tb):
    dialog = QDialog()
    dialog.setWindowTitle("Unhandled Exception")
    layout = QVBoxLayout()

    # Exception details
    error_message = f"An unhandled exception occurred: {value}"
    error_label = QLabel(error_message)
    layout.addWidget(error_label)

    # Traceback
    tb_text = "".join(traceback.format_exception(etype, value, tb))
    tb_edit = QTextEdit()
    tb_edit.setPlainText(tb_text)
    tb_edit.setReadOnly(True)
    layout.addWidget(tb_edit)

    # OK button
    ok_button = QPushButton("OK")
    ok_button.clicked.connect(dialog.accept)
    layout.addWidget(ok_button)

    dialog.setLayout(layout)
    dialog.exec()


sys.excepthook = show_error_dialog


# Prevent matplotlib cache building after compiling data

if __name__ == "__main__":
    __app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    pyi_splash.close()
    sys.exit(__app.exec())
