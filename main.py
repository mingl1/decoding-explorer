"""
Main script for starting the MIST-Explorer application.
"""

import io
import sys

if sys.stdout is None:
    sys.stdout = io.StringIO()

if sys.stderr is None:
    sys.stderr = io.StringIO()
import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore

from PyQt6.QtWidgets import QApplication

from view.MainWindow import MainWindow

# Prevent matplotlib cache building after compiling data


if __name__ == "__main__":
    __app = QApplication(sys.argv)
    window = MainWindow()

    window.show()

    sys.exit(__app.exec())
