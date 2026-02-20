# AGENTS.md

This file provides guidance for agentic coding assistants working in this repository.

## Build, Lint, and Test Commands

### Running the Application
```bash
python main.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Building Executable
```bash
pip install pyinstaller
pyinstaller main.py --onefile --windowed --name "Decoding-Explorer" --add-data "assets:assets"
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_file_manager_vm.py

# Run single test
pytest tests/unit/test_file_manager_vm.py::test_function_name

# Run by marker
pytest -m unit
pytest -m integration
pytest -m e2e

# Run with verbose output
pytest -v

# Generate coverage report
pytest --cov=. --cov-report=html
```

### Python Version
- Python 3.10+ required (see `.python-version`)

---

## Code Style Guidelines

### Imports
- Use absolute imports: `from package.module import Class`
- Group imports in this order: stdlib, third-party, local application
- Sort imports alphabetically within each group
- Use type hint imports from `typing` module

```python
import os
import sys
from pathlib import Path
from typing import Optional, list, dict

import numpy as np
from PyQt6.QtWidgets import QMainWindow

from model.file_item import FileItem
```

### Dataclasses
- Use `@dataclass` for model classes with default values
- Use `field(default_factory=...)` for mutable defaults (lists, dicts)
- Place dataclasses in `/model/` directory

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MetaData:
    max_size: int = 10000
    reference_channel: int = 0
    flors_layers: list[int] | None = None
```

### Type Hints
- Use Python 3.10+ union syntax: `int | str` instead of `Union[int, str]`
- Use `Optional[X]` for nullable types (not `X | None` for consistency)
- Use `numpy.typing.NDArray` for numpy arrays
- Use concrete types for collections: `list[str]`, `dict[str, int]`

### Naming Conventions
- **Classes**: PascalCase (`FileItem`, `MetadataView`)
- **Functions/methods**: snake_case (`load_image`, `apply_shading`)
- **Variables**: snake_case (`max_size`, `bright_field`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_ZOOM`, `DEFAULT_CHANNEL`)
- **Private members**: leading underscore (`_internal_method`, `_cache`)

### File Organization
- **Model** (`/model/`): Data structures, dataclasses, enums
- **View** (`/view/`): PyQt6 UI components, dialogs, widgets
- **ViewModel** (`/viewmodel/`): Business logic, state management, signals
- **Root**: Utility modules (`utils.py`, `image_processing.py`, `align_arrays.py`)

### MVVM Architecture
- Views emit signals; ViewModels connect to them
- ViewModels own Models and expose data via signals
- Views never import ViewModels directly (use signals/slots)
- Use `pyqtSignal` for inter-component communication

### Error Handling
- Use custom error messages that help users understand what went wrong
- Emit signals for errors in background threads (`self.align_error.emit(msg)`)
- Show QMessageBox for user-facing errors
- Use `try/except` with specific exception types, not bare `except:`

```python
try:
    result = risky_operation()
except ValueError as e:
    self.align_error.emit(f"Invalid value: {e}")
    return None
```

### Threading
- Long-running operations must use QThread
- Emit progress signals: `self.progress.emit(percentage, message)`
- Check `_is_running` flag for cancellation support
- Never directly modify UI from background threads

### PyQt6 Patterns
- Use `pyqtSignal` and `pyqtSlot` for type-safe signals
- Connect signals with `.connect()` and disconnect when cleaning up
- Use `QTimer.singleShot(0, callback)` for deferred execution
- Override event handlers with proper signature (`mousePressEvent(self, event)`)

### No Comments (Per Project Convention)
- Do not add comments unless explicitly requested
- Code should be self-documenting
- Complex logic should be extracted into well-named functions

### Array Slicing
- Always apply `max_size` constraint when loading images for dialogs:
```python
max_size = int(file_item.metadata.max_size)
return image[:max_size, :max_size]
```

### Image Processing
- Use memory mapping for large images: `tifffile.memmap(path, shape, mode="r")`
- Process channels as first dimension: `image[channel_index, y, x]`
- Use `to_uint8()` from `utils` for display conversion
