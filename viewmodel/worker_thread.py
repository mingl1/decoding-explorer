import logging
import threading

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class WorkerThread(QThread):
    """Generic QThread that drives a generator task.

    The task is a generator that yields (int, str) progress tuples and
    returns its result via `return value` (caught as StopIteration.value).
    Cancellation is signalled by setting the stop_event passed to the task.
    Unhandled exceptions emit `failed(str(e))` instead of crashing the thread.
    """

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task, stop_event: threading.Event):
        super().__init__()
        self._task = task
        self._stop = stop_event

    def run(self):
        try:
            result = None
            try:
                while True:
                    pct, msg = next(self._task)
                    self.progress.emit(int(pct), str(msg))
            except StopIteration as e:
                result = e.value
            self.completed.emit(result)
        except Exception as e:
            logger.error(f"WorkerThread error: {e}", exc_info=True)
            self.failed.emit(str(e))

    def cancel(self):
        self._stop.set()
