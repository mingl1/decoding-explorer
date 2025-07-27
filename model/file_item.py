# models/file_item.py
from dataclasses import dataclass

from model.status_enum import FileStatus


@dataclass
class FileItem:
    path: str
    metadata: dict
    status: FileStatus = FileStatus.RAW
    aligned_image: object = None
    reference_channel: int = 0  # also serves as shading correction channel
    max_size: int = 10000
    shape: tuple = ()
    dtype: str = ""
