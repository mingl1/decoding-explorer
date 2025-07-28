# models/file_item.py
from dataclasses import dataclass
from sys import prefix

from numpy.typing import NDArray
from pandas import DataFrame

from model.status_enum import FileStatus


@dataclass
class MetaData:
    axes: str = "CYX"
    unit: str = "um"
    PhysicalSizeX: float = 0.3250
    PhysicalSizeY: float = 0.3250
    max_size: int = 10000
    reference_channel: int = 0
    prefix: str = "changed_"
    overlap: int = 250
    num_tiles: int = 10
    flors_layers: list[int] | None = None  # filled in later


@dataclass
class FileItem:
    path: str
    metadata: MetaData = MetaData()
    status: FileStatus = FileStatus.RAW
    working_image: NDArray | None = None
    shape: tuple = ()
    dtype: str = ""
    beads: DataFrame | None = None
