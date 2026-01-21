# models/file_item.py

from dataclasses import dataclass, field
from typing import Optional

from numpy.typing import NDArray
from pandas import DataFrame, Series

from model.status_enum import FileStatus


@dataclass
class MetaData:
    axes: str = "CYX"
    unit: str = "um"
    PhysicalSizeX: float = 0.3250
    PhysicalSizeY: float = 0.3250
    max_size: int = 10000
    reference_channel: int = 0
    prefix: str = "raw"
    use_status_as_prefix: bool = True
    overlap: int = 250
    num_tiles: int = 10
    flors_layers: list[int] | None = None  # filled in later
    crop_bounds: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)


@dataclass
class FileItem:
    path: str
    metadata: MetaData = field(default_factory=MetaData)
    status: FileStatus = FileStatus.RAW
    working_image: Optional[NDArray] = None
    shape: tuple = ()
    original_shape: tuple = ()
    dtype: str = ""
    beads: Optional[DataFrame] = None
    cycles: Optional[dict[str, NDArray]] = None
    bboxs: Optional[Series] = None
    labeled_image: Optional[NDArray] = None
    reference_item: Optional["FileItem"] = None
    protein_profile: Optional[DataFrame] = None
    cycle_files: Optional[dict[int, "FileItem"]] = (
        None  # mapping cycle number to file item to retrieve brightfield images
    )
    threshold: float = 0.7
