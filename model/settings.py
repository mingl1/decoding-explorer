# models/settings.py
from dataclasses import dataclass


@dataclass
class Settings:
    pixel_size: float
    max_size: int
    reference_channel: int = 0
