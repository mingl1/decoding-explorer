from enum import Enum


class FileStatus(Enum):
    RAW = "Raw"
    SHADE_CORRECTED = "Shade Corrected"
    ALIGNED = "Aligned"
    BEADS_GENERATED = "Protein Generated"
    REFERENCE = "Reference"

    _STEP_ORDER = [RAW, SHADE_CORRECTED, ALIGNED, BEADS_GENERATED]

    def __str__(self):
        return f"{self.name}: {self.value}"

    @property
    def color(self):
        return {
            FileStatus.RAW: "#A0A0A0",  # medium gray
            FileStatus.SHADE_CORRECTED: "#FFD700",  # gold (better than plain yellow)
            FileStatus.ALIGNED: "#FF8C00",  # dark orange
            FileStatus.BEADS_GENERATED: "#32CD32",  # lime green
            FileStatus.REFERENCE: "#1E90FF",  # dodger blue
        }[self]
