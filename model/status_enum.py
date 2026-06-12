from enum import Enum


class FileStatus(Enum):
    RAW = "Raw"
    SHADE_CORRECTED = "Corrected"
    ALIGNED = "Aligned"
    MANUALLY_ALIGNED = "Manually Aligned"
    AUTO_ALIGNED = "Auto Aligned"
    BEADS_GENERATED = "Generated"
    CROPPED = "Cropped"

    def __str__(self):
        return f"{self.name}: {self.value}"

    @property
    def color(self):
        return {
            FileStatus.RAW: "#A0A0A0",
            FileStatus.SHADE_CORRECTED: "#FFD700",
            FileStatus.ALIGNED: "#FF8C00",
            FileStatus.MANUALLY_ALIGNED: "#FF8C00",
            FileStatus.AUTO_ALIGNED: "#FF8C00",
            FileStatus.BEADS_GENERATED: "#32CD32",
            FileStatus.CROPPED: "#1E90FF",
        }[self]
