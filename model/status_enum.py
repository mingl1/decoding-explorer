from enum import Enum


class FileStatus(Enum):
    RAW = "Raw"
    SHADE_CORRECTED = "Shade Corrected"
    ALIGNED = "Aligned"
    PROTEIN_GENERATED = "Protein Generated"
    __STEP_ORDER = [RAW, SHADE_CORRECTED, ALIGNED, PROTEIN_GENERATED]

    def __str__(self):
        return f"STEP {self.__STEP_ORDER.index(self.value) + 1}: {self.value}"

    @property
    def color(self):
        return {
            FileStatus.RAW: "gray",
            FileStatus.SHADE_CORRECTED: "yellow",
            FileStatus.ALIGNED: "orange",
            FileStatus.PROTEIN_GENERATED: "green",
        }[self]
