import numpy as np

import image_processing
from model.file_item import MetaData


class FakeStarDistModel:
    def __init__(self, sequences):
        self._sequences = list(sequences)

    def _guess_n_tiles(self, img):
        return (1, 1)

    def _predict_instances_generator(self, *args, **kwargs):
        if not self._sequences:
            raise AssertionError("No generator sequence configured")
        for item in self._sequences.pop(0):
            yield item


def test_predict_instances_generator_parser_emits_tile_progress():
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    details = {"prob": np.array([0.9, 0.7], dtype=np.float32)}
    model = FakeStarDistModel(
        [["predict", "tile", "tile", "nms", (labels, details)]]
    )
    events = []

    out_labels, out_details, completed_tiles, expected_tiles = (
        image_processing._predict_instances_with_generator(
            model=model,
            img=np.zeros((8, 8), dtype=np.float32),
            n_tiles=2,
            progress_units_callback=lambda s, d, t: events.append((s, d, t)),
            progress_stage="activation_regions",
        )
    )

    assert completed_tiles == 4
    assert expected_tiles == 4
    assert out_labels.shape == labels.shape
    assert np.allclose(out_details["prob"], details["prob"])
    assert events[-1] == ("activation_regions", 4, 4)


def test_get_labels_from_cycles_with_prob_extracts_probabilities():
    lbl1 = np.array([[0, 1], [2, 0]], dtype=np.int32)
    lbl2 = np.array([[0, 1], [1, 0]], dtype=np.int32)
    model = FakeStarDistModel(
        [
            ["predict", "tile", "nms", (lbl1, {"prob": np.array([0.9, 0.6], dtype=np.float32)})],
            ["predict", "tile", "nms", (lbl2, {"prob": np.array([0.8], dtype=np.float32)})],
        ]
    )
    cycle = np.zeros((3, 6, 6), dtype=np.uint16)
    metadata = MetaData(max_size=6, reference_channel=0)
    metadata.flors_layers = [1, 2]
    progress_events = []

    out = image_processing.get_labels_from_cycles_with_prob(
        cycles=[cycle],
        metadata_list=[metadata],
        max_size=6,
        model=model,
        n_tiles=1,
        progress_units_callback=lambda s, d, t: progress_events.append((s, d, t)),
    )

    assert len(out) == 1
    assert len(out[0]) == 2
    assert np.allclose(out[0][0]["prob_lut"], np.array([0.0, 0.9, 0.6], dtype=np.float32))
    assert np.allclose(out[0][1]["prob_lut"], np.array([0.0, 0.8], dtype=np.float32))
    assert progress_events[-1] == ("activation_regions", 2, 2)
