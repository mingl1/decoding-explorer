import numpy as np

import image_processing
from model.file_item import MetaData


class FakeStarDistModel:
    def __init__(
        self,
        *,
        instances_outputs=None,
        raise_on_predict=False,
        fallback_sequences=None,
    ):
        self._instances_outputs = list(instances_outputs or [])
        self._raise_on_predict = bool(raise_on_predict)
        self._fallback_sequences = list(fallback_sequences or [])
        self.predict_calls = []
        self.instances_calls = []

    def _guess_n_tiles(self, img):
        return (1, 1)

    def predict(self, img, axes=None, n_tiles=None, show_tile_progress=None):
        self.predict_calls.append(
            {
                "img_shape": img.shape,
                "axes": axes,
                "n_tiles": n_tiles,
            }
        )
        if self._raise_on_predict:
            raise MemoryError("simulated dense predict OOM")
        total_tiles = int(np.prod(n_tiles)) if n_tiles is not None else 1
        if callable(show_tile_progress) and total_tiles > 1:
            for _ in show_tile_progress(range(total_tiles), total=total_tiles):
                pass
        prob = np.ones(img.shape, dtype=np.float32)
        dist = np.ones(img.shape + (4,), dtype=np.float32)
        return prob, dist

    def _instances_from_prediction(
        self,
        img_shape,
        prob,
        dist,
        prob_class=None,
        prob_thresh=None,
        nms_thresh=None,
        scale=None,
    ):
        self.instances_calls.append(
            {
                "img_shape": tuple(img_shape),
                "prob_shape": prob.shape,
                "dist_shape": dist.shape,
                "scale": scale,
                "prob_thresh": prob_thresh,
                "nms_thresh": nms_thresh,
            }
        )
        if self._instances_outputs:
            return self._instances_outputs.pop(0)
        labels = np.zeros(img_shape, dtype=np.int32)
        details = {"prob": np.array([], dtype=np.float32)}
        return labels, details

    def _predict_instances_generator(self, *args, **kwargs):
        if not self._fallback_sequences:
            raise AssertionError("No generator sequence configured")
        for item in self._fallback_sequences.pop(0):
            yield item


def test_predict_instances_dense_split_emits_tile_progress():
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    details = {"prob": np.array([0.9, 0.7], dtype=np.float32)}
    model = FakeStarDistModel(instances_outputs=[(labels, details)])
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
    assert model.predict_calls[0]["n_tiles"] == (2, 2)


def test_predict_instances_dense_split_emits_nms_message():
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    details = {"prob": np.array([0.9, 0.7], dtype=np.float32)}
    model = FakeStarDistModel(instances_outputs=[(labels, details)])
    messages = []

    image_processing._predict_instances_with_generator(
        model=model,
        img=np.zeros((8, 8), dtype=np.float32),
        n_tiles=2,
        progress_message_callback=messages.append,
        nms_message="Running NMS",
    )

    assert messages == ["Running NMS"]


def test_predict_instances_dense_split_forwards_scale_and_rescales_input():
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    details = {"prob": np.array([0.9, 0.7], dtype=np.float32)}
    model = FakeStarDistModel(instances_outputs=[(labels, details)])

    image_processing._predict_instances_with_generator(
        model=model,
        img=np.zeros((8, 8), dtype=np.float32),
        n_tiles=1,
        scale=2,
    )

    assert model.predict_calls[0]["img_shape"] == (16, 16)
    assert model.instances_calls[0]["img_shape"] == (8, 8)
    assert model.instances_calls[0]["scale"] == {"Y": 2.0, "X": 2.0}


def test_predict_instances_falls_back_to_sparse_generator_on_memory_error():
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    details = {"prob": np.array([0.9, 0.7], dtype=np.float32)}
    model = FakeStarDistModel(
        raise_on_predict=True,
        fallback_sequences=[["predict", "tile", "tile", "nms", (labels, details)]],
    )
    events = []
    messages = []

    out_labels, out_details, completed_tiles, expected_tiles = (
        image_processing._predict_instances_with_generator(
            model=model,
            img=np.zeros((8, 8), dtype=np.float32),
            n_tiles=2,
            progress_units_callback=lambda s, d, t: events.append((s, d, t)),
            progress_stage="activation_regions",
            progress_message_callback=messages.append,
            nms_message="Running NMS",
        )
    )

    assert completed_tiles == 4
    assert expected_tiles == 4
    assert out_labels.shape == labels.shape
    assert np.allclose(out_details["prob"], details["prob"])
    assert events[-1] == ("activation_regions", 4, 4)
    assert messages == ["Running NMS"]


def test_get_labels_from_cycles_with_prob_extracts_probabilities():
    lbl1 = np.array([[0, 1], [2, 0]], dtype=np.int32)
    lbl2 = np.array([[0, 1], [1, 0]], dtype=np.int32)
    model = FakeStarDistModel(
        instances_outputs=[
            (lbl1, {"prob": np.array([0.9, 0.6], dtype=np.float32)}),
            (lbl2, {"prob": np.array([0.8], dtype=np.float32)}),
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


def test_get_labels_from_cycles_with_prob_emits_channel_messages_without_nms_text():
    lbl = np.array([[0, 1], [1, 0]], dtype=np.int32)
    model = FakeStarDistModel(
        instances_outputs=[
            (lbl, {"prob": np.array([0.7], dtype=np.float32)}),
            (lbl, {"prob": np.array([0.8], dtype=np.float32)}),
        ]
    )
    cycle = np.zeros((3, 6, 6), dtype=np.uint16)
    metadata = MetaData(max_size=6, reference_channel=0)
    metadata.flors_layers = [1, 2]
    messages = []

    image_processing.get_labels_from_cycles_with_prob(
        cycles=[cycle],
        metadata_list=[metadata],
        max_size=6,
        model=model,
        n_tiles=1,
        progress_callback=messages.append,
    )

    assert messages == [
        "Processing fluorescence channels (1/2)",
        "Processing fluorescence channels (2/2)",
    ]
