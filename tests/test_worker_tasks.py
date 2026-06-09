import importlib
import sys
import types
from unittest.mock import MagicMock


def load_worker_module(monkeypatch):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    module_name = "hedonism.who_dis.worker"
    if module_name in sys.modules:
        del sys.modules[module_name]

    return importlib.import_module(module_name)


def test_caption_image_updates_caption(monkeypatch):
    worker = load_worker_module(monkeypatch)

    monkeypatch.setattr(worker, "get_photo_url", lambda photo_id: "https://example.com/photo.jpg")

    mock_pipe = MagicMock(
        return_value=[
            {
                "generated_text": [
                    {"content": "ignored"},
                    {"content": "ignored"},
                    {"content": "A professional generated caption"},
                ]
            }
        ]
    )
    monkeypatch.setattr(worker, "pipe", mock_pipe)

    client = MagicMock()
    graph_client = MagicMock(return_value=client)
    monkeypatch.setattr(worker, "graph_client", graph_client)

    worker.caption_image("photo-1")

    assert mock_pipe.call_count == 1
    assert client.execute.call_count == 1
    mutation = client.execute.call_args[0][0]
    assert mutation.variable_values == {
        "photoId": "photo-1",
        "caption": "A professional generated caption",
    }


def test_extract_facial_data_filters_low_confidence_faces(monkeypatch):
    worker = load_worker_module(monkeypatch)

    monkeypatch.setattr(worker, "get_photo_url", lambda photo_id: "https://example.com/photo.jpg")

    monkeypatch.setattr(
        worker.DeepFace,
        "represent",
        MagicMock(
            return_value=[
                {
                    "embedding": [0.1, 0.2],
                    "facial_area": {"x": 1, "y": 2, "w": 3, "h": 4},
                    "face_confidence": 0.95,
                },
                {
                    "embedding": [0.3, 0.4],
                    "facial_area": {"x": 5, "y": 6, "w": 7, "h": 8},
                    "face_confidence": 0.4,
                },
            ]
        ),
    )

    client = MagicMock()
    monkeypatch.setattr(worker, "graph_client", MagicMock(return_value=client))

    worker.extract_facial_data("photo-2")

    assert client.execute.call_count == 1
    mutation = client.execute.call_args[0][0]
    assert mutation.variable_values == {
        "photoId": "photo-2",
        "faceObjects": [
            {
                "embedding": [0.1, 0.2],
                "facialArea": {"x": 1, "y": 2, "w": 3, "h": 4},
                "faceConfidence": 0.95,
            }
        ],
    }


def test_extract_facial_data_handles_exception(monkeypatch):
    worker = load_worker_module(monkeypatch)

    monkeypatch.setattr(worker, "get_photo_url", lambda photo_id: "https://example.com/photo.jpg")

    monkeypatch.setattr(
        worker.DeepFace,
        "represent",
        MagicMock(side_effect=RuntimeError("deepface failure")),
    )

    client = MagicMock()
    monkeypatch.setattr(worker, "graph_client", MagicMock(return_value=client))

    worker.extract_facial_data("photo-3")

    client.execute.assert_not_called()
