from collections import Counter

from tracker import Analytics, normalize_names, parse_source, resolve_class_ids


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("video.mp4") == "video.mp4"


def test_resolve_class_ids_by_name_and_id():
    names = {0: "person", 2: "car", 5: "bus"}
    assert resolve_class_ids(names, ["car", "bus"]) == [2, 5]
    assert resolve_class_ids(names, ["0"]) == [0]
    assert resolve_class_ids(names, None) is None


def test_resolve_unknown_class():
    try:
        resolve_class_ids({0: "person"}, ["airplane"])
    except ValueError as exc:
        assert "Unknown class" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_analytics_records_frames_and_track_summary():
    analytics = Analytics(source="sample.mp4", model="yolov8n.pt")
    analytics.record(1, [{"track_id": 7, "label": "car", "confidence": 0.8, "bbox": [1, 2, 30, 40]}])
    analytics.record(2, [
        {"track_id": 7, "label": "car", "confidence": 0.9, "bbox": [2, 3, 31, 41]},
        {"track_id": 8, "label": "person", "confidence": 0.7, "bbox": [5, 6, 20, 25]},
    ])
    result = analytics.to_dict()
    assert result["frames_processed"] == 2
    assert result["max_objects_in_frame"] == 2
    assert result["class_totals"] == {"car": 2, "person": 1}
    assert result["unique_track_ids"] == 2
    assert result["tracks"][0]["frames_seen"] == 2
    assert result["tracks"][0]["last_bbox"] == [2.0, 3.0, 31.0, 41.0]
