"""In-memory YOLO tracking with lightweight JSON/CSV analytics.

The script displays an annotated frame when requested but never writes an
annotated video. Only compact detection statistics are exported.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MODEL = "yolov8n.pt"


@dataclass
class TrackSummary:
    track_id: int
    label: str
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    max_confidence: float = 0.0
    last_bbox: list[float] = field(default_factory=list)


@dataclass
class Analytics:
    source: str
    model: str
    started_at_unix: float = field(default_factory=time.time)
    frames_processed: int = 0
    max_objects_in_frame: int = 0
    class_totals: Counter[str] = field(default_factory=Counter)
    unique_track_ids: set[int] = field(default_factory=set)
    tracks: dict[int, TrackSummary] = field(default_factory=dict)
    frame_counts: list[dict[str, Any]] = field(default_factory=list)

    def record(self, frame_number: int, detections: Iterable[dict[str, Any]]) -> None:
        detections = list(detections)
        self.frames_processed += 1
        by_class = Counter(str(item["label"]) for item in detections)
        self.class_totals.update(by_class)
        self.max_objects_in_frame = max(self.max_objects_in_frame, len(detections))
        self.frame_counts.append({"frame": frame_number, "total": len(detections), "classes": dict(by_class)})
        for item in detections:
            track_id = item.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            self.unique_track_ids.add(track_id)
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackSummary(
                    track_id=track_id,
                    label=str(item["label"]),
                    first_frame=frame_number,
                    last_frame=frame_number,
                )
            summary = self.tracks[track_id]
            summary.last_frame = frame_number
            summary.frames_seen += 1
            summary.max_confidence = max(summary.max_confidence, float(item["confidence"]))
            summary.last_bbox = [round(float(v), 2) for v in item["bbox"]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "frames_processed": self.frames_processed,
            "max_objects_in_frame": self.max_objects_in_frame,
            "class_totals": dict(self.class_totals),
            "unique_track_ids": len(self.unique_track_ids),
            "tracks": [asdict(item) for item in sorted(self.tracks.values(), key=lambda x: x.track_id)],
            "frame_counts": self.frame_counts,
        }


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def resolve_class_ids(model_names: dict[int, str], requested: list[str] | None) -> list[int] | None:
    if not requested:
        return None
    by_name = {name.lower(): index for index, name in model_names.items()}
    ids: list[int] = []
    for item in requested:
        if item.isdigit() and int(item) in model_names:
            ids.append(int(item))
        elif item.lower() in by_name:
            ids.append(by_name[item.lower()])
        else:
            available = ", ".join(sorted(by_name))
            raise ValueError(f"Unknown class {item!r}. Available model labels include: {available}")
    return ids


def extract_detections(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().tolist()
    cls_values = boxes.cls.cpu().tolist()
    conf_values = boxes.conf.cpu().tolist()
    id_tensor = getattr(boxes, "id", None)
    ids = id_tensor.cpu().tolist() if id_tensor is not None else [None] * len(xyxy)
    return [
        {
            "track_id": int(track_id) if track_id is not None else None,
            "label": names.get(int(class_id), str(int(class_id))),
            "confidence": round(float(confidence), 4),
            "bbox": [round(float(value), 2) for value in bbox],
        }
        for bbox, class_id, confidence, track_id in zip(xyxy, cls_values, conf_values, ids)
    ]


def write_outputs(analytics: Analytics, json_path: Path | None, csv_path: Path | None) -> None:
    if json_path:
        json_path.write_text(json.dumps(analytics.to_dict(), indent=2) + "\n", encoding="utf-8")
    if csv_path:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["track_id", "label", "first_frame", "last_frame", "frames_seen", "max_confidence", "last_bbox"])
            writer.writeheader()
            for item in sorted(analytics.tracks.values(), key=lambda x: x.track_id):
                row = asdict(item)
                row["last_bbox"] = json.dumps(row["last_bbox"])
                writer.writerow(row)


def run_tracking(source: str | int, model_name: str, display: bool, json_path: Path | None, csv_path: Path | None, requested_classes: list[str] | None = None) -> Analytics:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: py -m pip install -r requirements.txt") from exc

    model = YOLO(model_name)
    names = normalize_names(model.names)
    class_ids = resolve_class_ids(names, requested_classes)
    analytics = Analytics(source=str(source), model=model_name)
    try:
        results = model.track(source=source, stream=True, persist=True, classes=class_ids, verbose=False)
        for frame_number, result in enumerate(results, start=1):
            detections = extract_detections(result, names)
            analytics.record(frame_number, detections)
            counts = Counter(item["label"] for item in detections)
            count_text = " ".join(f"{label}={count}" for label, count in sorted(counts.items())) or "none"
            print(f"Frame {frame_number} | {count_text} | total={len(detections)}")
            if display:
                annotated = result.plot()
                cv2.imshow("Day 6 - YOLOv8 tracking (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if display:
            cv2.destroyAllWindows()
    write_outputs(analytics, json_path, csv_path)
    return analytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track objects in a video/webcam stream without writing an annotated video.")
    parser.add_argument("--source", required=True, help="Video path, URL supported by OpenCV, or webcam index such as 0")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics model, default: yolov8n.pt")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV window")
    parser.add_argument("--classes", nargs="+", help="Optional class names or IDs, for example car bus truck person")
    parser.add_argument("--json-out", type=Path, default=Path("tracking_stats.json"), help="Compact summary output path")
    parser.add_argument("--csv-out", type=Path, default=Path("track_summaries.csv"), help="Per-track bounding-box summary path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        analytics = run_tracking(parse_source(args.source), args.model, not args.no_display, args.json_out, args.csv_out, args.classes)
        print(f"Completed {analytics.frames_processed} frames; unique track IDs={len(analytics.unique_track_ids)}")
        print(f"Wrote statistics: {args.json_out}")
        print(f"Wrote track summaries: {args.csv_out}")
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
