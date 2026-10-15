"""YOLOv8 face+hand detection + ByteTrack association.

Strategy ("detector-first"):
  * Two single-class YOLOv8n models — one face, one hand — run on every frame.
  * For face: a BYTETracker keeps a face#1 identity that survives a few-frame
    miss via Kalman prediction.
  * For hand: YOLO detections are the source of truth for the right panel;
    a second BYTETracker is fed every frame and its lost_stracks act as
    "fill-in" boxes for frames where YOLO missed a hand. A HandSlotMapper
    keeps left/right hand IDs stable (hand#1 / hand#2) across YOLO dropouts
    and identity swaps.

"""

from __future__ import annotations

import itertools
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker


REPO = Path(__file__).resolve().parent
MODELS = REPO / "models"
FACE_CKPT = MODELS / "face_yolov8n.pt"
HAND_CKPT = MODELS / "hand_yolov8n.pt"


# ByteTracker settings
def make_tracker(track_buffer: int = 75, fps: float = 25.0) -> BYTETracker:
    args = Namespace(
        track_high_thresh=0.25,
        track_low_thresh=0.05,
        new_track_thresh=0.20,
        match_thresh=0.80,
        track_buffer=track_buffer,
        fuse_score=True,
    )
    return BYTETracker(args, frame_rate=int(round(fps)))


@dataclass
class Box:
    """A single detected/tracked box (xyxy) with label, score, and state."""

    xyxy: tuple[float, float, float, float]
    score: float
    label: str           # 'face' | 'hand'
    track_id: int        # logical id used in label string (1, 2, ...)
    state: str           # 'active' | 'predicted'
    lost_age: int = 0    # only meaningful for 'predicted'


@dataclass
class FrameResult:
    """Per-frame YOLO-only vs YOLO + ByteTrack outputs.

    yolo_boxes are the raw YOLOv8 detections, kept independently of the
    tracker (for visualisation only). track_boxes are the slot-mapped active +
    predicted boxes used by the feature extraction.
    """

    frame_idx: int
    yolo_boxes: list[Box]
    track_boxes: list[Box]


# --- internals ---------------------------------------------------------------


class _DetWrap:
    """Minimal adapter so ultralytics' BYTETracker.update() can read xywh/conf/cls.

    Recent ultralytics BYTETracker.update() slices its detections via
    ``results[inds_second]`` for the low-conf rescue step, so we need __len__
    and __getitem__ that return a new _DetWrap.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.xywh = xywh
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.xywh.shape[0])

    def __getitem__(self, idx) -> "_DetWrap":
        return _DetWrap(self.xywh[idx], self.conf[idx], self.cls[idx])


def _boxes_to_tracker(result) -> _DetWrap:
    if result.boxes is None or len(result.boxes) == 0:
        z = np.zeros((0,), dtype=np.float32)
        return _DetWrap(np.zeros((0, 4), dtype=np.float32), z, z)
    boxes = result.boxes
    return _DetWrap(
        boxes.xywh.detach().cpu().numpy(),
        boxes.conf.detach().cpu().numpy(),
        np.zeros((len(boxes),), dtype=np.float32),
    )


def _active_rows_to_items(rows: np.ndarray, label: str) -> list[dict]:
    items = []
    for row in rows:
        x1, y1, x2, y2, track_id, score, _cls, _idx = row.tolist()
        items.append({
            "label": label,
            "track_id": int(track_id),
            "score": float(score),
            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "state": "active",
            "lost_age": 0,
            "raw_track_id": int(track_id),
            "det_index": int(_idx),
        })
    return items


def _yolo_result_to_items(result, label: str, limit: int) -> list[dict]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    order = np.argsort(-conf)[:limit]
    items = []
    for det_index in order:
        items.append({
            "label": label,
            "track_id": None,
            "score": float(conf[det_index]),
            "xyxy": [float(v) for v in xyxy[det_index]],
            "state": "active",
            "lost_age": 0,
            "raw_track_id": None,
            "det_index": int(det_index),
        })
    return items


def _predicted_items(tracker: BYTETracker, label: str, max_pred_draw: int) -> list[dict]:
    items = []
    for track in tracker.lost_stracks:
        lost_age = tracker.frame_id - track.end_frame
        if lost_age <= max_pred_draw:
            items.append({
                "label": label,
                "track_id": int(track.track_id),
                "score": float(track.score),
                "xyxy": [float(v) for v in track.xyxy],
                "state": "predicted",
                "lost_age": int(lost_age),
                "raw_track_id": int(track.track_id),
            })
    return items


def _choose_tracks(items: list[dict], limit: int) -> list[dict]:
    active = [item for item in items if item["state"] == "active"]
    lost = [item for item in items if item["state"] == "predicted"]
    active.sort(key=lambda item: item["score"], reverse=True)
    lost.sort(key=lambda item: (item["lost_age"], -item["score"]))
    chosen = active[:limit]
    seen = {item["track_id"] for item in chosen}
    for item in lost:
        if len(chosen) >= limit:
            break
        if item["track_id"] not in seen:
            chosen.append(item)
            seen.add(item["track_id"])
    return chosen


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


class HandSlotMapper:
    """Map fragmented ByteTrack hand tracklets onto two stable logical IDs."""

    def __init__(self, num_slots: int, width: int, height: int):
        self.num_slots = num_slots
        self.width = width
        self.height = height
        self.diag = float((width ** 2 + height ** 2) ** 0.5)
        self.slots = [
            {"track_id": slot_id, "xyxy": None, "raw_ids": set(), "last_frame": 0}
            for slot_id in range(1, num_slots + 1)
        ]

    def _cost(self, item: dict, slot: dict) -> float:
        raw_id = item.get("raw_track_id")
        box = item["xyxy"]
        if slot["xyxy"] is None:
            target_x = self.width * slot["track_id"] / (self.num_slots + 1)
            cx, _cy = _box_center(box)
            return abs(cx - target_x) / max(1.0, self.width)

        cx, cy = _box_center(box)
        sx, sy = _box_center(slot["xyxy"])
        center_cost = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5 / max(1.0, self.diag)
        iou_cost = 1.0 - _box_iou(box, slot["xyxy"])
        cost = center_cost + 0.7 * iou_cost
        if raw_id is not None and raw_id in slot["raw_ids"]:
            cost *= 0.35
        elif raw_id is not None and any(raw_id in other["raw_ids"] for other in self.slots):
            return 1_000_000.0
        if item["state"] == "predicted":
            cost += 0.60
        return cost

    def assign(self, candidates: list[dict], frame_idx: int) -> list[dict]:
        if not candidates:
            return []

        active = [item for item in candidates if item["state"] == "active"]
        lost = [item for item in candidates if item["state"] == "predicted"]
        active.sort(key=lambda item: item["score"], reverse=True)
        lost.sort(key=lambda item: (item["lost_age"], -item["score"]))

        def dedupe_by_raw(items: list[dict]) -> list[dict]:
            seen = set()
            out = []
            for item in items:
                raw_id = item.get("raw_track_id")
                if raw_id is not None and raw_id in seen:
                    continue
                if raw_id is not None:
                    seen.add(raw_id)
                out.append(item)
            return out

        active = dedupe_by_raw(active)
        lost = dedupe_by_raw(lost)

        outputs: list[dict] = []
        used_slots: set[int] = set()

        def best_pairs(pool: list[dict], slot_indexes: list[int]) -> tuple[tuple[int, int], ...]:
            if not pool or not slot_indexes:
                return ()
            max_pairs = min(len(pool), len(slot_indexes))
            for pair_count in range(max_pairs, 0, -1):
                best_score = float("inf")
                pairs: tuple[tuple[int, int], ...] = ()
                for cand_indexes in itertools.combinations(range(len(pool)), pair_count):
                    for chosen_slots in itertools.permutations(slot_indexes, pair_count):
                        score = sum(self._cost(pool[ci], self.slots[si]) for ci, si in zip(cand_indexes, chosen_slots))
                        if score < best_score:
                            best_score = score
                            pairs = tuple(zip(cand_indexes, chosen_slots))
                if best_score < 999_999.0:
                    return pairs
            return ()

        active_pool = active[: max(self.num_slots * 3, self.num_slots)]
        for cand_index, slot_index in best_pairs(active_pool, list(range(self.num_slots))):
            item = dict(active_pool[cand_index])
            slot = self.slots[slot_index]
            raw_id = item.get("raw_track_id")
            slot["xyxy"] = item["xyxy"]
            if raw_id is not None:
                slot["raw_ids"].add(raw_id)
            slot["last_frame"] = frame_idx
            item["track_id"] = slot["track_id"]
            item["raw_track_id"] = int(raw_id) if raw_id is not None else None
            outputs.append(item)
            used_slots.add(slot_index)

        lost_pool = lost[: max(self.num_slots * 3, self.num_slots)]
        free_slots = [si for si in range(self.num_slots) if si not in used_slots]
        for cand_index, slot_index in best_pairs(lost_pool, free_slots):
            item = dict(lost_pool[cand_index])
            slot = self.slots[slot_index]
            raw_id = item.get("raw_track_id")
            slot["xyxy"] = item["xyxy"]
            if raw_id is not None:
                slot["raw_ids"].add(raw_id)
            slot["last_frame"] = frame_idx
            item["track_id"] = slot["track_id"]
            item["raw_track_id"] = int(raw_id) if raw_id is not None else None
            outputs.append(item)
            used_slots.add(slot_index)

        outputs.sort(key=lambda item: item["track_id"])
        return outputs


def _item_to_box(item: dict) -> Box:
    return Box(
        xyxy=tuple(float(v) for v in item["xyxy"]),
        score=float(item["score"]),
        label=item["label"],
        track_id=int(item["track_id"]),
        state=item["state"],
        lost_age=int(item.get("lost_age", 0)),
    )


# --- public API --------------------------------------------------------------


class HandFaceDetector:
    """Two YOLOv8n single-class models (face, hand) + a ByteTracker pair."""

    def __init__(
        self,
        face_ckpt: Path = FACE_CKPT,
        hand_ckpt: Path = HAND_CKPT,
        device: str | int | None = None,
        conf: float = 0.10,
        iou: float = 0.50,
        imgsz: int = 640,
        fps: float = 25.0,
        track_buffer: int = 75,
        max_pred_draw: int = 3,          # fill detector gaps of at most 3 frames with the tracker's prediction
        keep_face: int = 1,
        keep_hands: int = 2,
        half: bool | None = None,
    ):
        self.face_model = YOLO(str(face_ckpt))
        self.hand_model = YOLO(str(hand_ckpt))

        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = bool(torch.cuda.is_available()) if half is None else half

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.fps = fps
        self.track_buffer = track_buffer
        self.max_pred_draw = max_pred_draw
        self.keep_face = keep_face
        self.keep_hands = keep_hands

        self._reset_state()

    def _reset_state(self) -> None:
        self.face_tracker = make_tracker(self.track_buffer, self.fps)
        self.hand_tracker = make_tracker(self.track_buffer, self.fps)
        self._hand_slots: HandSlotMapper | None = None
        self._frame_idx = 0

    def reset(self, width: int | None = None, height: int | None = None) -> None:
        self._reset_state()
        if width is not None and height is not None:
            self._hand_slots = HandSlotMapper(self.keep_hands, width, height)

    # ----- batch interface (preferred — fast) -------------------------------

    def predict_batch(self, frames_or_paths) -> list[FrameResult]:
        """Run face+hand detection on a list of frames or image paths and apply
        tracking. Returns one FrameResult per input."""
        face_results = self.face_model.predict(
            frames_or_paths, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, half=self.half, verbose=False,
        )
        hand_results = self.hand_model.predict(
            frames_or_paths, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            device=self.device, half=self.half, verbose=False,
        )
        return list(self._apply_tracking(face_results, hand_results))

    def _apply_tracking(self, face_results, hand_results) -> Iterator[FrameResult]:
        for face_result, hand_result in zip(face_results, hand_results):
            # init slot mapper lazily from first frame size
            if self._hand_slots is None:
                h, w = face_result.orig_shape if hasattr(face_result, "orig_shape") else (260, 210)
                self._hand_slots = HandSlotMapper(self.keep_hands, int(w), int(h))

            face_active = _active_rows_to_items(
                self.face_tracker.update(_boxes_to_tracker(face_result), None), "face"
            )
            self.hand_tracker.update(_boxes_to_tracker(hand_result), None)

            face_candidates = face_active + _predicted_items(self.face_tracker, "face", self.max_pred_draw)
            hand_candidates = _yolo_result_to_items(hand_result, "hand", self.keep_hands) + _predicted_items(
                self.hand_tracker, "hand", self.max_pred_draw
            )
            face_chosen = _choose_tracks(face_candidates, self.keep_face)
            for item in face_chosen:
                item["raw_track_id"] = item["track_id"]
                item["track_id"] = 1
            chosen = face_chosen + self._hand_slots.assign(hand_candidates, self._frame_idx + 1)

            # YOLO-only: raw top-K detections, no track logic (visualisation only).
            yolo_face = _yolo_result_to_items(face_result, "face", self.keep_face)
            for det_index, item in enumerate(yolo_face, start=1):
                item["track_id"] = det_index
                item["state"] = "active"
            yolo_hand = _yolo_result_to_items(hand_result, "hand", self.keep_hands)
            # label hand1/hand2 left-to-right by box center
            yolo_hand.sort(key=lambda it: _box_center(it["xyxy"])[0])
            for slot_id, item in enumerate(yolo_hand, start=1):
                item["track_id"] = slot_id
                item["state"] = "active"
            yolo_items = yolo_face + yolo_hand

            yield FrameResult(
                frame_idx=self._frame_idx,
                yolo_boxes=[_item_to_box(it) for it in yolo_items],
                track_boxes=[_item_to_box(it) for it in chosen],
            )
            self._frame_idx += 1


# --- video / frame-folder helpers --------------------------------------------


def iter_video_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def fps_of(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        f = cap.get(cv2.CAP_PROP_FPS)
        return float(f) if f and f > 1e-3 else 25.0
    finally:
        cap.release()


def iter_frame_folder(folder: Path, pattern: str = "*.png") -> Iterator[tuple[int, np.ndarray]]:
    frames = sorted(folder.glob(pattern))
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        yield i, img


def run_on_video(
    detector: HandFaceDetector,
    source: Path,
    batch: int = 32,
) -> Iterator[tuple[np.ndarray, FrameResult]]:
    """Yield (BGR frame, FrameResult) per frame for a video file or PNG folder."""
    if source.is_dir():
        it = iter_frame_folder(source)
    else:
        it = iter_video_frames(source)

    frames: list[np.ndarray] = []
    detector.reset()
    buf: list[np.ndarray] = []

    for _idx, img in it:
        buf.append(img)
        if len(buf) >= batch:
            results = detector.predict_batch(buf)
            for f, r in zip(buf, results):
                yield f, r
            buf = []

    if buf:
        results = detector.predict_batch(buf)
        for f, r in zip(buf, results):
            yield f, r
