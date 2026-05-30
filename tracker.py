"""
tracker.py — Computer vision pipeline classes for the Football-Comment system.
All classes are defined at module level for testability and reuse.
"""

import torch


def _cuda_available() -> bool:
    """Return True if a CUDA-capable GPU is available."""
    return torch.cuda.is_available()


def _device_str() -> str:
    """Return 'cuda' or 'cpu' depending on hardware."""
    return "cuda" if _cuda_available() else "cpu"


print(f"[INFO] Hardware: {'GPU (CUDA)' if _cuda_available() else 'CPU only'}")

import os
import pickle
import shutil
from collections import deque

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ultralytics import YOLO
import supervision as sv
import easyocr


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return int(bbox[2] - bbox[0])


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def check_ffmpeg():
    """Raise a clear error if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required but was not found on PATH. "
            "Install ffmpeg and ensure it is available in your system PATH."
        )


# ---------------------------------------------------------------------------
# Jersey OCR
# ---------------------------------------------------------------------------

class JerseyNumberRecognizer:
    def __init__(self, use_gpu=None):
        # Default: auto-detect GPU rather than hardcoding to CPU
        if use_gpu is None:
            use_gpu = _cuda_available()
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
        self.jersey_cache = {}
        print(f"[INFO] Jersey OCR initialized ({'GPU' if use_gpu else 'CPU'})")

    def recognize_jersey_number(self, player_crop, tracker_id):
        if tracker_id in self.jersey_cache:
            return self.jersey_cache[tracker_id]
        if player_crop.size == 0:
            return None
        crop_gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(crop_gray, allowlist="0123456789", detail=1)
        best_result = None
        for bbox, text, prob in results:
            if prob > 0.6 and text.isdigit() and len(text) <= 2:
                if best_result is None or prob > best_result[2]:
                    best_result = (bbox, text, prob)
        if best_result:
            self.jersey_cache[tracker_id] = best_result[1]
            return best_result[1]
        return None


# ---------------------------------------------------------------------------
# Object Tracker
# ---------------------------------------------------------------------------

class Tracker:
    PLAYER_CLASS_ID = 0   # COCO "person"
    BALL_CLASS_ID = 32    # COCO "sports ball"
    DETECTION_CONFIDENCE = 0.25  # raised from 0.1 to reduce false positives

    def __init__(self, model_name="yolov8x.pt"):
        self.device = _device_str()
        self.model = YOLO(model_name)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.tracker = sv.ByteTrack()  # ByteTracker doesn't exist until sv>0.28
        # EasyOCR also auto-detects GPU via JerseyNumberRecognizer default
        self.jersey_recognizer = JerseyNumberRecognizer()
        print(f"[INFO] YOLO tracker initialized (device: {self.device})")

    def get_object_tracks(self, frames, read_from_stub=True, stub_path=None):
        """Detect and track players/ball. Loads from cache if available."""
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                print(f"[INFO] Loaded cached tracks from {stub_path}")
                return pickle.load(f)

        tracks = {"players": [], "referees": [], "ball": []}
        for frame_num, frame in enumerate(frames):
            if frame_num % 20 == 0:
                print(f"  Detection: {frame_num}/{len(frames)} frames")
            results = self.model.predict(
                    frame, conf=self.DETECTION_CONFIDENCE, device=self.device, verbose=False
                )[0]
            detections = sv.Detections.from_ultralytics(results)

            player_detections = detections[detections.class_id == self.PLAYER_CLASS_ID]
            tracked_players = self.tracker.update_with_detections(player_detections)

            tracks["players"].append({})
            tracks["referees"].append({})

            for detection_data in tracked_players:
                bbox = detection_data[0]
                track_id = detection_data[4]
                player_crop = frame[
                    int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])
                ]
                jersey_num = self.jersey_recognizer.recognize_jersey_number(
                    player_crop, track_id
                )
                tracks["players"][frame_num][track_id] = {
                    "bbox": bbox.tolist(),
                    "jersey_number": jersey_num,
                }

            ball_detections = detections[detections.class_id == self.BALL_CLASS_ID]
            tracks["ball"].append({})
            if len(ball_detections) > 0:
                tracks["ball"][frame_num][1] = {
                    "bbox": ball_detections.xyxy[0].tolist()
                }

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            print(f"[INFO] Saved tracks cache to {stub_path}")

        return tracks

    def add_position_to_tracks(self, tracks):
        for obj_type, obj_tracks in tracks.items():
            for track in obj_tracks:
                for obj_id, info in track.items():
                    bbox = info["bbox"]
                    info["position"] = (
                        get_foot_position(bbox)
                        if obj_type != "ball"
                        else get_center_of_bbox(bbox)
                    )

    def interpolate_ball_positions(self, ball_positions):
        ball_bboxes = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df = (
            pd.DataFrame(ball_bboxes, columns=["x1", "y1", "x2", "y2"])
            .interpolate()
            .bfill()
        )
        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_player_ellipse(self, frame, bbox, color, track_id, jersey_num):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )
        label = f"#{jersey_num}" if jersey_num else str(track_id)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        rect_w, rect_h = w + 10, h + 10
        x1_rect = x_center - rect_w // 2
        y1_rect = (y2 - rect_h // 2) + 15
        cv2.rectangle(
            frame,
            (x1_rect, y1_rect),
            (x1_rect + rect_w, y1_rect + rect_h),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (x1_rect + 5, y1_rect + h + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        return frame

    def _draw_triangle(self, frame, bbox, color):
        y, x = int(bbox[1]), int(get_center_of_bbox(bbox)[0])
        points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [points], 0, (0, 0, 0), 2)
        return frame

    def _draw_team_ball_control(self, frame, frame_num, team_ball_control):
        x1, y1, x2, y2 = 10, 10, 350, 70
        overlay = frame[y1:y2, x1:x2].copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.5, frame[y1:y2, x1:x2], 0.5, 0, frame[y1:y2, x1:x2])
        arr = np.asarray(team_ball_control[: frame_num + 1])
        team_1_frames = int(np.sum(arr == 1))
        team_2_frames = int(np.sum(arr == 2))
        total = max(1, team_1_frames + team_2_frames)
        p1 = (team_1_frames / total) * 100
        p2 = (team_2_frames / total) * 100
        cv2.putText(frame, f"Team 1 Possession: {p1:.1f}%", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2 Possession: {p2:.1f}%", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def _draw_commentary_overlay(self, frame, text):
        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        font_scale = 1.0
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        target_w = w * 0.9
        if text_w > target_w:
            font_scale = target_w / text_w
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        banner_h = text_h + 20
        y1 = h - banner_h
        overlay = frame[y1:h, 0:w].copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame[y1:h, 0:w], 0.4, 0, frame[y1:h, 0:w])
        text_x = (w - text_w) // 2
        text_y = h - 10
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        return frame


# ---------------------------------------------------------------------------
# Team Assigner
# ---------------------------------------------------------------------------

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if image.size == 0:
            return np.array([0, 0, 0])
        top_half = image[0:int(image.shape[0] / 2), :]
        if top_half.size == 0:
            return np.array([0, 0, 0])
        kmeans = KMeans(
            n_clusters=2, init="k-means++", n_init=1, random_state=0
        ).fit(top_half.reshape(-1, 3))
        labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])
        corner_clusters = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        return kmeans.cluster_centers_[1 - non_player_cluster]

    def assign_team_color(self, frame, player_detections):
        if not player_detections:
            return
        colors = [
            self.get_player_color(frame, det["bbox"])
            for _, det in player_detections.items()
        ]
        self.kmeans = KMeans(
            n_clusters=2, init="k-means++", n_init=10, random_state=0
        ).fit(colors)
        self.team_colors[1], self.team_colors[2] = self.kmeans.cluster_centers_

    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        if self.kmeans is None:
            return 0
        color = self.get_player_color(frame, bbox)
        team_id = self.kmeans.predict(color.reshape(1, -1))[0] + 1
        self.player_team_dict[player_id] = team_id
        return team_id


# ---------------------------------------------------------------------------
# Ball-Player Assignment
# ---------------------------------------------------------------------------

class PlayerBallAssigner:
    def __init__(self, max_dist=70):
        self.max_dist = max_dist

    def assign_ball_to_player(self, players, ball_bbox):
        if not ball_bbox:
            return -1
        ball_pos = get_center_of_bbox(ball_bbox)
        min_dist = float("inf")
        assigned_player = -1
        for player_id, player in players.items():
            dist = measure_distance(get_foot_position(player["bbox"]), ball_pos)
            if dist < self.max_dist and dist < min_dist:
                min_dist = dist
                assigned_player = player_id
        return assigned_player


# ---------------------------------------------------------------------------
# Camera Movement Estimator (with periodic feature re-detection)
# ---------------------------------------------------------------------------

class CameraMovementEstimator:
    FEATURE_REDETECT_INTERVAL = 30  # re-detect every N frames to prevent drift

    def __init__(self, frame):
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

    def _detect_features(self, gray_frame):
        return cv2.goodFeaturesToTrack(gray_frame, **self.feature_params)

    def get_camera_movement(self, frames, read_from_stub=True, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        movements = [[0, 0]] * len(frames)
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = self._detect_features(old_gray)
        if old_features is None:
            return movements

        for i in range(1, len(frames)):
            # Periodically re-detect to prevent tracking drift
            if i % self.FEATURE_REDETECT_INTERVAL == 0:
                fresh = self._detect_features(old_gray)
                if fresh is not None and len(fresh) > 0:
                    old_features = fresh

            new_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, new_gray, old_features, None, **self.lk_params
            )
            if new_features is None or status is None:
                movements[i] = [0, 0]
                old_gray = new_gray.copy()
                continue

            valid = status.flatten() == 1
            if not valid.any():
                movements[i] = [0, 0]
                old_gray = new_gray.copy()
                fresh = self._detect_features(new_gray)
                if fresh is not None:
                    old_features = fresh
                continue

            good_new = new_features[valid]
            good_old = old_features[valid]
            move_x, move_y = np.mean(good_old - good_new, axis=0).ravel()
            movements[i] = [move_x, move_y]
            old_gray = new_gray.copy()
            old_features = good_new.reshape(-1, 1, 2)

        if stub_path:
            with open(stub_path, "wb") as f:
                pickle.dump(movements, f)
        return movements

    def add_adjust_positions_to_tracks(self, tracks, movements):
        for obj_type, obj_tracks in tracks.items():
            for i, track in enumerate(obj_tracks):
                for obj_id, info in track.items():
                    info["position_adjusted"] = (
                        info["position"][0] + movements[i][0],
                        info["position"][1] + movements[i][1],
                    )


# ---------------------------------------------------------------------------
# Perspective Transform (configurable per video)
# ---------------------------------------------------------------------------

# Default pixel vertices calibrated for the CityUtdR.mp4 broadcast angle.
# Provide custom pixel_verts when processing other footage.
DEFAULT_PIXEL_VERTS = [[110, 1035], [265, 275], [910, 260], [1640, 915]]
DEFAULT_FIELD_DIMS = (52.5, 34)  # (length_m, width_m)


class ViewTransformer:
    def __init__(self, pixel_verts=None, field_dims=None):
        """
        pixel_verts : list of 4 [x, y] pixel coords marking pitch corners in
                      broadcast frame order: bottom-left, top-left, top-right,
                      bottom-right. Defaults to values for CityUtdR.mp4.
        field_dims  : (length_m, width_m) of the mapped pitch section.
        """
        if pixel_verts is None:
            pixel_verts = DEFAULT_PIXEL_VERTS
        court_l, court_w = field_dims if field_dims else DEFAULT_FIELD_DIMS
        self.pixel_verts = np.float32(pixel_verts)
        self.target_verts = np.float32(
            [[0, court_w], [0, 0], [court_l, 0], [court_l, court_w]]
        )
        self.transformer = cv2.getPerspectiveTransform(
            self.pixel_verts, self.target_verts
        )

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        if cv2.pointPolygonTest(self.pixel_verts, p, False) < 0:
            return None
        reshaped = np.array(point).reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.transformer)
        return transformed.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for obj_type, obj_tracks in tracks.items():
            for track in obj_tracks:
                for obj_id, info in track.items():
                    pos = info.get("position_adjusted", info.get("position"))
                    if pos:
                        transformed = self.transform_point(pos)
                        info["position_transformed"] = (
                            transformed.squeeze().tolist()
                            if transformed is not None
                            else None
                        )


# ---------------------------------------------------------------------------
# Speed and Distance Estimator
# ---------------------------------------------------------------------------

class SpeedAndDistanceEstimator:
    SPEED_SMOOTH_WINDOW = 5  # frames — rolling average to remove jitter

    def __init__(self, frame_rate=24):
        self.frame_rate = frame_rate
        self._speed_history: dict[int, deque] = {}  # {player_id: deque of raw speeds}

    def add_speed_and_distance_to_tracks(self, tracks):
        total_dist: dict[int, float] = {}
        for obj_type, obj_tracks in tracks.items():
            if obj_type not in ["players", "referees"]:
                continue
            for i in range(len(obj_tracks)):
                for obj_id, info in obj_tracks[i].items():
                    if i > 0:
                        prev_info = tracks[obj_type][i - 1].get(obj_id)
                        if (
                            prev_info
                            and info.get("position_transformed")
                            and prev_info.get("position_transformed")
                        ):
                            dist = measure_distance(
                                info["position_transformed"],
                                prev_info["position_transformed"],
                            )
                            total_dist[obj_id] = total_dist.get(obj_id, 0) + dist
                            raw_speed = dist * self.frame_rate * 3.6  # km/h

                            # Rolling average smoothing — eliminates jitter from
                            # bounding-box pixel shifts between frames
                            if obj_id not in self._speed_history:
                                self._speed_history[obj_id] = deque(
                                    maxlen=self.SPEED_SMOOTH_WINDOW
                                )
                            self._speed_history[obj_id].append(raw_speed)
                            smoothed = (
                                sum(self._speed_history[obj_id])
                                / len(self._speed_history[obj_id])
                            )

                            info["speed"] = smoothed
                            info["distance"] = total_dist[obj_id]

    def draw_speed_and_distance(self, frames, tracks):
        """Draw speed overlays for the ball carrier and the nearest pressing
        defender only, to avoid text clutter when players cluster together."""
        for i, frame in enumerate(frames):
            player_tracks = tracks["players"][i]

            # Identify ball carrier
            ball_carrier_id = None
            ball_carrier_team = None
            for pid, info in player_tracks.items():
                if info.get("has_ball"):
                    ball_carrier_id = pid
                    ball_carrier_team = info.get("team")
                    break

            # Identify nearest pressing defender
            pressing_defender_id = None
            if ball_carrier_id is not None:
                carrier_foot = get_foot_position(player_tracks[ball_carrier_id]["bbox"])
                min_dist = float("inf")
                for pid, info in player_tracks.items():
                    if info.get("team") != ball_carrier_team:
                        d = measure_distance(
                            carrier_foot, get_foot_position(info["bbox"])
                        )
                        if d < min_dist:
                            min_dist = d
                            pressing_defender_id = pid

            # Only render speed for the two relevant players
            priority_ids = {pid for pid in (ball_carrier_id, pressing_defender_id) if pid is not None}

            for obj_type, obj_tracks in tracks.items():
                if obj_type not in ["players", "referees"]:
                    continue
                for obj_id, info in obj_tracks[i].items():
                    if "speed" not in info:
                        continue
                    # Show for priority players; show all referees (usually <=2)
                    if obj_type == "players" and obj_id not in priority_ids:
                        continue
                    x, y = get_foot_position(info["bbox"])
                    label = f"{info['speed']:.1f} km/h"
                    # Ball carrier gets a highlighted colour
                    colour = (
                        (0, 200, 0) if obj_id == ball_carrier_id else (0, 0, 255)
                    )
                    cv2.putText(
                        frame, label, (x - 20, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
                    )
        return frames
