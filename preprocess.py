def preprocess(video_data):
    import os
    import cv2
    import pickle
    import pandas as pd
    import numpy as np
    from ultralytics import YOLO
    import supervision as sv
    from sklearn.cluster import KMeans
    import easyocr

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

    class JerseyNumberRecognizer:
        def __init__(self):
            self.reader = easyocr.Reader(["en"], gpu=True)
            self.jersey_cache = {}
            print("✅ Jersey OCR module initialized.")

        def recognize_jersey_number(self, player_crop, tracker_id):
            if tracker_id in self.jersey_cache:
                return self.jersey_cache[tracker_id]
            if player_crop.size == 0:
                return None

            crop_gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
            results = self.reader.readtext(
                crop_gray, allowlist="0123456789", detail=1
            )

            best_result = None
            for bbox, text, prob in results:
                if prob > 0.6 and text.isdigit() and len(text) <= 2:
                    if best_result is None or prob > best_result[2]:
                        best_result = (bbox, text, prob)

            if best_result:
                self.jersey_cache[tracker_id] = best_result[1]
                return best_result[1]

            return None

    class Tracker:
        def __init__(self, model_name="yolov8x.pt"):
            self.model = YOLO(model_name)
            self.tracker = sv.ByteTrack()
            self.jersey_recognizer = JerseyNumberRecognizer()

        def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
            if read_from_stub and stub_path and os.path.exists(stub_path):
                with open(stub_path, "rb") as f:
                    return pickle.load(f)

            tracks = {"players": [], "referees": [], "ball": []}

            for frame_num, frame in enumerate(frames):
                if frame_num % 20 == 0:
                    print(f"Processing frame {frame_num}/{len(frames)}")
                results = self.model.predict(frame, conf=0.1)[0]
                detections = sv.Detections.from_ultralytics(results)

                player_detections = detections[detections.class_id == 0]
                tracked_players = self.tracker.update_with_detections(player_detections)

                tracks["players"].append({})
                tracks["referees"].append({})

                for detection_data in tracked_players:
                    bbox = detection_data[0]
                    track_id = detection_data[4]

                    player_crop = frame[
                        int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
                    ]
                    jersey_num = self.jersey_recognizer.recognize_jersey_number(
                        player_crop, track_id
                    )
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox.tolist(),
                        "jersey_number": jersey_num,
                    }

                ball_detections = detections[detections.class_id == 32]
                tracks["ball"].append({})
                if len(ball_detections) > 0:
                    tracks["ball"][frame_num][1] = {
                        "bbox": ball_detections.xyxy[0].tolist()
                    }

            if stub_path:
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f)
            return tracks

        def add_position_to_tracks(self, tracks):
            for type, obj_tracks in tracks.items():
                for frame_num, track in enumerate(obj_tracks):
                    for id, info in track.items():
                        bbox = info["bbox"]
                        info["position"] = (
                            get_foot_position(bbox)
                            if type != "ball"
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
            x1_rect, y1_rect = x_center - rect_w // 2, (y2 - rect_h // 2) + 15

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
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 70), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            team_1_frames = np.sum(team_ball_control[: frame_num + 1] == 1)
            team_2_frames = np.sum(team_ball_control[: frame_num + 1] == 2)
            total = max(1, team_1_frames + team_2_frames)
            p1 = (team_1_frames / total) * 100
            p2 = (team_2_frames / total) * 100

            cv2.putText(
                frame,
                f"Team 1 Possession: {p1:.1f}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Team 2 Possession: {p2:.1f}%",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
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
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            text_x = (w - text_w) // 2
            text_y = h - 10
            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            return frame

    class TeamAssigner:
        def __init__(self):
            self.team_colors, self.player_team_dict, self.kmeans = {}, {}, None

        def get_player_color(self, frame, bbox):
            image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            if image.size == 0:
                return np.array([0, 0, 0])
            top_half = image[0 : int(image.shape[0] / 2), :]
            if top_half.size == 0:
                return np.array([0, 0, 0])
            kmeans = KMeans(
                n_clusters=2, init="k-means++", n_init=1, random_state=0
            ).fit(top_half.reshape(-1, 3))
            labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])
            corner_clusters = [
                labels[0, 0],
                labels[0, -1],
                labels[-1, 0],
                labels[-1, -1],
            ]
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

    class PlayerBallAssigner:
        def __init__(self):
            self.max_dist = 70

        def assign_ball_to_player(self, players, ball_bbox):
            if not ball_bbox:
                return -1
            ball_pos, min_dist, assigned_player = (
                get_center_of_bbox(ball_bbox),
                float("inf"),
                -1,
            )
            for id, player in players.items():
                dist = measure_distance(get_foot_position(player["bbox"]), ball_pos)
                if dist < self.max_dist and dist < min_dist:
                    min_dist, assigned_player = dist, id
            return assigned_player

    class CameraMovementEstimator:
        def __init__(self, frame):
            self.lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    10,
                    0.03,
                ),
            )
            self.features = dict(
                maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
            )

        def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
            if read_from_stub and stub_path and os.path.exists(stub_path):
                with open(stub_path, "rb") as f:
                    return pickle.load(f)
            movements = [[0, 0]] * len(frames)
            old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
            for i in range(1, len(frames)):
                new_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray, new_gray, old_features, None, **self.lk_params
                )

                good_new = new_features[status == 1]
                good_old = old_features[status == 1]

                move_x, move_y = 0, 0
                if len(good_new) > 0:
                    move_x, move_y = np.mean(good_old - good_new, axis=0).ravel()

                movements[i] = [move_x, move_y]
                old_gray = new_gray.copy()
                old_features = good_new.reshape(-1, 1, 2)
            if stub_path:
                with open(stub_path, "wb") as f:
                    pickle.dump(movements, f)
            return movements

        def add_adjust_positions_to_tracks(self, tracks, movements):
            for type, obj_tracks in tracks.items():
                for i, track in enumerate(obj_tracks):
                    for id, info in track.items():
                        info["position_adjusted"] = (
                            info["position"][0] + movements[i][0],
                            info["position"][1] + movements[i][1],
                        )

    class ViewTransformer:
        def __init__(self):
            court_w, court_l = 34, 52.5
            self.pixel_verts = np.float32(
                [[110, 1035], [265, 275], [910, 260], [1640, 915]]
            )
            self.target_verts = np.float32(
                [[0, court_w], [0, 0], [court_l, 0], [court_l, court_w]]
            )
            self.transformer = cv2.getPerspectiveTransform(
                self.pixel_verts, self.target_verts
            )

        def transform_point(self, point):
            p = (int(point[0]), int(point[1]))
            is_inside = cv2.pointPolygonTest(self.pixel_verts, p, False) >= 0
            if not is_inside:
                return None
            reshaped = np.array(point).reshape(-1, 1, 2).astype(np.float32)
            transformed = cv2.perspectiveTransform(reshaped, self.transformer)
            return transformed.reshape(-1, 2)

        def add_transformed_position_to_tracks(self, tracks):
            for type, obj_tracks in tracks.items():
                for track in obj_tracks:
                    for id, info in track.items():
                        pos = info.get("position_adjusted", info.get("position"))
                        if pos:
                            transformed = self.transform_point(pos)
                            info["position_transformed"] = (
                                transformed.squeeze().tolist()
                                if transformed is not None
                                else None
                            )

    class SpeedAndDistanceEstimator:
        def __init__(self):
            self.frame_window, self.frame_rate = 24, 24

        def add_speed_and_distance_to_tracks(self, tracks):
            total_dist = {}
            for type, obj_tracks in tracks.items():
                if type not in ["players", "referees"]:
                    continue
                for i in range(len(obj_tracks)):
                    for id, info in obj_tracks[i].items():
                        if i > 0:
                            prev_info = tracks[type][i - 1].get(id)
                            if (
                                prev_info
                                and info.get("position_transformed")
                                and prev_info.get("position_transformed")
                            ):
                                dist = measure_distance(
                                    info["position_transformed"],
                                    prev_info["position_transformed"],
                                )
                                total_dist[id] = total_dist.get(id, 0) + dist
                                speed_mps = dist * self.frame_rate
                                info["speed"] = speed_mps * 3.6
                                info["distance"] = total_dist[id]

        def draw_speed_and_distance(self, frames, tracks):
            output_frames = []
            for i, frame in enumerate(frames):
                for type, obj_tracks in tracks.items():
                    if type not in ["players", "referees"]:
                        continue
                    for id, info in obj_tracks[i].items():
                        if "speed" in info:
                            x, y = get_foot_position(info["bbox"])
                            cv2.putText(
                                frame,
                                f"{info['speed']:.1f} km/h",
                                (x - 20, y + 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2,
                            )
                output_frames.append(frame)
            return output_frames

    frames = video_data["frames"]
    STUB_PATH = video_data["STUB_PATH"]

    tracker = Tracker("yolov8x.pt")
    camera_estimator = CameraMovementEstimator(frames[0])
    view_transformer = ViewTransformer()
    speed_estimator = SpeedAndDistanceEstimator()
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()

    print("Stage 1: Performing object detection and tracking...")
    tracks = tracker.get_object_tracks(frames, read_from_stub=False, stub_path=STUB_PATH)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracker.add_position_to_tracks(tracks)

    print("Stage 2: Estimating camera motion and transforming perspective...")
    camera_movement = camera_estimator.get_camera_movement(frames)
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
    view_transformer.add_transformed_position_to_tracks(tracks)
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    print("Stage 3: Assigning teams...")
    team_assigner.assign_team_color(frames[0], tracks["players"][0])

    for frame_num, frame in enumerate(frames):
        player_track = tracks["players"][frame_num]
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frame, track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors.get(team, (0, 0, 255))
            )

    video_data["tracks"] = tracks
    video_data["tracker"] = tracker
    video_data["speed_estimator"] = speed_estimator
    video_data["player_assigner"] = player_assigner
    video_data["PlayerBallAssigner"] = PlayerBallAssigner
    return video_data
