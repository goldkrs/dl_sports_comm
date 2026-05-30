"""
preprocess.py — Stage 1–3 of the Football-Comment pipeline.
Imports all CV classes from tracker.py (module-level definitions).
"""

from tracker import (
    Tracker,
    TeamAssigner,
    PlayerBallAssigner,
    CameraMovementEstimator,
    ViewTransformer,
    SpeedAndDistanceEstimator,
)


def preprocess(video_data):
    frames = video_data["frames"]
    stub_path = video_data["STUB_PATH"]
    fps = video_data["fps"]

    tracker = Tracker("yolov8x.pt")
    camera_estimator = CameraMovementEstimator(frames[0])
    view_transformer = ViewTransformer(pixel_verts=video_data.get("pixel_verts"))
    speed_estimator = SpeedAndDistanceEstimator(frame_rate=int(fps))
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()

    print("Stage 1: Object detection and tracking...")
    # read_from_stub=True: loads cache if it exists, saves on first run
    tracks = tracker.get_object_tracks(
        frames, read_from_stub=True, stub_path=stub_path
    )
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracker.add_position_to_tracks(tracks)

    print("Stage 2: Camera motion estimation and perspective transform...")
    camera_movement = camera_estimator.get_camera_movement(frames)
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
    view_transformer.add_transformed_position_to_tracks(tracks)
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    print("Stage 3: Team assignment...")
    team_assigner.assign_team_color(frames[0], tracks["players"][0])
    for frame_num, frame in enumerate(frames):
        for player_id, track in tracks["players"][frame_num].items():
            team = team_assigner.get_player_team(frame, track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors.get(team, (0, 0, 255))
            )

    video_data["tracks"] = tracks
    video_data["tracker"] = tracker
    video_data["speed_estimator"] = speed_estimator
    video_data["player_assigner"] = player_assigner
    # Expose class so model.py can instantiate independently if needed
    video_data["PlayerBallAssigner"] = PlayerBallAssigner
    return video_data
