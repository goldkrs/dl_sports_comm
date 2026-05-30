"""
model.py — Stage 4 of the Football-Comment pipeline.
Imports all commentary classes from commentary_engine.py.
"""

from collections import Counter, deque

from commentary_engine import ImprovedCommentaryEngine, RealTimeTicker, EventDetector


def model(video_data):
    fps = video_data["fps"]
    frames = video_data["frames"]
    tracks = video_data["tracks"]
    player_assigner = video_data["player_assigner"]

    commentary_engine = ImprovedCommentaryEngine(fps=fps)
    ticker = RealTimeTicker(fps=fps)

    print("Stage 4: Detecting events for commentary context...")
    event_detector = EventDetector(frame_rate=int(fps))
    events_df = event_detector.detect_events(tracks, player_assigner)

    # ---------------------------------------------------------------------------
    # Possession smoothing
    # Rapid ball-contests (50-50s, clearances, loose balls) cause the raw
    # possession assignment to toggle frame-by-frame between teams, which
    # produces meaningless "wins possession" commentary on every clip.
    # Fix: maintain a rolling majority-vote window (~1.5 seconds of frames).
    # The visual possession bar uses the smoothed team; has_ball on the
    # individual player still tracks the geometrically nearest player so the
    # on-screen ellipse highlight remains accurate.
    # ---------------------------------------------------------------------------
    SMOOTH_FRAMES = max(1, int(fps * 1.5))
    raw_team_buffer: deque[int] = deque(maxlen=SMOOTH_FRAMES)

    print("Stage 5: Tracking ball possession and generating commentary...")
    team_ball_control = []
    ticker_history = []
    gemini_history = []

    for frame_num, frame in enumerate(frames):
        player_track = tracks["players"][frame_num]
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")

        # Reset has_ball for all players this frame
        for player_id in tracks["players"][frame_num]:
            tracks["players"][frame_num][player_id]["has_ball"] = False

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        raw_team = 0
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            raw_team = tracks["players"][frame_num][assigned_player].get("team", 0)

        # Majority-vote smoothing: only register possession once a team has
        # held the ball for the majority of the last 1.5 seconds
        raw_team_buffer.append(raw_team)
        non_zero = [t for t in raw_team_buffer if t != 0]
        if non_zero:
            smoothed_team = Counter(non_zero).most_common(1)[0][0]
        else:
            smoothed_team = team_ball_control[-1] if team_ball_control else 0

        team_ball_control.append(smoothed_team)

        ticker_history.append(ticker.update(tracks, frame_num))
        commentary_engine.update_with_context(
            frame, tracks, frame_num, events_df, smoothed_team
        )
        gemini_history.append(commentary_engine.latest_commentary)

        if frame_num % 100 == 0:
            print(f"  Commentary progress: {frame_num}/{len(frames)} frames")

    video_data["commentary_engine"] = commentary_engine
    video_data["team_ball_control"] = team_ball_control
    video_data["ticker_history"] = ticker_history
    video_data["gemini_history"] = gemini_history
    return video_data
