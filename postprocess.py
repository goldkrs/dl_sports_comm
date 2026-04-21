def postprocess(video_data):
    import numpy as np
    from text_aggregator import stabilize_commentary_timeline

    frames = video_data["frames"]
    tracks = video_data["tracks"]
    tracker = video_data["tracker"]
    speed_estimator = video_data["speed_estimator"]
    commentary_engine = video_data["commentary_engine"]
    ticker_history = video_data["ticker_history"]
    gemini_history = video_data["gemini_history"]
    team_ball_control = np.array(video_data["team_ball_control"])

    print("Stage 6: Combining commentary and saving final video...")
    display_commentary = ticker_history.copy()
    last_gemini_comment = gemini_history[0]
    for i, comment in enumerate(gemini_history):
        if comment != last_gemini_comment:
            start_frame = max(0, i - commentary_engine.clip_length_frames)
            for j in range(start_frame, i):
                if j < len(display_commentary):
                    display_commentary[j] = comment
            last_gemini_comment = comment

    display_commentary = stabilize_commentary_timeline(display_commentary, video_data["fps"])

    output_frames = []
    for frame_num, frame in enumerate(frames):
        frame_copy = frame.copy()
        current_commentary = (
            display_commentary[frame_num] if frame_num < len(display_commentary) else " "
        )

        player_dict = tracks["players"][frame_num]
        ball_dict = tracks.get("ball", [])[frame_num]

        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame_copy = tracker._draw_player_ellipse(
                frame_copy, player["bbox"], color, track_id, player.get("jersey_number")
            )
            if player.get("has_ball", False):
                frame_copy = tracker._draw_triangle(frame_copy, player["bbox"], (0, 0, 255))

        if 1 in ball_dict:
            frame_copy = tracker._draw_triangle(frame_copy, ball_dict[1]["bbox"], (0, 255, 0))

        frame_copy = tracker._draw_team_ball_control(frame_copy, frame_num, team_ball_control)
        frame_copy = tracker._draw_commentary_overlay(frame_copy, current_commentary)
        output_frames.append(frame_copy)

    output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)
    video_data["display_commentary"] = display_commentary
    video_data["output_frames"] = output_frames
    return video_data
