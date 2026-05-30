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
    display_commentary = [str(c).strip() if c else "" for c in display_commentary]

    # ---------------------------------------------------------------------------
    # Memory-efficient rendering
    # ---------------------------------------------------------------------------
    # We draw annotations DIRECTLY on each frame (no .copy()) and immediately
    # move the annotated frame into output_frames while discarding the original
    # reference.  This halves peak RAM vs. the old approach of building a full
    # second list of frame copies alongside the originals.
    #
    # Speed overlays (draw_speed_and_distance) also need the frames, so we pass
    # a reference to output_frames (same objects) rather than duplicating again.
    # ---------------------------------------------------------------------------
    output_frames = []
    for frame_num in range(len(frames)):
        # Pop the frame out of the originals list so its reference count drops
        # to 1 (only output_frames will hold it) — GC can free it next cycle.
        frame = frames[frame_num]
        frames[frame_num] = None   # release original slot immediately

        current_commentary = (
            display_commentary[frame_num] if frame_num < len(display_commentary) else " "
        )

        player_dict = tracks["players"][frame_num]
        ball_dict = tracks.get("ball", [])[frame_num]

        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = tracker._draw_player_ellipse(
                frame, player["bbox"], color, track_id, player.get("jersey_number")
            )
            if player.get("has_ball", False):
                frame = tracker._draw_triangle(frame, player["bbox"], (0, 0, 255))

        if 1 in ball_dict:
            frame = tracker._draw_triangle(frame, ball_dict[1]["bbox"], (0, 255, 0))

        frame = tracker._draw_team_ball_control(frame, frame_num, team_ball_control)
        frame = tracker._draw_commentary_overlay(frame, current_commentary)
        output_frames.append(frame)

    # draw_speed_and_distance operates completely in-place on the same frame objects
    speed_estimator.draw_speed_and_distance(output_frames, tracks)
    video_data["display_commentary"] = display_commentary
    video_data["output_frames"] = output_frames
    # Free the now-empty original frames list
    video_data["frames"] = []
    return video_data
