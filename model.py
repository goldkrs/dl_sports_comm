def model(video_data):
    import pandas as pd
    import google.generativeai as genai
    from collections import deque

    PlayerBallAssigner = video_data["PlayerBallAssigner"]

    class ImprovedCommentaryEngine:
        def __init__(self, clip_duration_seconds=5, fps=24):
            self.clip_length_frames = int(clip_duration_seconds * fps)
            self.frame_buffer = deque(maxlen=self.clip_length_frames)
            self.latest_commentary = "Match analysis is starting..."
            self.fps = fps
            self.match_context = {
                "possession_changes": [],
                "recent_events": [],
                "ball_position_history": [],
                "player_movements": [],
            }

            print("🎙️ Initializing Enhanced Gemini Commentary Engine...")
            try:
                api_key = "abc"
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("models/gemini-2.5-flash")
                print("✅ Gemini 2.5 Flash model loaded successfully.")
            except Exception as e:
                self.model = None
                print(f"⚠️ Could not initialize Gemini model: {e}")

        def update_with_context(self, frame, tracks_data, frame_num, events_data=None):
            if not self.model:
                return

            game_context = self._extract_game_context(tracks_data, frame_num, events_data)

            self.match_context["recent_events"].append(game_context)
            if len(self.match_context["recent_events"]) > 10:
                self.match_context["recent_events"].pop(0)

            self.frame_buffer.append(frame)

            if len(self.frame_buffer) == self.clip_length_frames:
                print("Generating tactical summary...")
                new_comment = self._generate_contextual_commentary(game_context)
                if new_comment:
                    self.latest_commentary = new_comment
                self.frame_buffer.clear()

        def _extract_game_context(self, tracks_data, frame_num, events_data):
            context = {
                "frame_num": frame_num,
                "timestamp": f"{int(frame_num / (self.fps * 60))}:{int((frame_num / self.fps) % 60):02d}",
                "players_detected": len(tracks_data["players"][frame_num]),
                "ball_detected": 1 in tracks_data["ball"][frame_num],
                "possession": None,
                "ball_speed": 0,
                "recent_events": [],
            }

            for player_id, player_info in tracks_data["players"][frame_num].items():
                if player_info.get("has_ball", False):
                    context["possession"] = (
                        f"Player {player_id} (Team {player_info.get('team', 'Unknown')})"
                    )
                    break

            if events_data is not None and not events_data.empty:
                recent_events = events_data[
                    (events_data["minute"] * 60 + events_data["second"])
                    >= (frame_num / self.fps - 10)
                ].tail(3)
                context["recent_events"] = recent_events.to_dict("records")

            return context

        def _generate_contextual_commentary(self, game_context):
            return self._generate_fallback_commentary(game_context)

        def _create_detailed_prompt(self, context):
            prompt = f"""You are a professional football (soccer) tactical analyst.

        CURRENT GAME STATE:
        - Match Time: {context['timestamp']}
        - Ball Possession: {context.get('possession', 'Unclear')}
        - Recent Match Events: {self._format_recent_events(context.get('recent_events', []))}

        INSTRUCTIONS:
        1. Analyze the short video clip of a football match.
        2. Provide a brief, factual, tactical summary of the most significant action.
        3. Describe the sequence of play objectively. Example: "The player in red receives a pass, moves past a defender, and attempts a shot which is blocked."
        4. Do NOT use emotional or exciting commentary language like "incredible!" or "what a save!".
        5. Your entire response must be a single, concise sentence (max 25 words).

        Analyze the clip and provide your tactical summary:"""
            return prompt

        def _format_recent_events(self, events):
            if not events:
                return "No recent significant events detected."

            formatted = []
            for event in events[-3:]:
                if isinstance(event, dict):
                    event_type = event.get("type_name", "Unknown")
                    team = event.get("team_name", "Unknown Team")
                    formatted.append(f"- {event_type} by {team}")

            return "\n".join(formatted) if formatted else "No recent significant events detected."

        def _generate_fallback_commentary(self, context):
            if context.get("possession"):
                return f"Play continues with {context['possession']} in possession."
            return "The match continues with both teams looking for opportunities."

    class RealTimeTicker:
        """
        Generates a simple, real-time text commentary for each frame based on game state.
        """

        def __init__(self, fps=24):
            self.fps = fps
            self.last_player_id = -1
            self.last_team_id = -1
            self.ticker_text = "Match begins!"
            self.text_display_frames = 0

        def _get_ball_carrier(self, player_track):
            for player_id, data in player_track.items():
                if data.get("has_ball", False):
                    return player_id, data.get("team")
            return -1, -1

        def update(self, tracks, frame_num):
            if self.text_display_frames > 0:
                self.text_display_frames -= 1
                return self.ticker_text

            player_track = tracks["players"][frame_num]
            current_player_id, current_team_id = self._get_ball_carrier(player_track)

            if (
                current_player_id != -1
                and self.last_player_id != -1
                and current_player_id != self.last_player_id
                and current_team_id == self.last_team_id
            ):
                self.ticker_text = (
                    f"Pass from Player {self.last_player_id} to Player {current_player_id}."
                )
                self.text_display_frames = self.fps * 2

            elif (
                current_player_id != -1
                and self.last_team_id != -1
                and current_team_id != self.last_team_id
            ):
                self.ticker_text = f"Team {current_team_id} gains possession!"
                self.text_display_frames = self.fps * 2

            else:
                if current_player_id != -1:
                    self.ticker_text = (
                        f"Player {current_player_id} (Team {current_team_id}) on the ball."
                    )
                else:
                    self.ticker_text = "Ball is loose."

            if current_player_id != -1:
                self.last_player_id = current_player_id
                self.last_team_id = current_team_id
            else:
                self.last_player_id = -1

            return self.ticker_text

    class EventDetector:
        def __init__(self):
            self.shot_speed_threshold_mps = 15
            self.frame_rate = 24

        def detect_events(self, tracks):
            player_assigner = PlayerBallAssigner()
            ball_possession_log = []
            for frame_num in range(len(tracks["players"])):
                player_track = tracks["players"][frame_num]
                ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")
                assigned_player_id = (
                    player_assigner.assign_ball_to_player(player_track, ball_bbox)
                    if ball_bbox
                    else -1
                )
                ball_possession_log.append(assigned_player_id)

            events = []
            last_player_with_ball, pass_start_info = -1, {}
            for frame_num, current_player_id in enumerate(ball_possession_log):
                ball_pos_transformed = tracks["ball"][frame_num].get(1, {}).get(
                    "position_transformed"
                )
                if not ball_pos_transformed:
                    continue

                is_valid_pass = (
                    current_player_id != last_player_with_ball
                    and last_player_with_ball != -1
                    and current_player_id != -1
                )
                if is_valid_pass:
                    start_player_team = tracks["players"][pass_start_info["frame"]][
                        last_player_with_ball
                    ].get("team")
                    end_player_team = tracks["players"][frame_num].get(
                        current_player_id, {}
                    ).get("team")
                    if start_player_team == end_player_team and start_player_team is not None:
                        events.append(
                            {
                                "type_name": "Pass",
                                "player_name": f"Player_{last_player_with_ball}",
                                "team_name": f"Team {start_player_team}",
                                "x": pass_start_info["position"][0],
                                "y": pass_start_info["position"][1],
                                "end_x": ball_pos_transformed[0],
                                "end_y": ball_pos_transformed[1],
                                "minute": int(frame_num / (self.frame_rate * 60)),
                                "second": int((frame_num / self.frame_rate) % 60),
                            }
                        )

                if current_player_id != -1:
                    pass_start_info = {
                        "frame": frame_num,
                        "position": ball_pos_transformed,
                    }
                last_player_with_ball = current_player_id

            return pd.DataFrame(events)

    fps = video_data["fps"]
    frames = video_data["frames"]
    tracks = video_data["tracks"]
    player_assigner = video_data["player_assigner"]

    commentary_engine = ImprovedCommentaryEngine(fps=fps)
    ticker = RealTimeTicker(fps=fps)

    print("Stage 4: Detecting events for commentary context...")
    event_detector = EventDetector()
    events_df = event_detector.detect_events(tracks)
    print(f"Detected {len(events_df)} events for commentary context")

    print("Stage 5: Tracking ball possession and generating all commentary...")
    team_ball_control = []
    ticker_history = []
    gemini_history = []

    for frame_num, frame in enumerate(frames):
        player_track = tracks["players"][frame_num]
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")

        for player_id in tracks["players"][frame_num]:
            tracks["players"][frame_num][player_id]["has_ball"] = False

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

        ticker_history.append(ticker.update(tracks, frame_num))
        commentary_engine.update_with_context(frame, tracks, frame_num, events_df)
        gemini_history.append(commentary_engine.latest_commentary)

        if frame_num % 100 == 0:
            print(f"Commentary progress: {frame_num}/{len(frames)} frames")

    video_data["commentary_engine"] = commentary_engine
    video_data["team_ball_control"] = team_ball_control
    video_data["ticker_history"] = ticker_history
    video_data["gemini_history"] = gemini_history
    return video_data
