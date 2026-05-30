"""
commentary_engine.py — Commentary generation classes for the Football-Comment system.
Defined at module level for testability and reuse.
"""

import os
from collections import deque

import pandas as pd


# ---------------------------------------------------------------------------
# Gemini-powered commentary engine
# ---------------------------------------------------------------------------

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

        print("[INFO] Initializing Gemini Commentary Engine...")
        try:
            from google import genai
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env"
                )
            self._genai_client = genai.Client(api_key=api_key)
            # Model configurable via GEMINI_MODEL in .env
            # Default: gemini-2.5-flash-preview-05-20 (latest flash preview)
            self._genai_model = (
                os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-preview-05-20"
            )
            # Quick probe — fall back to stable model if preview not found
            self._genai_model = self._probe_model(self._genai_client, self._genai_model)
            print(f"[INFO] Gemini model ready: {self._genai_model}")
        except Exception as e:
            self._genai_client = None
            self._genai_model = None
            print(f"[WARN] Gemini init failed: {e}")
            print("   Commentary will use rule-based fallback.")

        # Possession sustain tracking
        self._last_possessing_team: int | None = None
        self._possession_start_frame: int = 0
        self._last_commentary_words: set[str] = set()

        # Goal / celebration detection
        self._ball_missing_streak: int = 0   # consecutive frames ball not detected
        self._celebration_active: bool = False
        self._celebration_cooldown: int = 0  # frames to hold celebration commentary
        self._low_velocity_streak: int = 0
    # ------------------------------------------------------------------ #
    # Model probe (auto-fallback)                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _probe_model(client, model_name: str) -> str:
        """Test-call the model with a minimal prompt.

        If the model returns 404 (not found / not available in this region),
        automatically fall back to the stable gemini-2.5-flash model so
        commentary always works even when a preview model name is wrong.
        """
        _FALLBACK = "gemini-2.5-flash"
        try:
            client.models.generate_content(
                model=model_name, contents="ping"
            )
            return model_name
        except Exception as e:
            err = str(e)
            if "404" in err or "NOT_FOUND" in err:
                print(
                    f"[WARN] Model '{model_name}' not available "
                    f"(404). Falling back to '{_FALLBACK}'."
                )
                return _FALLBACK
            # Any other error (auth, network) — still try the requested model
            return model_name

    # ------------------------------------------------------------------ #
    # Public update method                                                 #
    # ------------------------------------------------------------------ #

    def update_with_context(
        self,
        frame,
        tracks_data,
        frame_num,
        events_data=None,
        smoothed_team: int = 0,
    ):
        # Update ball-missing streak before extracting context
        ball_detected = 1 in tracks_data["ball"][frame_num]
        if ball_detected:
            self._ball_missing_streak = 0
        else:
            self._ball_missing_streak += 1

        # Tick down celebration cooldown
        if self._celebration_cooldown > 0:
            self._celebration_cooldown -= 1

        game_context = self._extract_game_context(
            tracks_data, frame_num, events_data, smoothed_team
        )

        self.match_context["recent_events"].append(game_context)
        if len(self.match_context["recent_events"]) > 10:
            self.match_context["recent_events"].pop(0)

        self.frame_buffer.append(frame)

        if len(self.frame_buffer) == self.clip_length_frames:
            special = game_context.get("special_event")
            is_celebration = special in ("goal_celebration", "potential_goal")

            print(
                f"Generating {'GOAL' if is_celebration else 'tactical'} summary..."
            )
            new_comment = self._generate_contextual_commentary(game_context)
            if new_comment:
                # Celebration commentary always overrides dedup
                if is_celebration or not self._is_duplicate(new_comment):
                    self.latest_commentary = new_comment
                    self._last_commentary_words = self._key_words(new_comment)
                    if is_celebration:
                        # Hold celebration text for 10 seconds
                        self._celebration_cooldown = int(self.fps * 10)
            self.frame_buffer.clear()

    def _extract_game_context(
        self, tracks_data, frame_num, events_data, smoothed_team: int = 0
    ):
        context = {
            "frame_num": frame_num,
            "timestamp": (
                f"{int(frame_num / (self.fps * 60))}:"
                f"{int((frame_num / self.fps) % 60):02d}"
            ),
            "players_detected": len(tracks_data["players"][frame_num]),
            "ball_detected": 1 in tracks_data["ball"][frame_num],
            "possession": None,
            "pitch_zone": None,
            "possession_contested": False,
            "special_event": None,
            "recent_events": [],
        }

        for player_id, player_info in tracks_data["players"][frame_num].items():
            if player_info.get("has_ball", False):
                context["possession"] = (
                    f"Player {player_id} (Team {player_info.get('team', 'Unknown')})"
                )
                break

        # --- Special event detection (goal / celebration) ---
        context["special_event"] = self._detect_special_event(
            tracks_data, frame_num
        )

        # --- Possession sustain tracking ---
        if smoothed_team and smoothed_team != self._last_possessing_team:
            self._last_possessing_team = smoothed_team
            self._possession_start_frame = frame_num
        sustained_frames = frame_num - self._possession_start_frame
        sustained_seconds = sustained_frames / self.fps
        context["possession_contested"] = (
            smoothed_team == 0 or sustained_seconds < 1.5
        )
        context["possession_sustained_seconds"] = sustained_seconds

        # --- Pitch zone (calibrated field-space only) ---
        ball_entry = tracks_data["ball"][frame_num].get(1, {})
        field_pos = ball_entry.get("position_transformed")
        context["pitch_zone"] = self._get_pitch_zone(field_pos) if field_pos else None

        if events_data is not None and not events_data.empty:
            recent = events_data[
                (events_data["minute"] * 60 + events_data["second"])
                >= (frame_num / self.fps - 10)
            ].tail(3)
            context["recent_events"] = recent.to_dict("records")

        return context

    # ------------------------------------------------------------------ #
    # Goal / Celebration detection                                         #
    # ------------------------------------------------------------------ #

    # Thresholds
    _BALL_MISSING_GOAL_SEC  = 1.5   # ball absent this long → possible goal
    _CLUSTER_RADIUS_PX      = 220   # pixel radius to count as "clustered"
    _CLUSTER_FRACTION       = 0.55  # fraction of players that must be in cluster
    _LOW_SPEED_KMH          = 5.0   # speed below this = standing/celebrating
    _LOW_SPEED_FRACTION     = 0.65  # fraction of players that must be slow

    def _detect_special_event(self, tracks_data, frame_num):
        """Heuristic goal/celebration detector and Dead Ball state machine.

        Returns one of: 'goal_celebration' | 'potential_goal' | 'spatial_anomaly: dead_ball_celebration' | None
        """
        player_tracks = tracks_data["players"][frame_num]
        ball_entry    = tracks_data["ball"][frame_num].get(1, {})

        ball_missing_long = (
            self._ball_missing_streak >= int(self.fps * self._BALL_MISSING_GOAL_SEC)
        )

        # Calibrated goal-zone check (only when position_transformed available)
        in_goal_zone = False
        field_pos = ball_entry.get("position_transformed")
        if field_pos:
            try:
                bx, by = float(field_pos[0]), float(field_pos[1])
                length = 52.5
                # Ball within 1 m behind either goal line and central width
                in_goal_zone = (
                    (bx < 0 or bx > length) and 12.0 < by < 22.0
                )
            except (TypeError, ValueError):
                pass

        if in_goal_zone:
            return "potential_goal"

        # Check for clustering and low velocity team by team
        for team_id in [1, 2]:
            clustered = self._check_player_clustering(player_tracks, target_team_id=team_id)
            self._update_velocity_streak(player_tracks, target_team_id=team_id)
            streak_triggered = self._low_velocity_streak >= int(self.fps * 4)

            if ball_missing_long and clustered and streak_triggered:
                # If we are strictly in the goal zone context we could say goal_celebration
                # but "spatial_anomaly" is a safer catch-all that instructs the LLM.
                print(f"[STATE TRIGGER] spatial_anomaly: dead_ball_celebration detected for team {team_id}.")
                return "spatial_anomaly: dead_ball_celebration"

        return None

    def _check_player_clustering(self, player_tracks, target_team_id) -> bool:
        """
        Calculates spatial clustering exclusively for players on the same team.
        Uses metric space (position_transformed) if available.
        """
        positions = []
        for info in player_tracks.values():
            if info.get("team") == target_team_id:
                # Prioritize metric space (meters) over pixel space
                pos = info.get("position_transformed") or info.get("position")
                if pos:
                    positions.append((float(pos[0]), float(pos[1])))
                    
        if len(positions) < 5:
            return False
            
        # Calculate group centroid
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        
        # Define radius dynamically: 3.0 meters for transformed pitch, ~220px for raw frame
        # Check if the first player has transformed positions
        first_info = next(iter(player_tracks.values()))
        is_transformed = "position_transformed" in first_info
        radius_threshold = 3.0 if is_transformed else self._CLUSTER_RADIUS_PX
        
        close_count = sum(
            1 for p in positions 
            if ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 < radius_threshold
        )
        return (close_count / len(positions)) >= self._CLUSTER_FRACTION

    def _update_velocity_streak(self, player_tracks, target_team_id):
        """Temporal State Machine: Update streak if team average speed < 3 km/h"""
        speeds = [
            info["speed"] for info in player_tracks.values()
            if info.get("team") == target_team_id and "speed" in info
        ]
        if not speeds:
            return

        avg_speed = sum(speeds) / len(speeds)
        # Check if average velocity drops below 3 km/h
        if avg_speed < 3.0:
            self._low_velocity_streak += 1
        else:
            self._low_velocity_streak = 0  # Reset if they start running again

    def _get_pitch_zone(self, field_pos):
        """Map calibrated field-space coordinates to a pitch zone label.

        Only called when position_transformed is available (homography-mapped).
        Returns None instead of guessing from pixel coordinates — pixel x is
        unreliable because camera panning moves zones relative to the frame.
        """
        if not field_pos:
            return None
        try:
            x = float(field_pos[0])
            y = float(field_pos[1])
        except (TypeError, IndexError):
            return None

        length = 52.5  # metres in the visible pitch section
        if x < length * 0.33:
            return "defensive third"
        if x < length * 0.66:
            return "midfield"
        # Attacking third — check for penalty area (last ~16.5 m, central width)
        if x > length * 0.68 and 7.0 < y < 27.0:
            return "penalty area"
        return "attacking third"

    def _generate_contextual_commentary(self, game_context):
        """Call Gemini, falling back to rule-based text on any failure."""
        if self._genai_client is None:
            return self._generate_fallback_commentary(game_context)

        try:
            prompt = self._create_detailed_prompt(game_context)
            response = self._genai_client.models.generate_content(
                model=self._genai_model,
                contents=prompt,
            )
            text = response.text.strip() if response.text else ""
            if text:
                return text
        except Exception as e:
            print(f"[WARN] Gemini generation failed: {e}. Falling back.")

        return self._generate_fallback_commentary(game_context)

    def _create_detailed_prompt(self, context):
        # --- Route to a goal-specific prompt if celebration detected ---
        special = context.get("special_event")

        if special == "spatial_anomaly: dead_ball_celebration":
            return (
                "You are an expert professional football commentator.\n\n"
                "THE SYSTEM HAS DETECTED A SPATIAL ANOMALY:\n"
                "- The ball has been untracked/absent from play.\n"
                "- A dense cluster of players from the same team has formed.\n"
                "- The team average speed has fallen under 3 km/h for > 4 seconds.\n"
                "- Note: The ball did not cross the goal line definitively, meaning this "
                "could be a goal celebration, a prolonged injury break, or a team huddle.\n\n"
                "OUTPUT RULES:\n"
                "1. Write ONE natural commentary sentence noting the break in play or team huddle (max 20 words).\n"
                "2. DO NOT describe passing sequences, tactical build-ups, or active ball movement.\n"
                "3. Never mention coordinates, metrics, or player numbers.\n\n"
                "Your commentary:"
            )

        if special == "goal_celebration":
            return (
                "You are an excited professional football commentator.\n\n"
                "THE SYSTEM HAS DETECTED:\n"
                "- The ball has disappeared from tracking for over 1.5 seconds\n"
                "- Most players are clustered tightly together\n"
                "- Player speeds have dropped significantly (players have stopped running)\n"
                "- This pattern strongly indicates a GOAL has been scored and "
                "players are celebrating.\n\n"
                "OUTPUT RULES:\n"
                "1. Write ONE excited sentence about the goal/celebration (max 20 words).\n"
                "2. Use classic goal commentary language — this is a big moment!\n"
                "   Good: 'The ball hits the back of the net and the players mob the scorer!'\n"
                "   Good: 'What a finish! The team erupts in wild celebration!'\n"
                "   Good: 'GOAL! The players rush together in jubilation!'\n"
                "3. NEVER mention player numbers, team numbers, or coordinates.\n\n"
                "Your commentary:"
            )
        if special == "potential_goal":
            return (
                "You are an excited professional football commentator.\n\n"
                "THE SYSTEM HAS DETECTED:\n"
                "- The ball has entered the goal-zone area behind the goal line\n"
                "- This indicates the ball is in or near the net.\n\n"
                "OUTPUT RULES:\n"
                "1. Write ONE excited sentence about the potential goal (max 20 words).\n"
                "   Good: 'The ball crosses the line! It must be a goal!'\n"
                "   Good: 'It\\'s in! The net bulges and the crowd goes wild!'\n"
                "2. NEVER mention player numbers, team numbers, or coordinates.\n\n"
                "Your commentary:"
            )

        possession = context.get("possession") or "unclear"
        if possession and "Team" in possession:
            try:
                team_part = possession.split("(")[1].rstrip(")")
                possession_str = f"a {team_part} player"
            except IndexError:
                possession_str = possession
        else:
            possession_str = "unknown"

        contested = context.get("possession_contested", False)
        sustained = context.get("possession_sustained_seconds", 0)
        zone = context.get("pitch_zone")
        zone_str = f" in the {zone}" if zone else ""
        events_str = self._format_recent_events(context.get("recent_events", []))
        has_events = events_str != "No recent significant events."

        # When possession is contested / hasn't been sustained, tell Gemini
        # to describe open play rather than attributing possession to anyone.
        if contested:
            possession_line = "- Ball possession: contested / loose ball in open play\n"
        else:
            possession_line = (
                f"- Ball currently held by: {possession_str} "
                f"(held for {sustained:.1f}s)\n"
            )

        return (
            "You are a professional football commentator providing live match commentary.\n\n"
            "WHAT THE SYSTEM HAS DETECTED:\n"
            + possession_line
            + (f"- Ball location: {zone}{zone_str}\n" if zone else "")
            + f"- Number of players visible: {context.get('players_detected', 0)}\n"
            + (f"- Recent detected action: {events_str}\n" if has_events else "")
            + "\nOUTPUT RULES (follow strictly):\n"
            "1. Write ONE single sentence of natural football commentary (max 20 words).\n"
            "2. Include the pitch zone naturally if it is known.\n"
            "   Good: 'The attack builds on the edge of the penalty area.'\n"
            "   Good: 'Both sides contest the loose ball in midfield.'\n"
            "3. If possession is contested, describe it as open play or a battle.\n"
            "4. NEVER say a team 'wins back' or 'regains' possession more than once\n"
            "   in a row for the same team. Vary the description.\n"
            "5. NEVER mention player numbers, team numbers, coordinates, or timestamps.\n"
            "6. NEVER invent goals, fouls, or events not in the detected data.\n"
            "7. Use natural color/side descriptions, not 'Team 1' or 'Team 2'.\n\n"
            "Your commentary:"
        )

    # ------------------------------------------------------------------ #
    # Deduplication helpers                                                #
    # ------------------------------------------------------------------ #

    # Action words that indicate the same semantic event
    _ACTION_WORDS = {
        "interception", "intercepts", "wins", "regains", "possession",
        "clearance", "clears", "pass", "passes", "build", "building",
        "attack", "attacking", "defend", "defending", "press", "pressing",
    }

    def _key_words(self, text: str) -> set[str]:
        """Extract action words from commentary text for dedup comparison."""
        words = set(w.strip('.,!').lower() for w in text.split())
        return words & self._ACTION_WORDS

    def _is_duplicate(self, new_text: str) -> bool:
        """Return True if new_text is semantically redundant vs the last line.

        Redundant = shares ≥2 key action words with the previous commentary.
        This prevents consecutive 'wins possession' / 'interception' lines.
        """
        if not self._last_commentary_words:
            return False
        overlap = self._key_words(new_text) & self._last_commentary_words
        return len(overlap) >= 2

    def _format_recent_events(self, events):
        if not events:
            return "No recent significant events."
        formatted = []
        for event in events[-3:]:
            if isinstance(event, dict):
                event_type = event.get("type_name", "Unknown")
                team = event.get("team_name", "Unknown Team")
                formatted.append(f"{event_type} by {team}")
        return ", ".join(formatted) if formatted else "No recent significant events."

    def _generate_fallback_commentary(self, context):
        if context.get("possession"):
            return f"Play continues with {context['possession']} in possession."
        return "The match continues with both teams looking for opportunities."


# ---------------------------------------------------------------------------
# Real-time ticker (frame-level rule-based commentary)
# ---------------------------------------------------------------------------

class RealTimeTicker:
    """Generates simple real-time text commentary for each frame."""

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
                f"Pass from Player {self.last_player_id} "
                f"to Player {current_player_id}."
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
                    f"Player {current_player_id} "
                    f"(Team {current_team_id}) on the ball."
                )
            else:
                self.ticker_text = "Ball is loose."

        if current_player_id != -1:
            self.last_player_id = current_player_id
            self.last_team_id = current_team_id
        else:
            self.last_player_id = -1

        return self.ticker_text


# ---------------------------------------------------------------------------
# Event Detector (pass inference from possession sequence)
# ---------------------------------------------------------------------------

class EventDetector:
    def __init__(self, frame_rate=24):
        self.frame_rate = frame_rate

    def detect_events(self, tracks, player_ball_assigner):
        """Scan possession sequence to infer pass, interception, and clearance events.

        - Same-team possession change  → Pass
        - Different-team possession change → Interception (or Clearance if in def. third)
        Uses position_transformed when available, falls back to raw pixel position.
        """
        ball_possession_log = []
        for frame_num in range(len(tracks["players"])):
            player_track = tracks["players"][frame_num]
            ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox")
            assigned = (
                player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)
                if ball_bbox
                else -1
            )
            ball_possession_log.append(assigned)

        events = []
        last_player = -1
        last_team = None
        pass_start_info = {}

        for frame_num, current_player_id in enumerate(ball_possession_log):
            ball_entry = tracks["ball"][frame_num].get(1, {})
            ball_pos = (
                ball_entry.get("position_transformed")
                or ball_entry.get("position")
            )
            if not ball_pos:
                continue

            is_possession_change = (
                current_player_id != last_player
                and last_player != -1
                and current_player_id != -1
            )

            if is_possession_change and pass_start_info:
                start_team = (
                    tracks["players"][pass_start_info["frame"]]
                    .get(last_player, {})
                    .get("team")
                )
                end_team = (
                    tracks["players"][frame_num]
                    .get(current_player_id, {})
                    .get("team")
                )

                if start_team is not None and end_team is not None:
                    if start_team == end_team:
                        event_type = "Pass"
                    else:
                        # Distinguish interception from clearance:
                        # clearance = possession won in the defensive half of the
                        # tracked section (low x values when using transformed pos)
                        ball_x = float(ball_pos[0]) if ball_pos else 0
                        is_defensive = ball_x < 17.5  # approx defensive third
                        event_type = "Clearance" if is_defensive else "Interception"

                    events.append(
                        {
                            "type_name": event_type,
                            "player_name": f"Player_{last_player}",
                            "team_name": f"Team {start_team}",
                            "minute": int(frame_num / (self.frame_rate * 60)),
                            "second": int((frame_num / self.frame_rate) % 60),
                        }
                    )

            if current_player_id != -1:
                pass_start_info = {"frame": frame_num, "position": ball_pos}
                last_team = (
                    tracks["players"][frame_num]
                    .get(current_player_id, {})
                    .get("team")
                )
            last_player = current_player_id

        print(f"  Detected {len(events)} events (passes, interceptions, clearances)")
        return pd.DataFrame(events)
