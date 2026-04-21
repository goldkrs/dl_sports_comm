def clean_commentary(gemini_history):
    cleaned = []
    blocked_sentences = {
        "you are a football commentator",
        "match analysis is starting",
    }

    for comment in gemini_history:
        if not isinstance(comment, str):
            continue

        comment = comment.strip()
        if not comment:
            continue
        normalized = comment.lower().strip(" .!?")
        if normalized in blocked_sentences:
            continue
        cleaned.append(comment)

    return cleaned


def stabilize_commentary_timeline(history, fps, min_hold_seconds=3):
    stabilized = []
    last_text = ""
    hold_frames = max(int(fps * min_hold_seconds), 1)
    remaining_hold_frames = 0

    for comment in history:
        if not isinstance(comment, str):
            comment = ""
        comment = comment.strip()

        normalized = comment.lower().strip(" .!?")
        if normalized in {
            "you are a football commentator",
            "match analysis is starting",
        }:
            comment = ""

        if not last_text:
            last_text = comment
            remaining_hold_frames = hold_frames
        elif comment and comment != last_text and remaining_hold_frames <= 0:
            last_text = comment
            remaining_hold_frames = hold_frames
        elif remaining_hold_frames > 0:
            remaining_hold_frames -= 1

        stabilized.append(last_text if last_text else comment)

    return stabilized
