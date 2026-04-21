def extract_segments(history, fps):
    segments = []
    current_text = None
    start_frame = None

    for frame_index, text in enumerate(history):
        if not isinstance(text, str):
            text = ""
        text = text.strip()

        if text == "Match analysis is starting...":
            text = ""

        if text != current_text:
            if current_text:
                start_time_sec = start_frame / fps
                end_time_sec = frame_index / fps
                segments.append((current_text, start_time_sec, end_time_sec))
            current_text = text
            start_frame = frame_index

    if current_text:
        start_time_sec = start_frame / fps
        end_time_sec = len(history) / fps
        segments.append((current_text, start_time_sec, end_time_sec))

    return [segment for segment in segments if segment[0]]
