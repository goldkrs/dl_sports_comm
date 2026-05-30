"""
video_loader.py — Frame extraction and video-save helper.
"""

import os
import shutil
import subprocess

import cv2


def check_ffmpeg():
    """Raise a clear RuntimeError if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required but was not found on PATH. "
            "Install ffmpeg and add it to your system PATH, then restart."
        )


# Maximum frame height to load into RAM.  Frames taller than this are scaled
# down proportionally on read.  1080 means 4K (2160p) becomes 1080p, cutting
# per-frame memory from ~23.7 MB to ~5.9 MB (4× reduction).
MAX_LOAD_HEIGHT = 1080


def read_video(video_path, max_height: int = MAX_LOAD_HEIGHT):
    """Read a video file and return a list of BGR frames.

    Frames are resized to at most `max_height` pixels tall (preserving aspect
    ratio) to avoid OOM on high-resolution footage (e.g. 4K UHD).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if max_height and h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            frame = cv2.resize(frame, (new_w, max_height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if frames:
        h, w = frames[0].shape[:2]
        print(f"[INFO] Loaded {len(frames)} frames at {w}x{h} px")
    return frames


def save_video(output_video_frames, output_video_path, fps):
    """
    Save a list of frames as a browser-compatible MP4.

    Parameters
    ----------
    output_video_frames : list of ndarray
    output_video_path   : str  — final output path (e.g. "out.mp4")
    fps                 : float — frame rate for the output file
    """
    if not output_video_frames:
        print("No frames to save.")
        return

    check_ffmpeg()

    temp_path = output_video_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        temp_path,
        fourcc,
        fps,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", temp_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            output_video_path,
        ],
        check=True,
    )

    if os.path.exists(temp_path):
        os.remove(temp_path)


def video_loader(
    INPUT_VIDEO_PATH="CityUtdR.mp4",
    STUB_PATH="tracks_stub.pkl",
    OUTPUT_VIDEO_PATH="final_analysis_video-gemini.mp4",
):
    check_ffmpeg()

    frames = read_video(INPUT_VIDEO_PATH)
    if not frames:
        print("Video file not found or could not be read. Check the path.")
        return None

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()

    return {
        "INPUT_VIDEO_PATH": INPUT_VIDEO_PATH,
        "STUB_PATH": STUB_PATH,
        "OUTPUT_VIDEO_PATH": OUTPUT_VIDEO_PATH,
        "frames": frames,
        "fps": fps,
    }
