"""
audio_video_merge.py — Merge a WAV audio track into a processed MP4 via ffmpeg.
"""

import subprocess

from video_loader import check_ffmpeg


def merge_audio_video(video_path, audio_path, output_path):
    check_ffmpeg()
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ],
        check=True,
    )
    return output_path
