def merge_audio_video(video_path, audio_path, output_path):
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path,
        ],
        check=True,
    )
    return output_path
