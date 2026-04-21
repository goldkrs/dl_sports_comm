def video_loader(
    INPUT_VIDEO_PATH="CityUtdR.mp4",
    STUB_PATH="tracks_stub.pkl",
    OUTPUT_VIDEO_PATH="final_analysis_video-gemini.mp4",
):
    import cv2
    import os
    import subprocess

    def read_video(video_path):
        """Reads a video file and returns a list of its frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def save_video(output_video_frames, output_video_path):
        """Saves a list of frames as a video file."""
        if not output_video_frames:
            print("No frames to save.")
            return

        temp_output_video_path = output_video_path.replace(".mp4", "_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            temp_output_video_path,
            fourcc,
            fps,
            (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
        )
        for frame in output_video_frames:
            out.write(frame)
        out.release()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_output_video_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                output_video_path,
            ],
            check=True,
        )
        if os.path.exists(temp_output_video_path):
            os.remove(temp_output_video_path)

    frames = read_video(INPUT_VIDEO_PATH)
    if not frames:
        print("Video file not found or could not be read. Check the path.")
        return None

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.release()

    return {
        "INPUT_VIDEO_PATH": INPUT_VIDEO_PATH,
        "STUB_PATH": STUB_PATH,
        "OUTPUT_VIDEO_PATH": OUTPUT_VIDEO_PATH,
        "frames": frames,
        "fps": fps,
        "save_video": save_video,
    }
