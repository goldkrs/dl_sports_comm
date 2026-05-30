def output_writer(video_data):
    from video_loader import save_video

    output_frames = video_data["output_frames"]
    OUTPUT_VIDEO_PATH = video_data["OUTPUT_VIDEO_PATH"]
    fps = video_data["fps"]

    save_video(output_frames, OUTPUT_VIDEO_PATH, fps)

    print("\n" + "=" * 50)
    print("MATCH ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"[INFO] Video saved to: {OUTPUT_VIDEO_PATH}")
    return video_data
