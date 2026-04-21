def output_writer(video_data):
    save_video = video_data["save_video"]
    output_frames = video_data["output_frames"]
    OUTPUT_VIDEO_PATH = video_data["OUTPUT_VIDEO_PATH"]

    save_video(output_frames, OUTPUT_VIDEO_PATH)

    print("\n" + "=" * 50)
    print("MATCH ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"✅ Video saved to: {OUTPUT_VIDEO_PATH}")
    return video_data
