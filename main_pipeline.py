from audio_video_merge import merge_audio_video
from model import model
from output_writer import output_writer
from postprocess import postprocess
from preprocess import preprocess
from tts_generator import generate_full_commentary_audio
from video_loader import video_loader


def process_video(
    input_video_path="CityUtdR.mp4",
    stub_path="tracks_stub.pkl",
    output_video_path="final_analysis_video-gemini.mp4",
):
    video_data = video_loader(input_video_path, stub_path, output_video_path)
    if video_data is None:
        return None
    video_data = preprocess(video_data)
    video_data = model(video_data)
    video_data = postprocess(video_data)
    commentary_audio_path = output_video_path.replace(".mp4", "_commentary_audio.wav")
    video_data["commentary_audio_path"] = generate_full_commentary_audio(
        video_data["display_commentary"], video_data["fps"], commentary_audio_path
    )
    video_data = output_writer(video_data)
    final_output_path = output_video_path
    if video_data["commentary_audio_path"]:
        final_output_path = output_video_path.replace(".mp4", "_final.mp4")
        final_output_path = merge_audio_video(
            output_video_path,
            video_data["commentary_audio_path"],
            final_output_path,
        )
    commentary_path = output_video_path.replace(".mp4", "_commentary.txt")
    cleaned_commentary = []
    last_comment = None
    for comment in video_data["display_commentary"]:
        if not isinstance(comment, str):
            continue
        comment = comment.strip()
        if not comment:
            continue
        if comment == last_comment:
            continue
        cleaned_commentary.append(comment)
        last_comment = comment
    with open(commentary_path, "w", encoding="utf-8") as commentary_file:
        commentary_file.write("\n".join(cleaned_commentary))
    video_data["commentary_path"] = commentary_path
    video_data["final_output_path"] = final_output_path
    return video_data


def main():
    return process_video()


if __name__ == "__main__":
    main()
