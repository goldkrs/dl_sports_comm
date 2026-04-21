KOKORO_LANG_CODE = "a"
KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24000
KOKORO_BASE_SPEED = 1.2

_pipeline_cache = {}


def get_pipeline(lang_code="a"):
    from kokoro import KPipeline

    if lang_code not in _pipeline_cache:
        _pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
    return _pipeline_cache[lang_code]


def generate_full_commentary_audio(history, fps, output_path):
    import numpy as np
    import soundfile as sf

    from segment_extractor import extract_segments
    from text_aggregator import stabilize_commentary_timeline

    stabilized_history = stabilize_commentary_timeline(history, fps)
    segments = extract_segments(stabilized_history, fps)
    if not segments:
        return None

    total_samples = max(int((len(history) / fps) * KOKORO_SAMPLE_RATE), 1)
    final_audio = np.zeros(total_samples, dtype=np.float32)
    pipeline = get_pipeline(KOKORO_LANG_CODE)

    for text, start_time_sec, end_time_sec in segments:
        if not text or not text.strip():
            continue

        chunks = []
        generator = pipeline(
            text,
            voice=KOKORO_VOICE,
            speed=KOKORO_BASE_SPEED,
            split_pattern=r"\n+",
        )

        for _, _, audio in generator:
            audio_np = np.asarray(audio, dtype=np.float32)
            if audio_np.size > 0:
                chunks.append(audio_np)

        if not chunks:
            continue

        segment_audio = np.concatenate(chunks)
        start_sample = int(start_time_sec * KOKORO_SAMPLE_RATE)
        max_segment_samples = max(
            int((end_time_sec - start_time_sec) * KOKORO_SAMPLE_RATE), 0
        )
        if max_segment_samples <= 0:
            continue

        if segment_audio.size > max_segment_samples:
            original_positions = np.linspace(0, 1, num=segment_audio.size, endpoint=False)
            target_positions = np.linspace(0, 1, num=max_segment_samples, endpoint=False)
            segment_audio = np.interp(target_positions, original_positions, segment_audio)

        trimmed_segment_audio = segment_audio[:max_segment_samples]
        end_sample = min(start_sample + trimmed_segment_audio.size, final_audio.size)
        segment_length = end_sample - start_sample
        if segment_length > 0:
            final_audio[start_sample:end_sample] = trimmed_segment_audio[:segment_length]

    final_audio = np.clip(final_audio, -1.0, 1.0)
    sf.write(output_path, final_audio, KOKORO_SAMPLE_RATE)
    return output_path
