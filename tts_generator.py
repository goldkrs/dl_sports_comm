"""
tts_generator.py — Text-to-speech generation for the Football-Comment pipeline.

Audio quality fixes applied:
  - Segments that are too long for their time window are TRUNCATED (not
    time-compressed).  np.interp compression was creating aliasing noise,
    especially noticeable in the final seconds of the video.
  - total_samples includes a 2-second tail buffer so the last segment always
    has room to play beyond the exact video end without hard clipping.
  - A short fade-out is applied to the final 0.5 s of the mixed audio to
    prevent a harsh stop at the end.
"""

import numpy as np
import soundfile as sf

from segment_extractor import extract_segments
from text_aggregator import stabilize_commentary_timeline

KOKORO_LANG_CODE = "a"
KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24000
KOKORO_BASE_SPEED = 1.2

# Extra silence (seconds) appended after the last video frame so the final
# commentary segment is not cut off mid-sentence.
TAIL_BUFFER_SECONDS = 2.0

# Length of the fade-out applied to the very end of the final mix (seconds).
FADE_OUT_SECONDS = 0.5

_pipeline_cache = {}


def get_pipeline(lang_code="a"):
    from kokoro import KPipeline

    if lang_code not in _pipeline_cache:
        _pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
    return _pipeline_cache[lang_code]


def _apply_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply a linear fade-out to the last `fade_samples` samples."""
    fade_samples = min(fade_samples, len(audio))
    if fade_samples <= 0:
        return audio
    fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    audio[-fade_samples:] *= fade
    return audio


def generate_full_commentary_audio(history, fps, output_path):
    stabilized_history = stabilize_commentary_timeline(history, fps)
    segments = extract_segments(stabilized_history, fps)
    if not segments:
        return None

    # Add a tail buffer so the last segment is not squeezed by the hard
    # video-frame boundary.
    video_duration_sec = len(history) / fps
    total_samples = max(
        int((video_duration_sec + TAIL_BUFFER_SECONDS) * KOKORO_SAMPLE_RATE), 1
    )
    final_audio = np.zeros(total_samples, dtype=np.float32)
    pipeline = get_pipeline(KOKORO_LANG_CODE)

    for text, start_time_sec, end_time_sec in segments:
        if not text or not text.strip():
            continue

        # Generate TTS audio for this segment
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

        if start_sample >= final_audio.size:
            continue

        # --- KEY FIX ---
        # If the generated audio is longer than the available window, just
        # TRUNCATE it.  The old code tried to time-compress using np.interp
        # (linear sample interpolation), which has no anti-aliasing filter and
        # produces audible aliasing noise — exactly what caused the distortion
        # in the final seconds of the video.
        available_samples = final_audio.size - start_sample
        if segment_audio.size > available_samples:
            segment_audio = segment_audio[:available_samples]

        end_sample = start_sample + segment_audio.size
        final_audio[start_sample:end_sample] = segment_audio

    # Fade out the tail so the audio ends cleanly instead of hard-stopping
    fade_samples = int(FADE_OUT_SECONDS * KOKORO_SAMPLE_RATE)
    final_audio = _apply_fade_out(final_audio, fade_samples)

    final_audio = np.clip(final_audio, -1.0, 1.0)
    sf.write(output_path, final_audio, KOKORO_SAMPLE_RATE)
    return output_path
