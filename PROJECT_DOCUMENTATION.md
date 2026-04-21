# Football-Comment: AI-Powered Football Analysis and Commentary System

## Abstract

Football-Comment is an end-to-end AI pipeline that accepts a football match video, analyzes the visual gameplay frame by frame, and produces an annotated output video with tactical text commentary, possession insights, player overlays, speed indicators, and synthesized audio commentary. The project combines computer vision, player tracking, team clustering, ball-possession inference, event detection, text generation, text-to-speech synthesis, and web-based delivery into a single workflow. Its main objective is to turn raw match footage into a richer analytical viewing experience with minimal manual intervention.

## Working

The system starts with a web interface built in React, where a user uploads a match video. The frontend sends the file to a FastAPI backend through the `/upload-video` endpoint. The backend stores the uploaded file, creates output paths, and triggers the main processing pipeline through `process_video()` in [main_pipeline.py](./main_pipeline.py).

The first stage is video loading. In [video_loader.py](./video_loader.py), the input video is decoded into frames using OpenCV, and the original FPS is extracted. A helper is also prepared to write the processed frames back into a compressed MP4 using FFmpeg.

The second stage is preprocessing and computer vision analysis in [preprocess.py](./preprocess.py). A YOLOv8x model is used to detect players and the ball. ByteTrack is then applied to maintain identities across frames. Each player crop is passed through EasyOCR to estimate jersey numbers. Ball positions are interpolated when detections are missing. The pipeline also estimates camera movement using optical flow, compensates tracked positions, and projects player and ball coordinates into a transformed field-space view using perspective transformation. Team assignment is performed through KMeans clustering on jersey colors, and player speed and cumulative movement distance are estimated from transformed positions.

The third stage is match understanding and commentary generation in [model.py](./model.py). The pipeline determines which player is most likely in possession of the ball by measuring proximity between the ball and player foot positions. It builds two commentary streams:

- A rule-based real-time ticker that reports possession changes and short pass events.
- A higher-level commentary engine intended for tactical summaries using Gemini context handling.

An event detector also scans the possession sequence to infer pass events and convert them into structured records. These records help maintain contextual awareness during commentary generation.

The fourth stage is postprocessing in [postprocess.py](./postprocess.py). The rule-based ticker and higher-level commentary outputs are merged into a stable timeline. The system then renders overlays for player markers, jersey labels, ball indicators, possession percentages, commentary banners, and player speed. This creates the final annotated frame sequence.

The fifth stage is media generation. In [tts_generator.py](./tts_generator.py), the stabilized commentary timeline is segmented and converted into speech using the Kokoro text-to-speech pipeline. The generated narration is aligned to the commentary time windows and saved as audio. In [audio_video_merge.py](./audio_video_merge.py), FFmpeg merges the commentary audio with the processed video to produce a final MP4. A cleaned text commentary file is also exported for reference.

The final output of the project is:

- A processed football video with visual analytics overlays.
- A narrated version of the processed video with synthesized commentary.
- A text file containing the cleaned commentary timeline.

## Tech Stack

### Backend and Application Layer

- Python
- FastAPI
- Uvicorn-compatible API serving pattern
- Pathlib, UUID, and standard Python file utilities

### Computer Vision and Tracking

- OpenCV
- Ultralytics YOLOv8x
- Supervision
- ByteTrack
- NumPy
- Pandas

### Machine Learning and Analysis

- Scikit-learn KMeans for team color clustering
- Optical flow for camera motion estimation
- Perspective transformation for field-space coordinate mapping
- Rule-based possession and pass inference

### OCR and Commentary

- EasyOCR for jersey number recognition
- Google Generative AI client for Gemini integration
- Kokoro TTS for commentary voice synthesis

### Media Processing

- FFmpeg for MP4 encoding and audio-video merging
- SoundFile for writing generated audio

### Frontend

- React
- Vite
- JavaScript

## Challenges

### 1. Real-time-quality analysis from offline video

The project attempts to simulate live football commentary from recorded footage. This is challenging because the pipeline must maintain frame-level consistency across detection, tracking, possession estimation, event extraction, and overlay rendering.

### 2. Ball tracking instability

The football is a small and fast-moving object, and it is frequently occluded. The project addresses this partly through interpolation, but precise ball localization remains difficult in crowded or low-quality footage.

### 3. Robust player identification

Tracking identities across frames is hard when players overlap, change direction rapidly, or leave and re-enter the scene. Jersey OCR adds useful metadata, but recognition accuracy depends heavily on crop quality, resolution, and visibility.

### 4. Team classification from appearance

The system uses unsupervised clustering over jersey colors. This can fail when kits are visually similar, lighting changes across the pitch, shadows are strong, or non-player regions affect the crop.

### 5. Camera-motion compensation and field transformation

Broadcast football footage contains pans, zooms, and perspective shifts. Estimating camera movement and mapping image coordinates to pitch coordinates is non-trivial, and errors here propagate into speed, distance, and tactical interpretations.

### 6. Commentary quality and grounding

Although the architecture includes a Gemini-based commentary engine, the current implementation falls back to a simpler possession-oriented summary. This means the project structure supports richer language generation, but the delivered commentary is still closer to lightweight tactical narration than full expert analysis.

### 7. System performance

The use of YOLOv8x, OCR, tracking, optical flow, clustering, frame rendering, TTS generation, and FFmpeg makes the pipeline computationally heavy. Processing long videos can become slow and resource-intensive, especially on machines without strong GPU support.

### 8. Production readiness concerns

The code currently includes implementation shortcuts that would need cleanup for a production deployment. For example, secrets should not be hardcoded, GPU assumptions may not hold across environments, and external tools such as FFmpeg and model dependencies must be installed correctly.

## Conclusion

Football-Comment is a strong applied AI project that demonstrates how multiple vision and language components can be combined into a practical sports-analysis system. Its main strength is the integration of detection, tracking, team inference, event reasoning, commentary generation, and TTS into one coherent pipeline accessible through a web app.

The project is especially valuable as a prototype for automated football analytics, smart broadcasting overlays, and assistive commentary generation. While accuracy, robustness, and deployment hardening still need improvement, the current implementation already shows a complete and technically meaningful workflow for turning raw football footage into an enriched analytical media product.
