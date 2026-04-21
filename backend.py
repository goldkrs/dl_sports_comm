from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from main_pipeline import process_video


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    file_suffix = Path(file.filename).suffix
    file_stem = Path(file.filename).stem
    file_id = uuid.uuid4().hex
    input_video_path = UPLOAD_DIR / f"{file_stem}_{file_id}{file_suffix}"
    stub_path = OUTPUT_DIR / f"{file_stem}_{file_id}_tracks_stub.pkl"
    output_video_path = OUTPUT_DIR / f"{file_stem}_{file_id}_processed.mp4"

    with input_video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_data = process_video(
        input_video_path=str(input_video_path),
        stub_path=str(stub_path),
        output_video_path=str(output_video_path),
    )

    if video_data is None:
        raise HTTPException(status_code=400, detail="Video processing failed.")

    final_output_path = Path(video_data.get("final_output_path", output_video_path))

    return {
        "output_video_path": str(final_output_path),
        "output_video_url": f"http://localhost:8000/output/{final_output_path.name}",
    }
