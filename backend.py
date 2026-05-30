"""
backend.py — FastAPI backend for Football-Comment.

Uses a background job queue so the long-running pipeline does NOT block
the HTTP request (fixes the timeout issue with synchronous processing).

Endpoints
---------
POST /upload-video          — Accept upload, start background job, return job_id
GET  /job/{job_id}          — Poll job status: queued | running | done | error
GET  /output/{filename}     — Serve processed video files (static mount)
"""

import asyncio
import json
import logging
import threading
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil

from main_pipeline import process_video


# Suppress the harmless WinError 10054 "connection reset" noise that Windows
# asyncio raises whenever a browser closes a video streaming connection early.
def _silence_connection_reset(loop, context):
    exc = context.get("exception")
    if isinstance(exc, ConnectionResetError):
        return   # browser closed a video stream — completely normal
    loop.default_exception_handler(context)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Football-Comment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.on_event("startup")
async def _install_exception_handler():
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(_silence_connection_reset)

# ---------------------------------------------------------------------------
# In-memory job store  {job_id: {status, output_video_url, error, progress}}
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# Process pool for heavy ML tasks (bypasses GIL).
# max_workers=1 acts as a strict FIFO queue to prevent parallel GPU OOM.
_process_pool = ProcessPoolExecutor(max_workers=1)


def _process_video_worker(input_path: str, stub_path: str, output_path: str, pixel_verts: list):
    """Picklable worker function executed in the separate process."""
    video_data = process_video(
        input_video_path=input_path,
        stub_path=stub_path,
        output_video_path=output_path,
        pixel_verts=pixel_verts,
    )
    if video_data is None:
        return None
    return video_data.get("final_output_path", output_path)


async def _run_pipeline_async(job_id: str, input_path: str, stub_path: str, output_path: str, pixel_verts: list):
    """Async task that awaits the process pool execution."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["progress"] = "Starting pipeline in background process..."

        loop = asyncio.get_running_loop()
        final_output_path = await loop.run_in_executor(
            _process_pool,
            _process_video_worker,
            input_path,
            stub_path,
            output_path,
            pixel_verts,
        )

        if final_output_path is None:
            raise RuntimeError("process_video returned None — check the video file.")

        final_path = Path(final_output_path)
        output_url = f"http://localhost:8000/output/{final_path.name}"

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["output_video_url"] = output_url
            _jobs[job_id]["output_video_path"] = str(final_path)
            _jobs[job_id]["progress"] = "Complete"

    except Exception:
        tb = traceback.format_exc()
        print(f"\n[ERROR] Job {job_id} failed:\n{tb}", flush=True)
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = tb


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    pixel_verts: str = Form(None)
):
    """Accept a video upload and immediately return a job_id for polling."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    parsed_verts = None
    if pixel_verts:
        try:
            parsed_verts = json.loads(pixel_verts)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid pixel_verts JSON format.")

    suffix = Path(file.filename).suffix
    stem = Path(file.filename).stem
    job_id = uuid.uuid4().hex

    input_path = UPLOAD_DIR / f"{stem}_{job_id}{suffix}"
    stub_path = OUTPUT_DIR / f"{stem}_{job_id}_tracks_stub.pkl"
    output_path = OUTPUT_DIR / f"{stem}_{job_id}_processed.mp4"

    with input_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "output_video_url": None,
            "output_video_path": None,
            "error": None,
            "progress": "Queued",
        }

    asyncio.create_task(
        _run_pipeline_async(
            job_id, str(input_path), str(stub_path), str(output_path), parsed_verts
        )
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Poll the status of a processing job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", ""),
    }
    if job["status"] == "done":
        response["output_video_url"] = job["output_video_url"]
        response["output_video_path"] = job["output_video_path"]
    if job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")

    return response
