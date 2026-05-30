import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const BASE_URL = "http://localhost:8000";
const UPLOAD_URL = `${BASE_URL}/upload-video`;
const POLL_INTERVAL_MS = 5000;
const POLL_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes max

function formatFileSize(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

// Status display helpers
const STATUS_LABEL = {
  queued:  "Queued — waiting to start",
  running: "Running pipeline…",
  done:    "Complete",
  error:   "Failed",
};

export default function App() {
  const inputRef      = useRef(null);
  const pollTimerRef  = useRef(null);
  const startTimeRef  = useRef(null);

  const [file,      setFile]      = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [jobId,     setJobId]     = useState(null);
  const [jobStatus, setJobStatus] = useState(null); // queued | running | done | error
  const [progress,  setProgress]  = useState("");
  const [videoUrl,  setVideoUrl]  = useState("");
  const [error,     setError]     = useState("");

  const fileDetails = useMemo(() => {
    if (!file) return null;
    return { name: file.name, size: formatFileSize(file.size), type: file.type || "Video file" };
  }, [file]);

  // ------------------------------------------------------------------
  // Polling
  // ------------------------------------------------------------------
  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const pollJob = useCallback(async (id) => {
    // Timeout guard
    if (Date.now() - startTimeRef.current > POLL_TIMEOUT_MS) {
      stopPolling();
      setLoading(false);
      setJobStatus("error");
      setError("Processing timed out after 30 minutes. The video may be too long.");
      return;
    }

    try {
      const res  = await fetch(`${BASE_URL}/job/${id}`);
      const data = await res.json();

      setJobStatus(data.status);
      setProgress(data.progress || "");

      if (data.status === "done") {
        stopPolling();
        setLoading(false);
        setVideoUrl(data.output_video_url);
      } else if (data.status === "error") {
        stopPolling();
        setLoading(false);
        setError("Processing failed on the server. Check the backend logs.");
      }
    } catch {
      // Network hiccup — keep polling
    }
  }, [stopPolling]);

  // Start polling whenever jobId changes
  useEffect(() => {
    if (!jobId) return;
    startTimeRef.current = Date.now();
    pollTimerRef.current = setInterval(() => pollJob(jobId), POLL_INTERVAL_MS);
    pollJob(jobId); // immediate first check
    return stopPolling;
  }, [jobId, pollJob, stopPolling]);

  // ------------------------------------------------------------------
  // Handlers
  // ------------------------------------------------------------------
  const handleFileChange = (selectedFile) => {
    stopPolling();
    setError("");
    setVideoUrl("");
    setJobId(null);
    setJobStatus(null);
    setProgress("");
    setFile(selectedFile || null);
  };

  const handleCancel = () => {
    stopPolling();
    setLoading(false);
    setJobStatus(null);
    setJobId(null);
    setProgress("");
    setError("Processing cancelled.");
  };

  const handleSubmit = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setVideoUrl("");
    setJobId(null);
    setJobStatus("queued");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res  = await fetch(UPLOAD_URL, { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || "Upload failed.");

      setJobId(data.job_id);   // kicks off polling via useEffect
    } catch (err) {
      setLoading(false);
      setJobStatus(null);
      setError(err.message || "Could not reach the backend.");
    }
  };

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------
  const isProcessing = loading && jobStatus !== "done";

  return (
    <main className="app-shell">
      <section className="topbar" aria-label="Application header">
        <div>
          <p className="eyebrow">Football-Comment</p>
          <h1>Match Analysis Studio</h1>
        </div>
        <div className="service-pill">
          <span className="status-dot" />
          Backend: localhost:8000
        </div>
      </section>

      <section className="workspace">
        {/* ---- Upload panel ---- */}
        <div className="upload-panel">
          <div className="panel-heading">
            <p className="eyebrow">Input</p>
            <h2>Upload match footage</h2>
            <p>
              Send a football clip through the AI pipeline to generate overlays,
              possession context, commentary text, and narration.
            </p>
          </div>

          <button
            className="drop-zone"
            type="button"
            id="file-picker-btn"
            onClick={() => inputRef.current?.click()}
            disabled={isProcessing}
          >
            <span className="drop-icon">+</span>
            <span className="drop-title">
              {fileDetails ? fileDetails.name : "Choose a video file"}
            </span>
            <span className="drop-meta">
              {fileDetails
                ? `${fileDetails.size} · ${fileDetails.type}`
                : "MP4, MOV, AVI, or MKV"}
            </span>
          </button>

          <input
            ref={inputRef}
            className="file-input"
            type="file"
            id="video-file-input"
            accept="video/*"
            onChange={(e) => handleFileChange(e.target.files?.[0])}
          />

          <div className="actions">
            <button
              className="primary-button"
              type="button"
              id="run-analysis-btn"
              onClick={handleSubmit}
              disabled={!file || isProcessing}
            >
              {isProcessing ? "Processing…" : "Run analysis"}
            </button>

            {isProcessing ? (
              <button
                className="secondary-button cancel-button"
                type="button"
                id="cancel-btn"
                onClick={handleCancel}
              >
                Cancel
              </button>
            ) : (
              <button
                className="secondary-button"
                type="button"
                id="clear-btn"
                onClick={() => handleFileChange(null)}
                disabled={!file}
              >
                Clear
              </button>
            )}
          </div>

          {error && <p className="alert" role="alert">{error}</p>}
        </div>

        {/* ---- Result panel ---- */}
        <div className="result-panel">
          <div className="panel-heading">
            <p className="eyebrow">Output</p>
            <h2>Processed video</h2>
          </div>

          <div className="preview-frame">
            {isProcessing && (
              <div className="processing-state">
                <div className="progress-bar"><span /></div>
                <p className="status-label">
                  {STATUS_LABEL[jobStatus] ?? "Processing…"}
                </p>
                {progress && <p className="progress-detail">{progress}</p>}
                <p className="progress-note">
                  Processing is computationally heavy — this can take several minutes.
                  The page will update automatically when done.
                </p>
              </div>
            )}

            {!isProcessing && videoUrl && (
              <>
                <video key={videoUrl} className="video-preview" controls>
                  <source src={videoUrl} type="video/mp4" />
                </video>
                <a
                  className="download-link"
                  href={videoUrl}
                  download
                  id="download-btn"
                >
                  ⬇ Download video
                </a>
              </>
            )}

            {!isProcessing && !videoUrl && (
              <div className="empty-state">
                <p>Your analyzed video will appear here.</p>
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
