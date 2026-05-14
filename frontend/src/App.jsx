import { useMemo, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000/upload-video";

function formatFileSize(bytes) {
  if (!bytes) return "0 MB";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState("");
  const [error, setError] = useState("");

  const fileDetails = useMemo(() => {
    if (!file) return null;
    return {
      name: file.name,
      size: formatFileSize(file.size),
      type: file.type || "Video file",
    };
  }, [file]);

  const handleFileChange = (selectedFile) => {
    setError("");
    setVideoUrl("");
    setFile(selectedFile || null);
  };

  const handleSubmit = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setVideoUrl("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Video processing failed.");
      }

      setVideoUrl(data.output_video_url);
    } catch (err) {
      setError(err.message || "Unable to process this video.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="topbar" aria-label="Application status">
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
            onClick={() => inputRef.current?.click()}
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
            accept="video/*"
            onChange={(event) => handleFileChange(event.target.files?.[0])}
          />

          <div className="actions">
            <button
              className="primary-button"
              type="button"
              onClick={handleSubmit}
              disabled={!file || loading}
            >
              {loading ? "Processing video" : "Run analysis"}
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={() => handleFileChange(null)}
              disabled={!file || loading}
            >
              Clear
            </button>
          </div>

          {error && <p className="alert">{error}</p>}
        </div>

        <div className="result-panel">
          <div className="panel-heading">
            <p className="eyebrow">Output</p>
            <h2>Processed video</h2>
          </div>

          <div className="preview-frame">
            {loading && (
              <div className="processing-state">
                <div className="progress-bar">
                  <span />
                </div>
                <p>Running detection, tracking, overlays, TTS, and merge.</p>
              </div>
            )}

            {!loading && videoUrl && (
              <video className="video-preview" controls>
                <source src={videoUrl} type="video/mp4" />
              </video>
            )}

            {!loading && !videoUrl && (
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

export default App;
