import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState("");

  const handleSubmit = async () => {
    if (!file) {
      return;
    }

    setLoading(true);
    setVideoUrl("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/upload-video", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setVideoUrl(data.output_video_url);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "24px", fontFamily: "sans-serif" }}>
      <input
        type="file"
        accept="video/*"
        onChange={(event) => setFile(event.target.files?.[0] || null)}
      />
      <button
        onClick={handleSubmit}
        disabled={!file || loading}
        style={{ marginLeft: "12px" }}
      >
        Run
      </button>
      {loading && <p>Processing...</p>}
      {videoUrl && (
        <video controls width="640" style={{ display: "block", marginTop: "16px" }}>
          <source src={videoUrl} type="video/mp4" />
        </video>
      )}
    </div>
  );
}

export default App;
