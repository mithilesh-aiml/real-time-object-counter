from collections import Counter
from pathlib import Path
import threading

import av
import cv2
import streamlit as st
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from ultralytics import YOLO


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Real-Time Object Counter",
    page_icon="🎯",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "yolo11n.pt"

st.title("🎯 Real-Time Object Detection & Counting")
st.write("Detect and count objects from your webcam in real time.")


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "yolo11n.pt was not found. Keep the model file in the same folder as app.py."
        )
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the YOLO model: {exc}")
    st.stop()


# ---------------- VIDEO PROCESSOR ----------------
class ObjectDetector(VideoProcessorBase):
    def __init__(self, confidence: float = 0.40):
        self.confidence = confidence
        self.lock = threading.Lock()
        self.counts = Counter()
        self.total = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        results = model.predict(
            source=image,
            conf=self.confidence,
            device="cpu",
            verbose=False,
        )

        result = results[0]
        counts = Counter()

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            counts[class_name] += 1

        annotated = result.plot()
        cv2.putText(
            annotated,
            f"Objects detected: {sum(counts.values())}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        with self.lock:
            self.counts = counts
            self.total = sum(counts.values())

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.header("⚙️ Detection Settings")
confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
)
st.sidebar.info("Higher confidence means fewer, more certain detections.")
st.sidebar.caption("This hosted version runs inference on the CPU.")

# ---------------- CAMERA ----------------
st.markdown("### 📷 Camera")
st.write(
    "Allow camera permission in your browser, then click START. "
    "Click STOP when you are finished."
)

ctx = webrtc_streamer(
    key="object-counter",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: ObjectDetector(confidence),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ---------------- LIVE COUNTERS ----------------
st.markdown("### 📊 Live Statistics")


@st.fragment(run_every=1)
def show_statistics():
    processor = ctx.video_processor

    if processor is None:
        st.info("Start the camera to see live object counts.")
        return

    with processor.lock:
        current_counts = processor.counts.copy()
        total = processor.total

    col1, col2 = st.columns(2)
    col1.metric("Objects in current frame", total)
    col2.metric("Different object classes", len(current_counts))

    if current_counts:
        st.markdown("#### Object-wise count")
        st.dataframe(
            [
                {"Object": name, "Count": count}
                for name, count in current_counts.most_common()
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No objects detected in the current frame.")


show_statistics()

st.caption(
    "Counts show objects visible in the current frame. "
    "The same object can appear in counts again in later frames."
)
