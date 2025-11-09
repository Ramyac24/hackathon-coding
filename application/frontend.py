# frontend.py
import streamlit as st
import requests
import time
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLO + Gemini Monitor", layout="wide")
st.title("👁️ Real-Time YOLO + Gemini Monitoring")

# --- CONFIG ---
# URL of your Flask backend
FLASK_API_URL = os.environ.get("FLASK_API_URL", "http://127.0.0.1:5000")

# --- SESSION STATE ---
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'final_summary' not in st.session_state:
    st.session_state.final_summary = ""
if 'camera_error' not in st.session_state:
    st.session_state.camera_error = None

# --- API HELPER FUNCTIONS ---

def get_backend_status():
    """Polls the backend for its current status."""
    try:
        r = requests.get(f"{FLASK_API_URL}/status", timeout=1)
        r.raise_for_status()
        data = r.json()
        st.session_state.monitoring_active = data.get("monitoring_active", False)
        st.session_state.camera_error = data.get("camera_error")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Backend connection error: {e}")
        st.session_state.monitoring_active = False

def process_logs():
    """Gets and displays logs (toasts, errors) from the backend."""
    try:
        r = requests.get(f"{FLASK_API_URL}/get_logs", timeout=1)
        r.raise_for_status()
        logs = r.json().get("logs", [])
        for log in logs:
            if log["level"] == "toast":
                st.toast(log["message"])
            elif log["level"] == "error":
                st.error(log["message"])
            elif log["level"] == "warning":
                st.warning(log["message"])
    except requests.exceptions.RequestException:
        # Fail silently if log polling fails
        pass

# --- UI LAYOUT ---
col1, col2 = st.columns(2)

# Get status before drawing buttons to set their state
get_backend_status()

start_button = col1.button("🚀 Start Monitoring", type="primary", disabled=st.session_state.monitoring_active)
stop_button = col2.button("🛑 Stop Monitoring", disabled=not st.session_state.monitoring_active)

video_placeholder = st.empty()
summary_expander = st.expander("Final Summary Report", expanded=True)

# --- BUTTON LOGIC ---
if start_button:
    try:
        r = requests.post(f"{FLASK_API_URL}/start", timeout=5)
        r.raise_for_status()
        st.session_state.monitoring_active = True
        st.session_state.final_summary = "" # Clear old summary
        st.session_state.camera_error = None
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to start monitoring: {e}")

if stop_button:
    try:
        with st.spinner("Stopping monitors and generating final report... This may take a moment."):
            r = requests.post(f"{FLASK_API_URL}/stop", timeout=60) # Long timeout for summary
            r.raise_for_status()
            data = r.json()
            st.session_state.monitoring_active = False
            st.session_state.final_summary = data.get("summary", "Error retrieving summary.")
        st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to stop monitoring: {e}")

# --- MAIN UI LOOP ---

# Process any pending logs from the backend
process_logs()

if st.session_state.monitoring_active:
    if st.session_state.camera_error:
        video_placeholder.error(f"❌ Camera Error: {st.session_state.camera_error}")
    else:
        # Display the MJPEG stream from the Flask backend
        video_placeholder.image(f"{FLASK_API_URL}/video_feed")
    
    # Rerun to create a loop, polling for logs and status
    time.sleep(0.03) # ~30 fps update
    st.rerun()

else:
    video_placeholder.info("Monitoring is stopped. Click 'Start Monitoring' to begin.")

# Display final summary
with summary_expander:
    if st.session_state.final_summary:
        st.markdown(st.session_state.final_summary)
    else:
        st.write("No summary generated yet.")