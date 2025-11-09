import streamlit as st
import os
import threading
import time
from datetime import datetime
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import google.generativeai as genai
import cv2
import queue

# --- PAGE CONFIG ---
st.set_page_config(page_title="YOLO + Gemini Monitor", layout="wide")

# --- CONFIG ---
VIDEO_SOURCE = 0  # 0 = default webcam
SUMMARY_DIR = "summaries"
CLIP_DIR = "clips"
ALERTS_DIR = "alerts"
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)

# --- LOAD SECRETS (Streamlit Version) ---
# Create a .streamlit/secrets.toml file with your secrets
EMAIL_SENDER = "krishna1525606@gmail.com"
EMAIL_PASSWORD = "xmwx uxqm chjm bfcw"  # Use Gmail App Password
EMAIL_RECEIVER = "cr891557@gmail.com"
GEMINI_API_KEY = "AIzaSyCOO5dF5Bh6yQtdcF-ob30M0YQjku8CDBo"

# Check if secrets are loaded
if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER, GEMINI_API_KEY]):
    st.error("❌ Missing secrets. Please add EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER, and GEMINI_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()

# --- GEMINI CONFIG ---
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# --- MODIFIED: Thread-safe UI communication ---
# We will use a queue to send messages (toasts, errors) from threads to the main UI
def get_toast_queue():
    if "toast_queue" not in st.session_state:
        st.session_state.toast_queue = queue.Queue()
    return st.session_state.toast_queue

# --- EMAIL ALERT FUNCTION ---
def send_email_alert(subject, body, image_path=None):
    toast_queue = get_toast_queue()
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        toast_queue.put(("warning", "⚠️ Email config missing, skipping send."))
        return

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(body, "plain"))

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
            msg.attach(img)
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        toast_queue.put(("toast", f"📧 Email sent: {subject}"))
    except Exception as e:
        toast_queue.put(("error", f"❌ Email failed: {e}"))

# --- RECORD SHORT CLIP (MODIFIED) ---
def record_clip(duration, stop_event):
    toast_queue = get_toast_queue()
    toast_queue.put(("toast", f"🎥 Recording a {duration}-second clip..."))

    out = None         # VideoWriter object
    clip_path = None   # Path to the saved clip
    
    start_time = time.time()
    
    # Loop for the specified duration
    while (time.time() - start_time < duration) and not stop_event.is_set():
        frame_to_write = None
        
        # Safely read the latest frame from the YOLO thread
        with st.session_state.frame_lock:
            if st.session_state.latest_frame is not None:
                # Make a copy to avoid thread conflicts
                frame_to_write = st.session_state.latest_frame.copy() 
        
        if frame_to_write is not None:
            # --- Initialize the VideoWriter on the first valid frame ---
            if out is None:
                try:
                    height, width, _ = frame_to_write.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_path = os.path.join(CLIP_DIR, f"clip_{timestamp}.mp4")
                    # Use 20.0 FPS as specified in your original code
                    out = cv2.VideoWriter(clip_path, fourcc, 20.0, (width, height)) 
                    toast_queue.put(("toast", "Writing clip..."))
                except Exception as e:
                    toast_queue.put(("error", f"❌ Failed to init VideoWriter: {e}"))
                    return None # Critical failure
            # --- End of initialization ---
            
            out.write(frame_to_write)
        
        # Sleep to approximate the 20 FPS of your VideoWriter
        # This prevents writing the same frame 1000x
        time.sleep(0.05) 

    # Clean up
    if out:
        out.release()

    if stop_event.is_set():
        toast_queue.put(("toast", "Clip recording interrupted by stop signal."))
        return None

    if clip_path and os.path.exists(clip_path):
        toast_queue.put(("toast", f"✅ Saved clip: {clip_path}"))
        return clip_path
    else:
        toast_queue.put(("warning", "⚠️ Clip recording finished, but no frames were saved."))
        return None

# --- GEMINI THREAD ---
def run_gemini_inference(stop_event):
    toast_queue = get_toast_queue()
    count = 1
    while not stop_event.is_set():
        clip_path = record_clip(duration=30, stop_event=stop_event)
        
        if stop_event.is_set():
             break
        
        if clip_path:
            toast_queue.put(("toast", f"🧠 Gemini inference #{count} starting..."))
            uploaded_clip = None
            try:
                # 1. Start upload
                uploaded_clip = genai.upload_file(path=clip_path)
                
                # 2. Wait for file to be ACTIVE
                file = genai.get_file(name=uploaded_clip.name)
                file_state_name = file.state.name
                while file_state_name != "ACTIVE":
                    if stop_event.is_set():
                        toast_queue.put(("toast", "Stop signal during file processing."))
                        genai.delete_file(uploaded_clip.name)
                        return
                    
                    toast_queue.put(("toast", f"File state is '{file_state_name}'. Waiting 10s..."))
                    time.sleep(10)
                    file = genai.get_file(name=uploaded_clip.name)
                    file_state_name = file.state.name

                    if file_state_name == "FAILED":
                        raise Exception(f"File upload failed: {uploaded_clip.name}")
                
                toast_queue.put(("toast", "✅ File is ACTIVE. Running inference..."))
                
                # 3. Run inference
                response = gemini_model.generate_content([
                    "Analyze this short clip and describe what's happening in detail.",
                    uploaded_clip
                ])

                summary = response.text or "No response text."
                file_path = os.path.join(SUMMARY_DIR, f"summary_{count:03d}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                toast_queue.put(("toast", f"✅ Saved Gemini summary #{count}"))

                genai.delete_file(uploaded_clip.name)

            except Exception as e:
                toast_queue.put(("error", f"❌ Gemini inference failed: {e}"))
                if uploaded_clip:
                    try:
                        genai.delete_file(uploaded_clip.name)
                    except Exception:
                        pass
        
        count += 1
        
        # Wait 45 minutes or until stop_event is set
        toast_queue.put(("toast", f"Next Gemini analysis in 45 minutes."))
        wait_time_seconds = 45 * 60
        stop_event.wait(timeout=wait_time_seconds)

    toast_queue.put(("toast", "🛑 Gemini thread stopped."))

# --- YOLO DETECTION (STREAMLIT) ---
# <-- MODIFIED: This function no longer takes 'placeholder'
# It now communicates via st.session_state
def run_yolo_detection(stop_event):
    toast_queue = get_toast_queue()
    
    # <-- ADDED: Thread-safe lock initialization -->
    # This prevents the race condition by ensuring the lock exists
    # before the thread ever tries to use it.
    if 'frame_lock' not in st.session_state:
        st.session_state.frame_lock = threading.Lock()
    # <-- END OF ADDED BLOCK -->

    model = YOLO("yolov8s-world.pt")
    model.set_classes(["person"])
    seen_ids = set()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        # <-- MODIFIED: Use session state to report critical error
        st.session_state.camera_error = "Cannot open video source. Please check camera connection."
        return

    toast_queue.put(("toast", "🎥 Starting real-time detection..."))

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            st.session_state.camera_error = "Camera feed lost. Stopping detection."
            break
        
        try:
            # Run tracking
            # <-- MODIFIED: Removed classes=["person"] from this call.
            # The model was already configured with model.set_classes(["person"])
            results = model.track(source=frame, persist=True, show=False)
            annotated_frame = results[0].plot()
            
            # <-- MODIFIED: Share frame via session_state with a lock
            with st.session_state.frame_lock:
                st.session_state.latest_frame = annotated_frame

            # Check for new IDs (email logic)
            if results[0].boxes.id is not None:
                names = results[0].names
                classes = results[0].boxes.cls
                ids = results[0].boxes.id.int().tolist()

                for cls, track_id in zip(classes, ids):
                    label = names[int(cls)]
                    if label == "person" and track_id not in seen_ids:
                        seen_ids.add(track_id)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        image_name = f"alert_{timestamp}_id{track_id}.jpg"
                        image_path = os.path.join(ALERTS_DIR, image_name)
                        cv2.imwrite(image_path, frame) # Save original frame
                        
                        toast_queue.put(("toast", f"🚨 New person (ID {track_id}) detected! Sending email..."))
                        send_email_alert(
                            subject="🚨 New Person Detected Alert",
                            body=f"A new person (ID {track_id}) detected at {timestamp}.",
                            image_path=image_path
                        )
        except Exception as e:
            # Use console for non-critical errors in the loop
            print(f"Error in YOLO loop: {e}") 

    cap.release()
    toast_queue.put(("toast", "🛑 YOLO thread stopped."))

# --- MERGE FINAL SUMMARIES ---
def create_final_summary():
    toast_queue = get_toast_queue()
    toast_queue.put(("toast", "⏳ Generating final summary..."))
    summaries = []
    for file in sorted(os.listdir(SUMMARY_DIR)):
        if file.endswith(".txt"):
            with open(os.path.join(SUMMARY_DIR, file), "r", encoding="utf-8") as f:
                summaries.append(f"--- Summary from {file} ---\n{f.read()}")
    
    if not summaries:
        toast_queue.put(("warning", "⚠️ No summaries found to merge."))
        return "No summaries were generated."
        
    merged_text = "\n\n".join(summaries)
    
    try:
        final_summary = gemini_model.generate_content([
            "You are a security analyst. Combine these individual surveillance reports into a single, concise, chronological executive summary. Highlight any unusual patterns or frequently observed activities.",
            merged_text
        ]).text
        
        final_path = "final_summary_report.txt"
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        
        toast_queue.put(("toast", f"📄 Final summary saved as {final_path}"))
        return final_summary
        
    except Exception as e:
        toast_queue.put(("error", f"❌ Failed to generate final summary: {e}"))
        return f"Error generating summary: {e}"

# --- MAIN STREAMLIT APP ---
st.title("👁️ Real-Time YOLO + Gemini Monitoring")

# Initialize session state
# <-- MODIFIED: This block is changed to be more robust -->
# We now check for each variable individually to prevent race conditions.
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'gemini_thread' not in st.session_state:
    st.session_state.gemini_thread = None
if 'yolo_thread' not in st.session_state:
    st.session_state.yolo_thread = None
if 'final_summary' not in st.session_state:
    st.session_state.final_summary = ""
if 'frame_lock' not in st.session_state:
    st.session_state.frame_lock = threading.Lock() 
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'toast_queue' not in st.session_state:
    st.session_state.toast_queue = queue.Queue()
if 'camera_error' not in st.session_state:
    st.session_state.camera_error = None
# <-- END OF MODIFIED BLOCK -->

# UI Layout
col1, col2 = st.columns(2)
start_button = col1.button("🚀 Start Monitoring", type="primary", disabled=st.session_state.monitoring_active)
stop_button = col2.button("🛑 Stop Monitoring", disabled=not st.session_state.monitoring_active)

video_placeholder = st.empty()
summary_expander = st.expander("Final Summary Report", expanded=True)

# Handle Start Button
if start_button:
    st.session_state.monitoring_active = True
    st.session_state.stop_event = threading.Event()
    st.session_state.final_summary = ""   # Clear old summary
    st.session_state.latest_frame = None # Clear old frame
    st.session_state.camera_error = None # Clear old errors
    st.session_state.toast_queue = queue.Queue() # Clear old toasts
    
    # Start Gemini Thread
    st.session_state.gemini_thread = threading.Thread(
        target=run_gemini_inference,
        args=(st.session_state.stop_event,),
        daemon=True
    )
    st.session_state.gemini_thread.start()
    
    # Start YOLO Thread
    st.session_state.yolo_thread = threading.Thread(
        target=run_yolo_detection,
        # <-- MODIFIED: No placeholder passed
        args=(st.session_state.stop_event,), 
        daemon=True
    )
    st.session_state.yolo_thread.start()
    st.rerun()

# Handle Stop Button
if stop_button:
    st.toast("Stopping monitors... Please wait.")
    st.session_state.monitoring_active = False
    
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
        
    if st.session_state.yolo_thread:
        st.session_state.yolo_thread.join()
        
    if st.session_state.gemini_thread:
        st.session_state.gemini_thread.join()
    
    st.toast("All threads stopped. Generating final report.")
    
    # Generate and store summary
    st.session_state.final_summary = create_final_summary()
    
    # Clean up state
    st.session_state.stop_event = None
    st.session_state.yolo_thread = None
    st.session_state.gemini_thread = None
    
    st.rerun()

# --- MODIFIED: Main app loop for UI updates ---
# This loop runs on the main thread
# It displays toasts and the video frame
try:
    # Process all messages from the background threads
    while not st.session_state.toast_queue.empty():
        level, message = st.session_state.toast_queue.get_nowait()
        if level == "toast":
            st.toast(message)
        elif level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)

    if st.session_state.monitoring_active:
        # Check for critical camera errors from the YOLO thread
        if st.session_state.camera_error:
            st.error(st.session_state.camera_error)
            # Trigger a stop if the camera fails
            if stop_button: # Use the button's logic
                pass 
            else:
                st.session_state.monitoring_active = False
                st.rerun() # Rerun to show the "Stop" button logic

        # Draw the latest video frame
        with st.session_state.frame_lock:
            if st.session_state.latest_frame is not None:
                video_placeholder.image(st.session_state.latest_frame, channels="BGR", use_column_width=True)
            else:
                video_placeholder.info("Starting camera... Please wait.")
        
        # Rerun to create a loop
        time.sleep(0.03) # ~30 fps
        st.rerun()

    else:
        # Show this when monitoring is off
        video_placeholder.info("Monitoring is stopped. Click 'Start Monitoring' to begin.")
        
except Exception as e:
    # This might catch errors if the page is closed weirdly
    print(f"Error in main UI loop: {e}")


# Display final summary
with summary_expander:
    if st.session_state.final_summary:
        st.markdown(st.session_state.final_summary)
    else:
        st.write("No summary generated yet.")