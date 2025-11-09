# monitoring_logic.py
import os
import time
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from ultralytics import YOLO
import google.generativeai as genai
from config import (
    EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER, GEMINI_MODEL,
    SUMMARY_DIR, CLIP_DIR, ALERTS_DIR
)

# --- EMAIL ALERT FUNCTION ---
def send_email_alert(subject, body, log_queue, image_path=None):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        log_queue.put(("warning", "⚠️ Email config missing, skipping send."))
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
        log_queue.put(("toast", f"📧 Email sent: {subject}"))
    except Exception as e:
        log_queue.put(("error", f"❌ Email failed: {e}"))

# --- RECORD SHORT CLIP ---
def record_clip(duration, stop_event, log_queue, frame_lock, shared_state):
    log_queue.put(("toast", f"🎥 Recording a {duration}-second clip..."))
    out = None
    clip_path = None
    start_time = time.time()

    while (time.time() - start_time < duration) and not stop_event.is_set():
        frame_to_write = None
        
        with frame_lock:
            if shared_state["latest_annotated_frame"] is not None:
                frame_to_write = shared_state["latest_annotated_frame"].copy()
        
        if frame_to_write is not None:
            if out is None:
                try:
                    height, width, _ = frame_to_write.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_path = os.path.join(CLIP_DIR, f"clip_{timestamp}.mp4")
                    out = cv2.VideoWriter(clip_path, fourcc, 20.0, (width, height))
                    log_queue.put(("toast", "Writing clip..."))
                except Exception as e:
                    log_queue.put(("error", f"❌ Failed to init VideoWriter: {e}"))
                    return None
            
            out.write(frame_to_write)
        
        time.sleep(0.05) 

    if out:
        out.release()

    if stop_event.is_set():
        log_queue.put(("toast", "Clip recording interrupted."))
        return None

    if clip_path and os.path.exists(clip_path):
        log_queue.put(("toast", f"✅ Saved clip: {clip_path}"))
        return clip_path
    else:
        log_queue.put(("warning", "⚠️ Clip recording finished, but no frames were saved."))
        return None

# --- GEMINI THREAD ---
def run_gemini_inference(stop_event, log_queue, frame_lock, shared_state):
    if not GEMINI_MODEL:
        log_queue.put(("error", "❌ Gemini API key not configured. Stopping Gemini thread."))
        return

    count = 1
    while not stop_event.is_set():
        clip_path = record_clip(duration=30, stop_event=stop_event, log_queue=log_queue, frame_lock=frame_lock, shared_state=shared_state)
        
        if stop_event.is_set(): break
        
        if clip_path:
            log_queue.put(("toast", f"🧠 Gemini inference #{count} starting..."))
            uploaded_clip = None
            try:
                uploaded_clip = genai.upload_file(path=clip_path)
                file = genai.get_file(name=uploaded_clip.name)
                while file.state.name != "ACTIVE":
                    if stop_event.is_set():
                        log_queue.put(("toast", "Stop signal during file processing."))
                        genai.delete_file(uploaded_clip.name)
                        return
                    
                    log_queue.put(("toast", f"File state '{file.state.name}'. Waiting 10s..."))
                    time.sleep(10)
                    file = genai.get_file(name=uploaded_clip.name)
                    if file.state.name == "FAILED":
                        raise Exception(f"File upload failed: {uploaded_clip.name}")
                
                log_queue.put(("toast", "✅ File is ACTIVE. Running inference..."))
                
                response = GEMINI_MODEL.generate_content([
                    "Analyze this short clip and describe what's happening in detail.",
                    uploaded_clip
                ])

                summary = response.text or "No response text."
                file_path = os.path.join(SUMMARY_DIR, f"summary_{count:03d}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                log_queue.put(("toast", f"✅ Saved Gemini summary #{count}"))
                genai.delete_file(uploaded_clip.name)

            except Exception as e:
                log_queue.put(("error", f"❌ Gemini inference failed: {e}"))
                if uploaded_clip:
                    try: genai.delete_file(uploaded_clip.name)
                    except Exception: pass
        
        count += 1
        log_queue.put(("toast", "Next Gemini analysis in 45 minutes."))
        stop_event.wait(timeout=30)

    log_queue.put(("toast", "🛑 Gemini thread stopped."))

# --- YOLO DETECTION ---
def run_yolo_detection(stop_event, log_queue, frame_lock, shared_state):
    from config import VIDEO_SOURCE # Import here for freshness
    
    model = YOLO("yolov8s-world.pt")
    model.set_classes(["person"])
    seen_ids = set()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        log_queue.put(("error", "❌ Cannot open video source. Check camera."))
        with frame_lock:
            shared_state["camera_error"] = "Cannot open video source."
        return

    log_queue.put(("toast", "🎥 Starting real-time detection..."))

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            log_queue.put(("error", "❌ Camera feed lost. Stopping detection."))
            with frame_lock:
                shared_state["camera_error"] = "Camera feed lost."
            break
        
        try:
            results = model.track(source=frame, persist=True, show=False)
            annotated_frame = results[0].plot()
            
            with frame_lock:
                shared_state["latest_annotated_frame"] = annotated_frame

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
                        cv2.imwrite(image_path, frame)
                        
                        log_queue.put(("toast", f"🚨 New person (ID {track_id})! Sending email..."))
                        send_email_alert(
                            subject="🚨 New Person Detected Alert",
                            body=f"A new person (ID {track_id}) detected at {timestamp}.",
                            log_queue=log_queue,
                            image_path=image_path
                        )
        except Exception as e:
            print(f"Error in YOLO loop: {e}") # Print to console
            time.sleep(0.5)

    cap.release()
    with frame_lock:
        shared_state["latest_annotated_frame"] = None
    log_queue.put(("toast", "🛑 YOLO thread stopped."))

# --- MERGE FINAL SUMMARIES ---
def create_final_summary(log_queue):
    log_queue.put(("toast", "⏳ Generating final summary..."))
    summaries = []
    for file in sorted(os.listdir(SUMMARY_DIR)):
        if file.endswith(".txt"):
            with open(os.path.join(SUMMARY_DIR, file), "r", encoding="utf-8") as f:
                summaries.append(f"--- Summary from {file} ---\n{f.read()}")
    
    if not summaries:
        log_queue.put(("warning", "⚠️ No summaries found to merge."))
        return "No summaries were generated."
        
    merged_text = "\n\n".join(summaries)
    
    try:
        final_summary = GEMINI_MODEL.generate_content([
            "You are a security analyst. Combine these individual surveillance reports into a single, concise, chronological executive summary. Highlight any unusual patterns or frequently observed activities.",
            merged_text
        ]).text
        
        final_path = "final_summary_report.txt"
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        
        log_queue.put(("toast", f"📄 Final summary saved as {final_path}"))
        return final_summary
        
    except Exception as e:
        log_queue.put(("error", f"❌ Failed to generate final summary: {e}"))
        return f"Error generating summary: {e}"