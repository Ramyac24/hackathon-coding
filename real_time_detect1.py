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
#from dotenv import load_dotenv  # <-- IMPROVEMENT: Load secrets from .env file

# --- CONFIG ---
VIDEO_SOURCE = 0
SUMMARY_DIR = "summaries"
CLIP_DIR = "clips"
ALERTS_DIR = "alerts"  # <-- IMPROVEMENT: Directory for alert snapshots
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(ALERTS_DIR, exist_ok=True)  # <-- IMPROVEMENT

# <-- IMPROVEMENT: Use threading.Event for clean stop signal
stop_event = threading.Event()

# --- LOAD SECRETS ---
#load_dotenv()  # <-- IMPROVEMENT: Load from .env file
EMAIL_SENDER = "krishna1525606@gmail.com"
EMAIL_PASSWORD = "xmwx uxqm chjm bfcw"  # Use Gmail App Password
EMAIL_RECEIVER = "cr891557@gmail.com"
GEMINI_API_KEY = "AIzaSyCOO5dF5Bh6yQtdcF-ob30M0YQjku8CDBo"

# Check if secrets are loaded
if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER, GEMINI_API_KEY]):
    print("❌ Critical error: Missing environment variables.")
    print("Please create a .env file with EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER, and GEMINI_API_KEY")
    exit()

# --- GEMINI CONFIG ---
genai.configure(api_key=GEMINI_API_KEY)
# <-- IMPROVEMENT: Use a valid, current model name
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# --- EMAIL ALERT FUNCTION ---
def send_email_alert(subject, body, image_path=None):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        print("⚠️ Email config missing, skipping send.")
        return

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.attach(MIMEText(body, "plain"))

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(image_path)}"
            )
            msg.attach(img)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 Email sent with snapshot {image_path}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# --- RECORD SHORT CLIP ---
def record_clip(duration=30):
    print(f"🎥 Recording a {duration}-second clip...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌ Cannot access camera for clip recording.")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_path = os.path.join(CLIP_DIR, f"clip_{timestamp}.mp4")
    out = cv2.VideoWriter(clip_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    start = time.time()
    # <-- IMPROVEMENT: Check stop_event while recording
    while (time.time() - start < duration) and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()

    if stop_event.is_set():
        print("Clip recording interrupted by stop signal.")
        return None

    print(f"✅ Saved clip: {clip_path}")
    return clip_path

# --- GEMINI THREAD ---
def run_gemini_inference():
    count = 1
    while not stop_event.is_set():  # Check event
        clip_path = record_clip(duration=30)
        
        if stop_event.is_set(): # Check again after recording
             break

        if clip_path:
            print(f"🧠 Gemini inference #{count}...")
            uploaded_clip = None # <-- ADDED: Define variable in case of early exit
            try:
                # 1. Start the upload
                print(f"Uploading clip: {clip_path}")
                uploaded_clip = genai.upload_file(path=clip_path)
                print(f"📤 Uploaded {uploaded_clip.name}. Waiting for 'ACTIVE' state...")

                # 2. <-- MODIFIED: Wait for file to be ACTIVE -->
                file = genai.get_file(name=uploaded_clip.name)
                file_state_name = file.state.name # <-- Get the string NAME of the enum

                while file_state_name != "ACTIVE":
                    if stop_event.is_set(): # Check for stop signal while waiting
                        print("Stop signal received during file processing. Aborting.")
                        if uploaded_clip: # <-- Clean up
                           genai.delete_file(uploaded_clip.name) 
                        return # Exit the function
                        
                    # <-- MODIFIED: Print the name for clarity -->
                    print(f"File state is '{file_state_name}' (Value: {file.state.value}). Waiting 10 seconds...")
                    time.sleep(10)
                    
                    file = genai.get_file(name=uploaded_clip.name)
                    file_state_name = file.state.name # <-- Update the name

                    if file_state_name == "FAILED":
                        print(f"❌ File upload failed: {uploaded_clip.name}")
                        raise Exception(f"File upload failed: {uploaded_clip.name}")
                
                print("✅ File is ACTIVE. Running inference...")
                # <-- END OF MODIFIED SECTION -->

                # 3. Now that it's active, run inference
                response = gemini_model.generate_content([
                    "Analyze this short clip and describe what's happening in detail.",
                    uploaded_clip
                ])

                summary = response.text or "No response text."
                file_path = os.path.join(SUMMARY_DIR, f"summary_{count:03d}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"✅ Saved Gemini summary #{count} -> {file_path}")

                genai.delete_file(uploaded_clip.name)
                print(f"🗑️ Cleaned up uploaded file: {uploaded_clip.name}")

            except Exception as e:
                print(f"❌ Gemini inference failed: {e}")
                # Clean up the file if it exists and we failed
                try:
                    # <-- MODIFIED: Check if uploaded_clip was ever assigned -->
                    if uploaded_clip:
                        genai.get_file(name=uploaded_clip.name) # Check if it exists
                        genai.delete_file(uploaded_clip.name)
                        print(f"🗑️ Cleaned up failed file: {uploaded_clip.name}")
                except Exception as del_e:
                    pass # File might not exist or already be deleted
        
        count += 1
        
        # Use event.wait() for efficient waiting
        print(f"Next Gemini analysis in 30 seconds. Waiting...")
        wait_time_seconds = 30
        stop_event.wait(timeout=wait_time_seconds)

    print("🛑 Gemini thread stopped.")


# --- YOLO DETECTION (REAL TIME) ---
def run_yolo_detection():
    model = YOLO("yolov8s-world.pt")
    model.set_classes(["person"])
    seen_ids = set()

    print("🎥 Starting real-time detection... Press Ctrl+C to quit.")

    try:
        for result in model.track(source=VIDEO_SOURCE, show=True, stream=True):
            if stop_event.is_set():  # <-- IMPROVEMENT: Check event
                print("Stop signal received, ending detection.")
                break

            frame = result.orig_img
            boxes = result.boxes

            if boxes.id is None:
                continue

            names = result.names
            classes = boxes.cls
            ids = boxes.id.int().tolist()

            for cls, track_id in zip(classes, ids):
                label = names[int(cls)]
                if label == "person" and track_id not in seen_ids:
                    seen_ids.add(track_id)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # <-- IMPROVEMENT: Save to alerts directory
                    image_name = f"alert_{timestamp}_id{track_id}.jpg"
                    image_path = os.path.join(ALERTS_DIR, image_name)
                    
                    cv2.imwrite(image_path, frame)
                    print(f"📸 Saved snapshot {image_path}")

                    print(f"🚨 New person (ID {track_id}) detected! Sending email...")
                    send_email_alert(
                        subject="🚨 New Person Detected Alert",
                        body=f"A new person (ID {track_id}) detected at {timestamp}.",
                        image_path=image_path
                    )
                    
    except Exception as e:  # <-- IMPROVEMENT: Catch camera/model errors
        print(f"❌ An error occurred during YOLO detection: {e}")
    finally:
        print("🛑 Camera feed ended.")
        stop_event.set()  # <-- IMPROVEMENT: Signal all other threads to stop

# --- MERGE FINAL SUMMARIES ---
def create_final_summary():
    print("⏳ Generating final summary...")
    summaries = []
    # <-- IMPROVEMENT: Use sorted(os.listdir) to read in order
    for file in sorted(os.listdir(SUMMARY_DIR)):
        if file.endswith(".txt"):
            with open(os.path.join(SUMMARY_DIR, file), "r", encoding="utf-8") as f:
                summaries.append(f"--- Summary from {file} ---\n{f.read()}")
    
    if not summaries:
        print("⚠️ No summaries found to merge.")
        return
        
    merged_text = "\n\n".join(summaries)
    
    try:
        final_summary = gemini_model.generate_content([
            "You are a security analyst. Combine these individual surveillance reports into a single, concise, chronological executive summary. Highlight any unusual patterns or frequently observed activities.",
            merged_text
        ]).text
        
        final_path = "final_summary_report.txt"
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        print(f"📄 Final summary saved as {final_path}")
        
        # Optional: Send final summary by email
        send_email_alert(
            subject="📈 Daily Surveillance Report",
            body=f"Here is the final merged report for the session:\n\n{final_summary}"
        )
        
    except Exception as e:
        print(f"❌ Failed to generate final summary: {e}")

# --- MAIN ---
if __name__ == "__main__":
    print("🚀 Starting real-time YOLO + Gemini system...")

    gemini_thread = threading.Thread(target=run_gemini_inference, daemon=True)
    gemini_thread.start()

    try:
        run_yolo_detection()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Shutting down...")
        stop_event.set() # <-- IMPROVEMENT: Signal stop on Ctrl+C

    # Wait for the Gemini thread to finish its current task and stop
    gemini_thread.join()
    
    # Only create summary if the script ran for a bit
    if not os.listdir(SUMMARY_DIR):
        print("No summaries were generated. Exiting.")
    else:
        create_final_summary()
        
    print("✅ All tasks completed successfully!")