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

# --- CONFIG ---
VIDEO_SOURCE = 0  # 0 = default webcam, change if you have multiple cameras
SUMMARY_DIR = "summaries"
CLIP_DIR = "clips"
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)

STOP_ANALYSIS = False

# --- EMAIL CONFIG ---
EMAIL_SENDER = "krishna1525606@gmail.com"
EMAIL_PASSWORD = "xmwx uxqm chjm bfcw"  # Use Gmail App Password
EMAIL_RECEIVER = "cr891557@gmail.com"

# --- GEMINI CONFIG ---
genai.configure(api_key="AIzaSyCOO5dF5Bh6yQtdcF-ob30M0YQjku8CDBo")  # put your Gemini API key here
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# --- EMAIL ALERT FUNCTION (with image attachment) ---
def send_email_alert(subject, body, image_path=None):
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

# --- RECORD SHORT CLIP FOR GEMINI ---
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
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    print(f"✅ Saved clip: {clip_path}")
    return clip_path

# --- GEMINI THREAD ---
def run_gemini_inference():
    count = 1
    while not STOP_ANALYSIS:
        clip_path = record_clip(duration=30)
        if clip_path:
            print(f"🧠 Gemini inference #{count}...")
            try:
                uploaded_clip = genai.upload_file(path=clip_path)
                print(f"📤 Uploaded clip to Gemini: {uploaded_clip.uri}")

                response = gemini_model.generate_content([
                    "Analyze this short clip and describe what's happening in detail.",
                    uploaded_clip
                ])

                summary = response.text or "No response text."
                file_path = os.path.join(SUMMARY_DIR, f"summary_{count}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"✅ Saved Gemini summary #{count} -> {file_path}")

                # Optional cleanup
                genai.delete_file(uploaded_clip.name)

            except Exception as e:
                print(f"❌ Gemini inference failed: {e}")

        # check frequently for stop signal
        for _ in range(30):  # 30 seconds
            if STOP_ANALYSIS:
                break
            time.sleep(1)
        count += 1
    print("🛑 Gemini thread stopped.")


# --- YOLO DETECTION (REAL TIME) ---
def run_yolo_detection():
    global STOP_ANALYSIS

    model = YOLO("yolov8s-world.pt")
    model.set_classes(["person"])
    seen_ids = set()

    print("🎥 Starting real-time detection... Press Ctrl+C to quit.")

    for result in model.track(source=VIDEO_SOURCE, show=True, stream=True):
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

                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"alert_{timestamp}_id{track_id}.jpg"
                cv2.imwrite(image_path, frame)
                print(f"📸 Saved snapshot {image_path}")

                # Send email with image
                print(f"🚨 New person (ID {track_id}) detected! Sending email...")
                send_email_alert(
                    subject="🚨 New Person Detected Alert",
                    body=f"A new person (ID {track_id}) detected at {timestamp}.",
                    image_path=image_path
                )

    print("🛑 Camera feed ended.")
    STOP_ANALYSIS = True

# --- MERGE FINAL SUMMARIES ---
def create_final_summary():
    summaries = []
    for file in sorted(os.listdir(SUMMARY_DIR)):
        if file.endswith(".txt"):
            with open(os.path.join(SUMMARY_DIR, file), "r", encoding="utf-8") as f:
                summaries.append(f.read())
    if not summaries:
        print("⚠️ No summaries found.")
        return
    merged_text = "\n\n".join(summaries)
    final_summary = gemini_model.generate_content([
        f"Combine these summaries into one concise report:\n{merged_text}"
    ]).text
    with open("final_summary.txt", "w", encoding="utf-8") as f:
        f.write(final_summary)
    print("📄 Final summary saved as final_summary.txt")

# --- MAIN ---
if __name__ == "__main__":
    print("🚀 Starting real-time YOLO + Gemini system...")

    gemini_thread = threading.Thread(target=run_gemini_inference, daemon=True)
    gemini_thread.start()

    try:
        run_yolo_detection()
    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
        STOP_ANALYSIS = True

    gemini_thread.join()
    create_final_summary()
    print("✅ All tasks completed successfully!")
