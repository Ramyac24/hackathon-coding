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
import cv2  # <— needed to save frames

# --- CONFIG ---
VIDEO_PATH = "ghost.mp4"
SUMMARY_DIR = "summaries"
os.makedirs(SUMMARY_DIR, exist_ok=True)

STOP_ANALYSIS = False  # To stop Gemini after video ends

# --- EMAIL CONFIG ---
EMAIL_SENDER = "krishna1525606@gmail.com"
EMAIL_PASSWORD = "xmwx uxqm chjm bfcw"  # Use Gmail App Password
EMAIL_RECEIVER = "cr891557@gmail.com"

# --- GEMINI CONFIG ---
genai.configure(api_key="AIzaSyCOO5dF5Bh6yQtdcF-ob30M0YQjku8CDBo")
gemini_model = genai.GenerativeModel("gemini-2.5-pro")


# --- EMAIL ALERT FUNCTION (with image attachment) ---
def send_email_alert(subject, body, image_path=None):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    # Add text body
    msg.attach(MIMEText(body, "plain"))

    # Add image attachment (if provided)
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
            msg.attach(img)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"📧 Email alert sent successfully with image: {image_path}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# --- GEMINI PERIODIC ANALYSIS ---
def run_gemini_inference():
    count = 1
    while not STOP_ANALYSIS:
        print(f"🧠 Running Gemini inference #{count}...")
        try:
            response = gemini_model.generate_content([
                "Analyze this video and describe what's happening.",
                {"mime_type": "video/mp4", "data": open(VIDEO_PATH, "rb").read()}
            ])
            summary_text = response.text or "No response text."
            file_path = os.path.join(SUMMARY_DIR, f"summary_{count}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            print(f"✅ Saved Gemini summary #{count} -> {file_path}")
        except Exception as e:
            print(f"❌ Gemini inference failed: {e}")

        # Sleep (short interval for testing)
        total_sleep = 20  # 20 seconds (change to 45*60 for production)
        for _ in range(total_sleep):
            if STOP_ANALYSIS:
                break
            time.sleep(1)
        count += 1

    print("🛑 Gemini analysis thread stopped.")


# --- YOLO DETECTION FUNCTION ---
def run_yolo_detection():
    global STOP_ANALYSIS

    model = YOLO("yolov8s-world.pt")
    model.set_classes(["person"])
    seen_ids = set()  # to track unique person IDs

    print("🎥 Starting YOLO detection...")

    for result in model.track(source=VIDEO_PATH, show=True, stream=True):
        boxes = result.boxes
        frame = result.orig_img  # current frame (numpy array)

        if boxes.id is None:
            continue  # tracking ID not assigned yet

        names = result.names
        classes = boxes.cls
        ids = boxes.id.int().tolist()  # convert tensor -> list of ints

        for cls, track_id in zip(classes, ids):
            label = names[int(cls)]
            if label == "person" and track_id not in seen_ids:
                seen_ids.add(track_id)

                # Save current frame as image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"alert_{timestamp}_id{track_id}.jpg"
                cv2.imwrite(image_path, frame)
                print(f"📸 Saved detection snapshot: {image_path}")

                # Send email with attached image
                print(f"🚨 New person detected! (ID: {track_id}) Sending email alert...")
                send_email_alert(
                    subject="🚨 New Person Detected Alert",
                    body=f"A new person (ID {track_id}) has been detected by your YOLO system.",
                    image_path=image_path
                )

    print("🎥 Video stream ended.")
    STOP_ANALYSIS = True  # stop Gemini thread after video finishes


# --- FINAL SUMMARY MERGE ---
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
    final_summary = gemini_model.generate_content(
        [f"Combine these summaries into one concise final report:\n{merged_text}"]
    ).text

    with open("final_summary.txt", "w", encoding="utf-8") as f:
        f.write(final_summary)

    print("📄 Final consolidated summary saved as final_summary.txt")


# --- MAIN ---
if __name__ == "__main__":
    print("🚀 Starting YOLO detection and Gemini analysis...")

    # Start Gemini thread
    gemini_thread = threading.Thread(target=run_gemini_inference, daemon=True)
    gemini_thread.start()

    # Run YOLO detection
    run_yolo_detection()

    # Wait for Gemini to stop
    gemini_thread.join()

    # Merge summaries
    create_final_summary()
    print("✅ All tasks completed successfully!")
