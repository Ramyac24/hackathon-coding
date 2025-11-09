from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText

# --- CONFIG ---
VIDEO_PATH = "WhatsApp Video 2025-11-08 at 14.55.09.mp4"
ALERT_SENT = False  # Only send one alert

# --- EMAIL CONFIG ---
EMAIL_SENDER = "krishna1525606@gmail.com"
EMAIL_PASSWORD = "xmwx uxqm chjm bfcw"  # Use Gmail App Password
EMAIL_RECEIVER = "pandeyraunak1007@gmail.com"

# --- EMAIL ALERT FUNCTION ---
def send_email_alert(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

    print("📧 Email alert sent successfully!")

# --- YOLO MODEL ---
model = YOLO("yolov8s-world.pt")
model.set_classes(["person"])

# --- TRACK LOOP ---
for result in model.track(source=VIDEO_PATH, show=True, stream=True):
    if ALERT_SENT:
        continue  # Skip checking after first alert

    names = result.names
    boxes = result.boxes
    detected_classes = [names[int(cls)] for cls in boxes.cls]

    if "person" in detected_classes:
        print("🚨 Person detected for the first time! Sending email alert...")
        send_email_alert(
            subject="🚨 Person Detected Alert",
            body="Your YOLO detection system has detected a person. Go check it lodu."
        )
        ALERT_SENT = True
        break  # stop processing further frames once alert is sent
