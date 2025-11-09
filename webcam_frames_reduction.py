import cv2
import numpy as np


CAMERA_INDEX = 0
MSE_THRESHOLD = 500
MAX_SIMILAR_FRAMES = 5
OUTPUT_VIDEO_PATH = "webcam_output.mp4"

def calculate_frame_difference(frame1, frame2):
    frame1 = frame1.astype("float")
    frame2 = frame2.astype("float")
    mse = np.sum((frame1 - frame2) ** 2) / float(frame1.size)
    return mse

def process_realtime_video(camera_index, output_path, threshold, max_similar):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    previous_frame_gray = None
    consecutive_similar_count = 0
    frames_saved = 0
    frames_read = 0

    while True:
        ret, current_frame = cap.read()
        
        if not ret:
            break

        frames_read += 1

        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if previous_frame_gray is None:
            previous_frame_gray = current_frame_gray
            cv2.imshow('Optimized Real-Time Feed', current_frame)
            out.write(current_frame)
            frames_saved += 1
            continue

        mse = calculate_frame_difference(previous_frame_gray, current_frame_gray)
        should_display = False

        if mse > threshold:
            should_display = True
            consecutive_similar_count = 0
        elif consecutive_similar_count >= max_similar:
            should_display = True
            consecutive_similar_count = 0 
        else:
            consecutive_similar_count += 1

        if should_display:
            cv2.imshow('Optimized Real-Time Feed', current_frame)
            previous_frame_gray = current_frame_gray
            out.write(current_frame)
            frames_saved += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    return frames_saved, frames_read


if __name__ == "__main__":
    try:
        print(f"Starting real-time video stream from camera index {CAMERA_INDEX}.")
        print(f"The optimized video will be saved to: {OUTPUT_VIDEO_PATH}")
        print("Press 'q' in the video window to stop and save the file.")
        print("-" * 40)
        
        frames_saved, frames_read = process_realtime_video(
            CAMERA_INDEX,
            OUTPUT_VIDEO_PATH,
            MSE_THRESHOLD,
            MAX_SIMILAR_FRAMES
        )
        
        frames_removed = frames_read - frames_saved
        print(f"Real-time video processing stopped successfully.")
        print("-" * 40)
        print(f"Total frames read (present): {frames_read}")
        print(f"Total frames removed (skipped): {frames_removed}")
        print(f"Total frames saved to output: {frames_saved}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")