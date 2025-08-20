# extract_frames.py

import cv2
import os

def extract_frames(video_path, output_dir, resize=(640, 480)):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frame_name = f"frame_{frame_count:05d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        frame_count += 1

    cap.release()
    print(f"[INFO] ✅ Extracted {frame_count} frames to {output_dir}")

if __name__ == "__main__":
    video_path = r"C:\Users\Lenovo\Github\FSD-Navigation\Test_videos\Skidpad_FSE.mp4"
    output_dir = r"C:\Users\Lenovo\Github\FSD-Navigation\outputs\video_frames"
    extract_frames(video_path, output_dir)
