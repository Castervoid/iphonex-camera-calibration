import cv2
import os

video_path = 'path/to/your/video.mp4'
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join(output_dir, f'frame_{frame_idx:05d}.jpg')
    cv2.imwrite(frame_filename, frame)
    frame_idx += 1

cap.release()
print(f"Saved {frame_idx} frames to '{output_dir}'")
cv2.destroyAllWindows()