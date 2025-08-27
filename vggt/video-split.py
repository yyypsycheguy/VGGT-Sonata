import cv2
import os

def extract_frames_per_second(video_folder, output_folder="images"):
    # Create output directory if needed
    os.makedirs(output_folder, exist_ok=True)
    
    for video_file in os.listdir("video"):
        if not video_file.endswith(".mp4"):
            continue
            
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # 1 frame per second
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                output_path = os.path.join(output_folder, f"frame_{saved_count}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
            frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames from {video_file}")



extract_frames_per_second("video")  
