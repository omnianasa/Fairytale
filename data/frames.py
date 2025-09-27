import cv2
import os

def get_the_frames(video_pth, out, step = 10):
     """
     extract the frames from the video
     """
     os.makedirs(out, exist_ok=True)
     cap = cv2.VideoCapture(video_pth)
     frame_count = 0
     save_count = 0
     while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            #iterating on every frame turn it to jpg in the output folder of frames
            frame_path = os.path.join(out, f"frame_{save_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            save_count += 1

        frame_count += 1

     cap.release()
     print(f"Saved {save_count} frames to '{out}'")


video_path = 'vid.mp4'
output_folder_F = 'data/A'
get_the_frames(video_path, output_folder_F, step=1)