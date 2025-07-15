import cv2
import os

def frames_to_video(frames_dir, output_video_path, frame_prefix="stylized_frame_", ext=".jpg", fps=24):
    """
    return the stylized video from the generated stylized frames
    """
    
    #getting the video's frames in a sorted list
    frame_files = sorted([
        os.path.join(frames_dir, f) 
        for f in os.listdir(frames_dir) 
        if f.startswith(frame_prefix) and f.endswith(ext) ])

    if not frame_files:
        print("invalid")
        return

    #read the first frame to get width and height
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  ##video writer for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    #writing every frame
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_video_path}")


frames_to_video(frames_dir="stylized_frames",
    output_video_path="stylized_output.mp4",
    frame_prefix="stylized_frame_", 
    ext=".jpg",
    fps=2.4)

