"""
get_data.py

Simple video processing pipeline:
- extract_frames: extract frames from a video file
- SimpleSORT: a minimal tracker that assigns incremental IDs to detections
- extract_aligned_faces: detect facial landmarks and save aligned square crops

This script uses OpenCV for video I/O and Haar cascades for quick face
localization, and the `face_alignment` package for landmark-based alignment.

Usage:
    python get_data.py

Adjust the constants at the bottom (video_path, frames_out, ...) as needed.

Requirements:
    pip install opencv-python face-alignment filterpy numpy

Note: This script intentionally uses CPU by default for face_alignment. If you
want GPU acceleration, change the device argument when creating the
FaceAlignment object.
"""

import os
import json
from typing import List, Tuple

import cv2
import numpy as np
import face_alignment


def extract_frames(video_pth: str, out: str, step: int = 1) -> None:
    """
    Extract frames from a video file and save them as JPEG images.

    Args:
        video_pth: Path to the input video file.
        out: Directory where extracted frames will be saved. The directory will
             be created if it does not exist.
        step: Save every `step`-th frame (default is 1, meaning every frame).
    """
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(video_pth)
    frame_count = 0
    save_count = 0

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_pth}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            frame_path = os.path.join(out, f"frame_{save_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            save_count += 1
        frame_count += 1

    cap.release()
    print(f"Saved {save_count} frames to '{out}'")


class SimpleSORT:
    """
    A minimal, deterministic tracker that assigns incremental IDs to each
    detection. This is NOT a proper SORT implementation; it simply records
    detections per frame with a unique id. Use a real tracker (e.g.
    trackpy, norfair, or a full SORT/DeepSORT) for production tracking.
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self.track_id = 0
        self.tracks = []

    def update(self, detections: List[Tuple[int, int, int, int]], frame_idx: int) -> None:
        """
        Add detections for a single frame to the internal track list.

        Args:
            detections: A list of bounding boxes in (x, y, w, h) format.
            frame_idx: The index (or sequential number) of the frame.
        """
        for det in detections:
            x, y, w, h = det
            self.track_id += 1
            self.tracks.append({
                "frame": int(frame_idx),
                "id": int(self.track_id),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
            })

    def save(self, out_json: str) -> None:
        """
        Save the recorded tracks to a JSON file.

        Args:
            out_json: Path to the output JSON file.
        """
        with open(out_json, "w") as f:
            json.dump(self.tracks, f, indent=2)
        print(f"Saved tracks to '{out_json}'")


def extract_aligned_faces(frames_folder: str, out_faces: str, max_faces: int = 100) -> None:
    """
    Detect faces in saved frames, estimate 3D landmarks, and save square,
    aligned face crops.

    Args:
        frames_folder: Directory containing image frames (JPEGs) to process.
        out_faces: Directory where aligned face crops will be saved.
        max_faces: Maximum number of face crops to save.
    """
    os.makedirs(out_faces, exist_ok=True)

    # Create a face alignment model (3D landmarks). Default device is CPU.
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.THREE_D,
        flip_input=False,
        device="cpu",
    )

    saved = 0
    for fname in sorted(os.listdir(frames_folder)):
        if not fname.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(frames_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Detect landmarks for all faces in the image
        preds = fa.get_landmarks(img)
        if preds is None:
            continue

        for face_landmarks in preds:
            x_min = np.min(face_landmarks[:, 0])
            y_min = np.min(face_landmarks[:, 1])
            x_max = np.max(face_landmarks[:, 0])
            y_max = np.max(face_landmarks[:, 1])

            size = max(x_max - x_min, y_max - y_min)
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            x1, y1 = int(cx - size / 2), int(cy - size / 2)
            x2, y2 = int(cx + size / 2), int(cy + size / 2)

            face_crop = img[max(0, y1):y2, max(0, x1):x2]
            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (512, 512))
            save_path = os.path.join(out_faces, f"face_{saved:04d}.jpg")
            cv2.imwrite(save_path, face_resized)
            saved += 1

            if saved >= max_faces:
                print(f"Collected {saved} aligned faces in '{out_faces}'")
                return

    print(f"Collected {saved} aligned faces in '{out_faces}'")


def main() -> None:
    video_path = "vid.mp4"
    frames_out = "frames"
    faces_out = "face_refs"
    tracks_out = "tracks.json"

    # 1) Extract frames
    extract_frames(video_path, frames_out, step=1)

    # 2) Detect faces using Haar cascade and save simple tracks
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    tracker = SimpleSORT()

    for idx, fname in enumerate(sorted(os.listdir(frames_out))):
        if not fname.lower().endswith(".jpg"):
            continue
        frame = cv2.imread(os.path.join(frames_out, fname))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        tracker.update(faces, idx)

    tracker.save(tracks_out)

    # 3) Aligned face crops
    extract_aligned_faces(frames_out, faces_out, max_faces=100)

    print("Pipeline finished (frames + tracks.json + aligned faces)")


if __name__ == "__main__":
    main()
