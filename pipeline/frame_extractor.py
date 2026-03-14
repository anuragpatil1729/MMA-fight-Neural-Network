import cv2
import os


class FrameExtractor:

    def __init__(self, output_dir="runtime/temp_frames"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def extract(self, video_path):

        cap = cv2.VideoCapture(video_path)

        frames = []
        index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_path = os.path.join(
                self.output_dir,
                f"frame_{index:05d}.jpg"
            )

            cv2.imwrite(frame_path, frame)

            frames.append(frame)

            index += 1

        cap.release()

        return frames