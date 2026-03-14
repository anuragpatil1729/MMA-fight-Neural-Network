import cv2


class VideoRenderer:

    def render(self, frames, output_path, fps=30):

        if len(frames) == 0:
            raise ValueError("No frames to render")

        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        for frame in frames:
            writer.write(frame)

        writer.release()

        print("Video saved to:", output_path)