import cv2
from pipeline.frame_extractor import FrameExtractor
from pipeline.frame_processor import FrameProcessor
from pipeline.renderer import VideoRenderer


class VideoPipeline:

    def __init__(self):
        self.extractor = FrameExtractor()
        self.processor = FrameProcessor()
        self.renderer = VideoRenderer()

    def run(self, input_video, output_video):

        print("Extracting frames...")

        frames = self.extractor.extract(input_video)

        processed_frames = []

        print("Processing frames...")

        for frame in frames:
            processed = self.processor.process(frame)
            processed_frames.append(processed)

        print("Rendering video...")

        self.renderer.render(processed_frames, output_video)

        print("Finished processing video")