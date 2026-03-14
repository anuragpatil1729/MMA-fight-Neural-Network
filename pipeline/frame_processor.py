from models.inpainting.video_inpainter import VideoInpainter
from models.segmentation.cage_segmenter import CageSegmenter


class FrameProcessor:

    def __init__(self):
        self.segmenter = CageSegmenter()
        self.inpainter = VideoInpainter()

    def process(self, frame):
        """
        Process a single frame by segmenting cage wires and inpainting them.
        """
        cage_mask = self.segmenter.segment(frame)
        processed_frame = self.inpainter.inpaint(frame, cage_mask)
        return processed_frame
