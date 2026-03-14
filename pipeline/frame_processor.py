from llama_engine.agent import VideoAgent


class FrameProcessor:

    def __init__(self):
        self.agent = VideoAgent()

    def process(self, frame):
        """
        Process a single frame using the AI agent
        """
        processed_frame = self.agent.process_frame(frame)
        return processed_frame