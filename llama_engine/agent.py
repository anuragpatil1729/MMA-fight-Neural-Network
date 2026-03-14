import cv2
import numpy as np


class VideoAgent:

    def process_frame(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect cage edges
        edges = cv2.Canny(gray, 80, 180)

        # thin mask
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)

        # blur only cage pixels
        blurred = cv2.GaussianBlur(frame, (7,7), 0)

        result = frame.copy()
        result[mask > 0] = blurred[mask > 0]

        return result