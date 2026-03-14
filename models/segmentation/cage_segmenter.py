import cv2
import numpy as np


class CageSegmenter:
    """Generate a cage mask from an MMA frame.

    The approach is intentionally hybrid (classical CV + temporal smoothing)
    so it can run without a trained checkpoint in constrained environments.
    """

    def __init__(
        self,
        canny_low: int = 70,
        canny_high: int = 180,
        line_threshold: int = 80,
        min_line_length: int = 40,
        max_line_gap: int = 8,
        temporal_alpha: float = 0.65,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.line_threshold = line_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.temporal_alpha = temporal_alpha
        self._prev_mask_float = None

    @staticmethod
    def _line_angle_degrees(line: np.ndarray) -> float:
        x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return abs(angle)

    def _detect_mesh_lines(self, gray: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        line_mask = np.zeros_like(gray, dtype=np.uint8)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.line_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return line_mask

        for line in lines[:, 0, :]:
            angle = self._line_angle_degrees(line)
            # MMA cage mesh is mostly diagonal in perspective; suppress
            # near-horizontal/vertical arena features.
            if 20 <= angle <= 70 or 110 <= angle <= 160:
                x1, y1, x2, y2 = line
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

        return line_mask

    def _detect_low_saturation_wires(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Metallic cage wires are usually low-saturation + mid brightness.
        lower = np.array([0, 0, 35], dtype=np.uint8)
        upper = np.array([180, 65, 220], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def segment(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        line_mask = self._detect_mesh_lines(gray)
        color_mask = self._detect_low_saturation_wires(frame)

        mask = cv2.bitwise_and(line_mask, color_mask)

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

        mask_float = mask.astype(np.float32) / 255.0
        if self._prev_mask_float is None:
            stabilized = mask_float
        else:
            stabilized = (
                self.temporal_alpha * mask_float
                + (1.0 - self.temporal_alpha) * self._prev_mask_float
            )

        self._prev_mask_float = stabilized
        stabilized_mask = np.where(stabilized > 0.35, 255, 0).astype(np.uint8)
        return stabilized_mask
