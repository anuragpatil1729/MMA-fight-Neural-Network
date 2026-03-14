import cv2
import numpy as np


class VideoInpainter:
    """Inpaint cage regions while preserving motion detail around fighters."""

    def __init__(self, inpaint_radius: int = 3, blend_strength: float = 0.85):
        self.inpaint_radius = inpaint_radius
        self.blend_strength = blend_strength

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.max() == 0:
            return frame

        inpainted = cv2.inpaint(frame, mask, self.inpaint_radius, cv2.INPAINT_TELEA)

        # Feather mask edges to reduce popping/flicker around thin wires.
        feather = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
        feather = np.expand_dims(np.clip(feather * self.blend_strength, 0.0, 1.0), axis=-1)

        composed = (frame.astype(np.float32) * (1.0 - feather)) + (
            inpainted.astype(np.float32) * feather
        )
        return np.clip(composed, 0, 255).astype(np.uint8)
