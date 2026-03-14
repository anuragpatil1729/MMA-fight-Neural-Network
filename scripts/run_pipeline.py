import argparse
import importlib.util
import sys
from pathlib import Path

# Ensure repo root is importable when the script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run MMA De-Cage pipeline")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument(
        "--output",
        default="runtime/outputs/fight_no_cage.mp4",
        help="Output video path",
    )
    args = parser.parse_args()

    if importlib.util.find_spec("cv2") is None:
        raise SystemExit(
            "Missing dependency: opencv-python. Install requirements before running the pipeline."
        )

    from pipeline.video_pipeline import VideoPipeline

    print("Starting pipeline...")
    print("Input:", args.input)

    pipeline = VideoPipeline()
    pipeline.run(input_video=args.input, output_video=args.output)

    print("Pipeline finished!")
    print("Output saved to:", args.output)


if __name__ == "__main__":
    main()
