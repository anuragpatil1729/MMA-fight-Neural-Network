import argparse
from pipeline.video_pipeline import VideoPipeline


def main():

    parser = argparse.ArgumentParser(description="Run MMA De-Cage pipeline")

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--output",
        default="runtime/outputs/fight_no_cage.mp4",
        help="Output video path"
    )

    args = parser.parse_args()

    print("Starting pipeline...")
    print("Input:", args.input)

    pipeline = VideoPipeline()

    pipeline.run(
        input_video=args.input,
        output_video=args.output
    )

    print("Pipeline finished!")
    print("Output saved to:", args.output)


if __name__ == "__main__":
    main()