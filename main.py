import argparse
from pathlib import Path

from pipeline import OUTPUT_DIR, PROJECT_DIR, process_video


def parse_args():
    parser = argparse.ArgumentParser(description="Student engagement detection")
    parser.add_argument("--input", default="input.mp4", help="Path to input video")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "output_annotated.mp4"),
        help="Path to save the annotated output video",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the processed video in a window while running",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick testing",
    )
    parser.add_argument(
        "--report-dir",
        default=str(OUTPUT_DIR),
        help="Directory where logs, database, charts, and summary files will be saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = PROJECT_DIR / args.input
    output_path = Path(args.output)
    report_dir = Path(args.report_dir)
    results = process_video(
        input_path=input_path,
        output_path=output_path,
        report_dir=report_dir,
        display=args.display,
        max_frames=args.max_frames,
    )
    print(f"Processed {results['processed_frames']} frame(s)")
    print(f"Annotated video saved to: {results['output_video']}")
    print(f"Database saved to: {results['database']}")
    print(f"CSV log saved to: {results['csv_log']}")
    print(f"Summary saved to: {results['summary_json']}")
    print(f"Student summary saved to: {results['student_summary_csv']}")
    print(f"Trend chart saved to: {results['trend_chart']}")
    print(f"Distribution chart saved to: {results['distribution_chart']}")
    print(f"Dashboard saved to: {results['dashboard_html']}")


if __name__ == "__main__":
    main()
