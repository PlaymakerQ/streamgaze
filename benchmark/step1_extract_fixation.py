"""
Gaze Metadata Processing - Multi-Dataset Version
Supports EGTEA, Ego4D, and EgoExoLearn datasets
Uses functions from the preprocess module for better code organization
"""

import argparse
from pathlib import Path

from utils.config import get_dataset_choices, load_dataset_configs, str2bool

#   python step1_extract_fixation.py --dataset egtea
#   python step1_extract_fixation.py --dataset ego4d --fps 30
#   python step1_extract_fixation.py --dataset egoexo --fps 30
#   python step1_extract_fixation.py --dataset egoexo --fps 30 --no-viz  # Skip visualization (faster)
#   python step1_extract_fixation.py --dataset holoassist --fps 24.46 --no-viz
#   python step1_extract_fixation.py --dataset holoassist --fps 24.46 --viz-only  # Only regenerate visualization

DATASET_CHOICES = get_dataset_choices()


def print_visualization_mode(viz_only=False, skip_viz=False):
    """Print the current visualization mode."""
    if viz_only:
        print("[VISUALIZATION] Only regenerating visualization")
    elif skip_viz:
        print("[VISUALIZATION] Disabled for faster processing")
    else:
        print("[VISUALIZATION] Enabled")


def print_start(title, viz_only=False, skip_viz=False):
    print(f"Starting {title} Gaze Metadata Processing")
    print("=" * 60)
    print_visualization_mode(viz_only, skip_viz)


def print_finish(output_dir, completion_message, processed=None, skipped=None):
    print(f"\n{'='*60}")
    print(completion_message)
    print(f"{'='*60}" if processed is not None else f"Results saved in: {output_dir}")
    if processed is not None:
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


def build_output_checks(output_dir, video_name):
    output_check = Path(output_dir) / video_name / f"{video_name}_fixation_dataset.csv"
    viz_check = Path(output_dir) / video_name / f"{video_name}_gaze_visualization.mp4"
    return output_check, viz_check


def should_skip_video(video_name, output_check, viz_check, viz_only=False):
    """Return True when existing outputs mean the video should be skipped."""
    if viz_only:
        if Path(viz_check).exists():
            print(f"[SKIP] {video_name}: visualization already exists")
            return True
        if not Path(output_check).exists():
            print(f"[SKIP] {video_name}: fixation data not found")
            return True
        return False

    if Path(output_check).exists() and Path(viz_check).exists():
        print(f"[SKIP] {video_name}: already processed")
        return True
    return False


def run_dataset(adapter, fps=30, skip_viz=False, viz_only=False):
    """Run the shared processing flow while delegating dataset-specific work."""
    print_start(adapter.title, viz_only, skip_viz)

    if adapter.create_output_dir:
        Path(adapter.output_dir).mkdir(parents=True, exist_ok=True)

    adapter.prepare()
    tasks = list(adapter.iter_tasks())
    processed = 0
    skipped = 0

    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Processing {task.name}...")

        output_check, viz_check = build_output_checks(adapter.output_dir, task.name)
        if should_skip_video(task.name, output_check, viz_check, viz_only):
            if adapter.count_stats:
                skipped += 1
            continue

        if adapter.check_video_exists and not Path(task.video_path).exists():
            print(f"[SKIP] {task.name}: video file not found: {task.video_path}")
            if adapter.count_stats:
                skipped += 1
            continue

        if adapter.check_gaze_exists and not Path(task.gaze_path).exists():
            if adapter.missing_gaze_includes_path:
                print(f"[SKIP] {task.name}: gaze file not found: {task.gaze_path}")
            else:
                print(f"[SKIP] {task.name}: gaze file not found")
            if adapter.count_stats:
                skipped += 1
            continue

        action_data = adapter.load_action_data(task)
        adapter.run_task(task, action_data, fps=fps, skip_viz=skip_viz, viz_only=viz_only)
        if adapter.count_stats:
            processed += 1

    if adapter.count_stats:
        print_finish(adapter.output_dir, adapter.completion_message, processed, skipped)
    else:
        print_finish(adapter.output_dir, adapter.completion_message)


def process_dataset(dataset, fps=None, skip_viz=False, viz_only=False):
    """Process one configured dataset."""
    from utils.dataset_adapter import get_adapter_class

    cfg = load_dataset_configs(dataset)
    adapter_cls = get_adapter_class(cfg.info.dataset)
    effective_fps = cfg.data.default_fps if fps is None else fps
    run_dataset(adapter_cls(), fps=effective_fps, skip_viz=skip_viz, viz_only=viz_only)


def main():
    """Main processing function with dataset selection"""
    parser = argparse.ArgumentParser(description='Process gaze metadata for different datasets')
    parser.add_argument('--dataset', type=str,
                        default="egtea",
                        choices=DATASET_CHOICES,
                        help='Dataset to process')
    parser.add_argument('--fps', type=float, default=None,
                        help='Override the FPS from configs/dataset.yaml')
    parser.add_argument('--no-viz', type=str2bool, default=False,
                        help='Skip visualization generation (true/false)')
    parser.add_argument('--viz-only', action='store_true',
                        help='Only regenerate visualization (requires existing fixation data)')

    args = parser.parse_args()

    # Check for conflicting flags
    if args.no_viz and args.viz_only:
        print("Error: --no-viz and --viz-only are mutually exclusive")
        return

    process_dataset(
        args.dataset,
        fps=args.fps,
        skip_viz=args.no_viz,
        viz_only=args.viz_only,
    )


if __name__ == "__main__":
    main()
