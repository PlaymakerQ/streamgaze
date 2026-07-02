"""Shared dataset adapters for benchmark processing steps."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd

from constants import Const
from funcs.path_funcs import build_output_dir
from preprocess import (
    process_single_video,
    process_single_video_ego4d,
    process_single_video_egoexo,
    process_single_video_holoassist,
)

PIPELINE_DIR = Const.processed_data_root


@dataclass
class VideoTask:
    """A single video/session item ready for fixation extraction."""
    name: str
    video_path: Any
    gaze_path: Any
    action_data: Any = None
    metadata: dict = field(default_factory=dict)


class DatasetAdapter:
    """Dataset-specific hooks used by the shared step-1 processing loop."""
    title = ''
    completion_message = 'All videos processed!'
    create_output_dir = True
    count_stats = False
    check_video_exists = False
    check_gaze_exists = True
    missing_gaze_includes_path = False

    def prepare(self):
        """Load shared metadata needed before iterating tasks."""

    def iter_tasks(self):
        """Yield VideoTask objects in the dataset's original processing order."""
        raise NotImplementedError

    def load_action_data(self, task):
        """Load per-task action data after skip/existence checks."""
        return task.action_data

    def run_task(self, task, action_data, fps, skip_viz=False, viz_only=False):
        """Run one task for this dataset."""
        raise NotImplementedError


def build_fine_grained_action_data(annotations):
    """Convert fine-grained annotation rows to the action_data format."""
    return [
        {
            'start': row['start_sec'],
            'end': row['end_sec'],
            'textAttribute_en': row['narration_en_no_hand_prompt'],
            'annotation_id': row['annotation_id'],
            'subset': row['subset'],
        }
        for _, row in annotations.iterrows()
    ]


class EgteaAdapter(DatasetAdapter):
    title = 'EGTEA'
    create_output_dir = False

    def __init__(self):
        self.base_dir = Path(PIPELINE_DIR) / "raw_gaze_dataset" / "egtea"
        self.video_dir = Path(self.base_dir) / "videos"
        self.gaze_dir = Path(self.base_dir) / "gaze_data"
        self.output_dir = None
        self.action_data = None

    def prepare(self):
        self.output_dir = build_output_dir(data_name="etgea")
        action_file_path = Path(self.base_dir) / "raw_annotations" / "action_labels.csv"
        if Path(action_file_path).exists():
            print("Loading action data...")
            self.action_data = pd.read_csv(action_file_path, sep=';')
            print(f"Action data loaded: {self.action_data.shape}")
        else:
            print("Action data not found, proceeding without action information")

    def iter_tasks(self):
        video_files = sorted(Path(self.video_dir).glob("*.mp4"))
        print(f"\nFound {len(video_files)} video files")
        for video_path in video_files:
            video_name = Path(video_path).stem
            yield VideoTask(
                name=video_name,
                video_path=video_path,
                gaze_path=Path(self.gaze_dir) / f"{video_name}.txt",
                action_data=self.action_data,
            )

    def run_task(self, task, action_data, fps, skip_viz=False, viz_only=False):
        process_single_video(
            task.video_path,
            task.gaze_path,
            self.output_dir,
            action_data,
            skip_viz=skip_viz,
            viz_only=viz_only,
        )


class Ego4DAdapter(DatasetAdapter):
    title = 'Ego4D'

    def __init__(self):
        self.base_dir = Path(PIPELINE_DIR) / "raw_gaze_dataset" / "ego4d" / "v2"
        self.video_dir = Path(self.base_dir) / "gaze_videos" / "v2" / "full_scale"
        self.gaze_dir = Path(self.base_dir) / "gaze"
        self.output_dir = Path(PIPELINE_DIR) / "final_data" / "ego4d" / "metadata"

    def prepare(self):
        print("Note: Ego4D processing without action data")

    def iter_tasks(self):
        video_files = sorted(Path(self.video_dir).glob("*.mp4"))
        print(f"\nFound {len(video_files)} video files")
        for video_path in video_files:
            video_name = Path(video_path).stem
            yield VideoTask(
                name=video_name,
                video_path=video_path,
                gaze_path=Path(self.gaze_dir) / f"{video_name}.csv",
                action_data=None,
            )

    def run_task(self, task, action_data, fps, skip_viz=False, viz_only=False):
        process_single_video_ego4d(
            task.video_path,
            task.gaze_path,
            self.output_dir,
            action_data,
            fps=fps,
            skip_viz=skip_viz,
            viz_only=viz_only,
        )


class EgoExoAdapter(DatasetAdapter):
    title = 'EgoExoLearn'

    def __init__(self):
        self.base_dir = Path(PIPELINE_DIR) / "raw_gaze_dataset" / "egoexolearn" / "full"
        self.video_dir = self.base_dir
        self.gaze_dir = Path(self.base_dir) / "gazes_30fps_npy"
        self.annotation_dir = Path(self.base_dir) / "annotation"
        self.output_dir = Path(PIPELINE_DIR) / "final_data" / "egoexo" / "metadata"

    def iter_tasks(self):
        video_files = sorted(Path(self.video_dir).glob("*.mp4"))
        print(f"\nFound {len(video_files)} video files")
        for video_path in video_files:
            video_name = Path(video_path).stem
            yield VideoTask(
                name=video_name,
                video_path=video_path,
                gaze_path=Path(self.gaze_dir) / f"{video_name}.npy",
                metadata={
                    'annotation_path': Path(self.annotation_dir) / f"{video_name}.json",
                },
            )

    def load_action_data(self, task):
        action_data = None
        annotation_path = task.metadata['annotation_path']
        if Path(annotation_path).exists():
            try:
                with open(annotation_path, 'r') as f:
                    action_data = json.load(f)
                print(f"   Loaded {len(action_data)} annotations")
            except Exception as e:
                print(f"   Warning: failed to load annotation: {e}")
                action_data = None
        else:
            print("   Warning: no annotation file found")
        return action_data

    def run_task(self, task, action_data, fps, skip_viz=False, viz_only=False):
        process_single_video_egoexo(
            task.video_path,
            task.gaze_path,
            self.output_dir,
            action_data,
            fps=fps,
            skip_viz=skip_viz,
            viz_only=viz_only,
        )


class EgoExoFineGrainedAdapter(DatasetAdapter):
    count_stats = True
    check_video_exists = True
    missing_gaze_includes_path = True

    def __init__(self):
        self.base_dir = Path(PIPELINE_DIR) / "raw_gaze_dataset" / "egoexolearn" / "full"
        self.video_dir = self.base_dir
        self.gaze_dir = Path(self.base_dir) / "gazes_30fps_npy"
        self.fine_annotation_file = (
            Path(PIPELINE_DIR)
            / "raw_gaze_dataset"
            / "egoexolearn"
            / "annotations"
            / "fine_annotation_trainvaltest_en.csv"
        )
        self.annotations = None
        self.video_ids = []

    def prepare(self):
        print("Loading fine-grained annotations...")
        self.fine_annotations_df = pd.read_csv(self.fine_annotation_file)
        print(f"Total annotations loaded: {len(self.fine_annotations_df)}")
        self.prepare_fine_grained_subset()

    def prepare_fine_grained_subset(self):
        raise NotImplementedError

    def iter_tasks(self):
        for video_name in self.video_ids:
            yield VideoTask(
                name=video_name,
                video_path=Path(self.video_dir) / f"{video_name}.mp4",
                gaze_path=Path(self.gaze_dir) / f"{video_name}.npy",
            )

    def load_action_data(self, task):
        video_annotations = self.annotations[self.annotations['video_uid'] == task.name].copy()
        action_data = build_fine_grained_action_data(video_annotations)
        print(f"   Loaded {len(action_data)} fine-grained annotations")
        return action_data

    def run_task(self, task, action_data, fps, skip_viz=False, viz_only=False):
        process_single_video_egoexo(
            task.video_path,
            task.gaze_path,
            self.output_dir,
            action_data,
            fps=fps,
            skip_viz=skip_viz,
            viz_only=viz_only,
        )


class EgoExoLabAdapter(EgoExoFineGrainedAdapter):
    title = 'EgoExoLearn Lab'
    completion_message = 'EgoExo Lab Processing Complete!'

    def __init__(self):
        super().__init__()
        self.output_dir = Path(PIPELINE_DIR) / "final_data" / "egoexo" / "metadata" / "lab"

    def prepare_fine_grained_subset(self):
        self.annotations = self.fine_annotations_df[self.fine_annotations_df['scene'] == 'lab'].copy()
        print(f"Lab annotations: {len(self.annotations)}")

        lab_video_ids = self.annotations['video_uid'].unique()
        print(f"Found {len(lab_video_ids)} unique lab videos")
        self.video_ids = sorted(lab_video_ids)


class EgoExoKitchenAdapter(EgoExoFineGrainedAdapter):
    title = 'EgoExoLearn Kitchen'
    completion_message = 'EgoExo Kitchen Processing Complete!'

    def __init__(self):
        super().__init__()
        self.output_dir = Path(PIPELINE_DIR) / "final_data" / "egoexo" / "metadata" / "kitchen_160"

    def prepare(self):
        print("Scanning kitchen_160 directory for existing videos...")
        self.all_dirs = [
            d for d in [p.name for p in Path(self.output_dir).iterdir()]
            if (Path(self.output_dir) / d).is_dir()
        ]
        print(f"Found {len(self.all_dirs)} directories in kitchen_160")
        super().prepare()

    def prepare_fine_grained_subset(self):
        self.annotations = self.fine_annotations_df[
            self.fine_annotations_df['video_uid'].isin(self.all_dirs)
        ].copy()
        print(f"Annotations for kitchen_160 videos: {len(self.annotations)}")

        self.video_ids = sorted(self.all_dirs)
        print(f"Processing {len(self.video_ids)} videos from kitchen_160 directory")


class HoloAssistAdapter(DatasetAdapter):
    title = 'HoloAssist'
    check_gaze_exists = False

    def __init__(self):
        self.base_dir = Path(PIPELINE_DIR) / "raw_gaze_dataset" / "holoassist" / "full"
        self.output_dir = Path(PIPELINE_DIR) / "final_data" / "holoassist" / "metadata"
        self.annotation_file = Path(self.base_dir) / "data-annnotation-trainval-v1_1.json"
        self.annotation_data = None
        self.valid_sessions = []

    def prepare(self):
        annotated_video_names = set()
        if Path(self.annotation_file).exists():
            print("Loading annotation data...")
            with open(self.annotation_file, 'r') as f:
                self.annotation_data = json.load(f)
            print(f"Annotation data loaded: {len(self.annotation_data)} videos")

            annotated_video_names = set([
                v.get('video_name') for v in self.annotation_data if 'video_name' in v
            ])
            print(f"Videos with annotations: {len(annotated_video_names)}")
        else:
            print("Warning: annotation file not found, proceeding without action information")

        session_dirs = sorted([d for d in Path(self.base_dir).iterdir() if d.is_dir()])
        skipped_no_annotation = 0

        for session_dir in session_dirs:
            export_py = session_dir / "Export_py"
            if not export_py.exists():
                continue

            video_file = export_py / "Video_pitchshift.mp4"
            gaze_csv = export_py / f"{session_dir.name}_gaze_2d.csv"

            if video_file.exists() and gaze_csv.exists():
                if self.annotation_data and session_dir.name not in annotated_video_names:
                    skipped_no_annotation += 1
                    continue

                self.valid_sessions.append({
                    'name': session_dir.name,
                    'video_path': str(video_file),
                    'gaze_path': str(gaze_csv),
                })

        print(f"\nFound {len(self.valid_sessions)} valid sessions with gaze data AND annotations")
        if skipped_no_annotation > 0:
            print(f"Skipped {skipped_no_annotation} sessions without annotations")

    def iter_tasks(self):
        for session in self.valid_sessions:
            session_name = session['name']
            yield VideoTask(
                name=session_name,
                video_path=session['video_path'],
                gaze_path=session['gaze_path'],
            )

    def load_action_data(self, task):
        video_action_data = None
        if self.annotation_data is not None:
            for video_anno in self.annotation_data:
                if video_anno.get('video_name') == task.name:
                    video_action_data = video_anno.get('events', [])
                    video_action_data = [
                        e for e in video_action_data
                        if e.get('label') == 'Fine grained action'
                    ]
                    print(f"   Loaded {len(video_action_data)} fine-grained actions")
                    break

        if video_action_data is None:
            print(f"   Warning: no fine-grained action annotations found for {task.name}")
        return video_action_data

    def run_task(self, task, action_data, fps, skip_viz=False, viz_only=False):
        process_single_video_holoassist(
            task.video_path,
            task.gaze_path,
            self.output_dir,
            action_data,
            fps=fps,
            skip_viz=skip_viz,
            viz_only=viz_only,
            video_name=task.name,
        )


DATASET_ADAPTER_REGISTRY = {
    'egtea': EgteaAdapter,
    'ego4d': Ego4DAdapter,
    'egoexo': EgoExoAdapter,
    'egoexo-lab': EgoExoLabAdapter,
    'kitchen': EgoExoKitchenAdapter,
    'holoassist': HoloAssistAdapter,
}


def get_adapter_class(dataset):
    try:
        return DATASET_ADAPTER_REGISTRY[dataset]
    except KeyError as exc:
        choices = ", ".join(DATASET_ADAPTER_REGISTRY.keys())
        raise KeyError(f"Unknown dataset adapter: {dataset}. Available datasets: {choices}") from exc
