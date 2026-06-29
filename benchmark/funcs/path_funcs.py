from pathlib import Path

from constants import Const


PIPELINE_DIR = Const.processed_data_root


def build_output_dir(data_name):
    output_dir = Path(PIPELINE_DIR) / "processed_data" / data_name / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
