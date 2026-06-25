from pathlib import Path
from datetime import datetime


class MyTool:

    @staticmethod
    def get_root_path() -> Path:
        path = Path(__file__).resolve()

        for p in [path] + list(path.parents):
            if (p / ".git").is_dir():
                return p

        raise RuntimeError("Project root (.git) not found")

    @staticmethod
    def set_save_path(
        model_name: str,
        data_name: str,
        save_root_name: str = "save",
        task_name: str | None = None,
    ) -> Path:

        root_path = MyTool.get_root_path()

        save_root = root_path / save_root_name

        if task_name:
            base_path = save_root / task_name / model_name / data_name
        else:
            base_path = save_root / model_name / data_name

        base_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")

        save_path = base_path / timestamp

        save_path.mkdir(parents=False, exist_ok=False)

        return save_path

    @staticmethod
    def print_metric_results(val_metrics, prefix="[Validation Results]"):
        metric_names = list(val_metrics.keys())

        key_width = 10
        metric_width = 10

        print(f"\n\033[1;35m{prefix}\033[0m")

        header = "Metric".ljust(key_width)
        for m in metric_names:
            header += m.upper().rjust(metric_width)
        print(header)

        print("-" * len(header))

        row = "Value".ljust(key_width)
        for m in metric_names:
            row += f"{val_metrics[m]:>{metric_width}.4f}"
        print(row)

        print("=" * len(header))