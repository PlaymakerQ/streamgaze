"""Shared config loading helpers for benchmark scripts."""

import json
from pathlib import Path
import argparse


DEFAULT_DATASET_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset.yaml"


def str2bool(value):
    """Parse common string booleans for argparse options."""
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class Config(dict):
    """Small attribute-access config compatible with the local YAML format."""

    def __init__(self, init=None, **kwargs):
        super().__init__()
        if isinstance(init, (str, Path)):
            self._load_from_file(init)
        elif isinstance(init, dict):
            self._load_from_dict(init)
        if kwargs:
            self._load_from_dict(kwargs)

    def _load_from_file(self, path):
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            data = self._load_yaml(text)
        else:
            raise ValueError(f"Unsupported config format: {path}")
        self._load_from_dict(data)

    def _load_from_dict(self, data):
        for key, value in data.items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def _load_yaml(self, text):
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            lines.append(line)
        _, data = self._parse_yaml_block(lines, 0, 0)
        return data

    def _parse_yaml_block(self, lines, index, indent):
        data = {}
        while index < len(lines):
            line = lines[index]
            current_indent = len(line) - len(line.lstrip(" "))
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ValueError(f"Invalid YAML indentation near: {line}")

            key, sep, value = line.strip().partition(":")
            if not sep:
                raise ValueError(f"Invalid YAML line: {line}")

            key = key.strip()
            value = value.strip()
            if value == "":
                index, child = self._parse_yaml_block(lines, index + 1, indent + 2)
                data[key] = child
                continue

            data[key] = self._parse_yaml_scalar(value)
            index += 1
        return index, data

    def _parse_yaml_scalar(self, value):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]
        try:
            if any(ch in value for ch in [".", "e", "E"]):
                return float(value)
            return int(value)
        except ValueError:
            return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def load_dataset_configs(dataset, config_path=DEFAULT_DATASET_CONFIG_PATH):
    """Load shared dataset configs in the same data-name-first style as gaze_vae."""
    all_datasets = Config(config_path)
    if dataset not in all_datasets:
        choices = ", ".join(all_datasets.keys())
        raise KeyError(f"Unknown dataset: {dataset}. Available datasets: {choices}")

    cfg = Config()
    cfg.data = all_datasets[dataset]
    cfg.info = Config({"dataset": dataset})
    return cfg


def get_dataset_choices(config_path=DEFAULT_DATASET_CONFIG_PATH):
    return list(Config(config_path).keys())
