"""
Common utility functions for filtering
"""

import re
from transformers import AutoProcessor

# Qwen3VL Model Configuration
QWEN3_VL_MODELS = {
    "1B": "Qwen/Qwen3-VL-1B-Instruct",
    "2B": "Qwen/Qwen3-VL-2B-Instruct",
    "4B": "Qwen/Qwen3-VL-4B-Instruct",
    "8B": "Qwen/Qwen3-VL-8B-Instruct",
    "30B": "Qwen/Qwen3-VL-30B-A3B-Instruct",
}
DEFAULT_QWEN_MODEL = "2B"

qwen_model_name = None
qwen_model = None
qwen_processor = None


def resolve_qwen_model_name(model_name):
    """Resolve a model size alias or pass through a full Hugging Face model name."""
    return QWEN3_VL_MODELS.get(model_name, model_name)


def get_qwen_model_class(model_name):
    """Choose the dense or MoE Qwen3-VL model class for the requested checkpoint."""
    if "A3B" in model_name:
        from transformers import Qwen3VLMoeForConditionalGeneration
        return Qwen3VLMoeForConditionalGeneration

    from transformers import Qwen3VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration


def configure_qwen_model(model_name=DEFAULT_QWEN_MODEL):
    """Configure and load the Qwen3VL model used by filtering."""
    global qwen_model_name, qwen_model, qwen_processor

    resolved_model_name = resolve_qwen_model_name(model_name)
    if qwen_model is not None and qwen_processor is not None and qwen_model_name == resolved_model_name:
        return resolved_model_name

    print(f"Loading Qwen3VL model: {model_name} -> {resolved_model_name}")
    model_class = get_qwen_model_class(resolved_model_name)
    qwen_model = model_class.from_pretrained(
        resolved_model_name,
        dtype="auto",
        device_map="auto"
    )
    qwen_processor = AutoProcessor.from_pretrained(resolved_model_name)
    qwen_model_name = resolved_model_name
    print(f"Qwen3VL model loaded successfully: {qwen_model_name}")
    return qwen_model_name

# Human-related keywords
HUMAN_KEYWORDS = ['hand', 'hands', 'finger', 'fingers', 'arm', 'arms', 
                  'foot', 'feet', 'leg', 'legs', 'body', 'face', 'head', 'person']


def parse_time_to_seconds(time_str):
    """Convert MM:SS or HH:MM:SS to seconds"""
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def seconds_to_time_str(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def is_human_related(obj_name):
    """Check if object is human-related"""
    obj_lower = str(obj_name).lower()
    return any(keyword in obj_lower for keyword in HUMAN_KEYWORDS)


def get_qwen_model():
    """Get Qwen3VL model"""
    if qwen_model is None:
        configure_qwen_model()
    return qwen_model

def get_qwen_processor():
    """Get Qwen3VL processor"""
    if qwen_processor is None:
        configure_qwen_model()
    return qwen_processor


def extract_object_name(option_str):
    """Extract object name from option string like 'A. spoon' -> 'spoon'"""
    if '. ' in option_str:
        return option_str.split('. ', 1)[1]
    return option_str


def normalize_object_name(obj_name):
    """Normalize object name: lowercase + underscore to space"""
    obj_name = obj_name.lower()
    obj_name = obj_name.replace('_', ' ')
    return obj_name


def get_gaze_visualization_path(original_video_path):
    """Convert original video path to gaze_visualization path"""
    return original_video_path.replace('/videos/', '/gaze_visualization/')


def time_to_seconds(time_str):
    """Convert time string (MM:SS) to seconds - alias for parse_time_to_seconds"""
    return parse_time_to_seconds(time_str)

