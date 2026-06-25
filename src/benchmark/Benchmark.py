import json
import os
from pathlib import Path

from tqdm import tqdm

from utils.data_execution import get_model_response

GAZE_INSTRUCTION = """In this video, the green dot represents the gaze point (where the person is looking), and the red circle represents the field of view (FOV) area.

"""

PROMPT_TEMPLATE = """
Question: {}
Options:
{}
{}
{}
{}
Answer the question with only the letter (A, B, C, or D) of the correct option."""

PROMPT_TEMPLATE_WITHOUT_OPTIONS = """
Question: {}
Analyze the video and provide the answer to the question.
"""

PROMPT_TEMPLATE_REMIND = """You are monitoring a video stream. Based on what you have seen so far in the video, answer the following question.
<video>

Question: {}

If you have already seen the requested object/event, answer "Yes".
If you have not seen it yet, answer "No".

Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""


def parse_timestamp(timestamp_str):
    """Convert MM:SS timestamp string to seconds."""
    parts = timestamp_str.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return float(timestamp_str)


def time_to_seconds(time_str):
    """Convert timestamp like '00:03:10' or '02:22' to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    return int(parts[0])


def format_time(seconds):
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


class Benchmark:
    def __init__(self, data, video_root=None, use_gaze_instruction=False):
        self.video_root = video_root
        self.data = data
        self.use_gaze_instruction = use_gaze_instruction

    def eval(self, data, model, output_path):
        """
        Evaluate the model on the given data and update the data with the model responses.
        data: data input
        model: model to evaluate
        """
        pass

    # def resolve_video_file(self, video_path):
    #     video_name = os.path.basename(video_path)
    #     if os.path.exists(video_path):
    #         return video_path, video_name
    #     return os.path.join(self.video_root, video_name), video_name

    def resolve_video_file(self, video_path):
        video_name = Path(video_path).name
        video_root = Path(self.video_root)

        if Path(video_path).exists():
            return str(Path(video_path)), video_name

        for p in video_root.rglob(video_name):
            if p.is_file():
                return str(p), video_name

        return str(video_root / video_name), video_name

    def gaze_prefix(self):
        return GAZE_INSTRUCTION if self.use_gaze_instruction else ""

    def build_gaze_prompt(self, question_text, options):
        gaze_prefix = self.gaze_prefix()

        if options:
            if not options[0].startswith("A."):
                formatted_options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
            else:
                formatted_options = options

            if len(formatted_options) == 2:
                return (
                    gaze_prefix
                    + f"Question: {question_text}\nOptions:\n{formatted_options[0]}\n{formatted_options[1]}\n\nThe best option is:"
                )
            if len(formatted_options) == 4:
                return (
                    gaze_prefix
                    + PROMPT_TEMPLATE.format(question_text, *formatted_options)
                    + "\n\nThe best option is:"
                )

            options_str = "\n".join(formatted_options)
            return gaze_prefix + f"Question: {question_text}\nOptions:\n{options_str}\n\nThe best option is:"

        return gaze_prefix + PROMPT_TEMPLATE_WITHOUT_OPTIONS.format(question_text) + "\n\nAnswer:"

    def build_remind_prompt(self, question):
        return self.gaze_prefix() + PROMPT_TEMPLATE_REMIND.format(question)

    def save_json(self, output_path, data, indent):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=indent)

    def get_model_response(self, model, file, inp, *args, **kwargs):
        return get_model_response(model, file, inp, *args, **kwargs)

    def get_gaze_time_window(self, question_time):
        raise NotImplementedError

    def eval_gaze_questions(self, data, model, output_path, desc):
        results = []

        for entry in tqdm(data, desc=desc):
            video_path = entry.get("video_path", "")
            file, _ = self.resolve_video_file(video_path)

            if not os.path.exists(file):
                print(f"⚠️  Video not found: {file}")
                results.append(entry)
                continue

            questions = entry.get("questions", [])
            if not questions:
                print("⚠️  No questions found in entry")
                results.append(entry)
                continue

            result_entry = entry.copy()
            result_entry["model_predictions"] = []

            for question_data in questions:
                question_text = question_data.get("question", "")
                time_stamp = question_data.get("time_stamp", "0:00")
                options = question_data.get("options", [])

                question_time = parse_timestamp(time_stamp)
                inp = self.build_gaze_prompt(question_text, options)
                start_time, end_time = self.get_gaze_time_window(question_time)

                response, response_time = self.get_model_response(
                    model,
                    file,
                    inp,
                    start_time=start_time,
                    end_time=end_time,
                    question_time=question_time + 0.1,
                    salience_map_path=None,
                )

                prediction = {
                    "question": question_text,
                    "time_stamp": time_stamp,
                    "model_prediction": response,
                    "model_response_time_sec": response_time,
                    "model_response_time_formatted": f"{int(response_time // 60)}:{int(response_time % 60):02d}",
                }
                result_entry["model_predictions"].append(prediction)

            results.append(result_entry)
            self.save_json(output_path, results, indent=2)

        print(f"✅ Evaluation complete! Results saved to {output_path}")
