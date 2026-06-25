import os
import time

from tqdm import tqdm

from benchmark.Benchmark import Benchmark, time_to_seconds


class StreamingBenchRemind_StreamGaze(Benchmark):
    """
    Benchmark for StreamGaze format data - Proactive/Remind Tasks
    Uses OVO-Bench style evaluation with multiple test points
    """

    def eval(self, data, model, output_path):
        """
        Evaluate model on StreamGaze format data for Proactive/Remind tasks.
        """
        for subset in tqdm(data, desc="Evaluating Proactive/Remind Tasks"):
            video_path = subset.get("video_path", "")
            file, video_name = self.resolve_video_file(video_path)

            if not os.path.exists(file):
                print(f"⚠️  Video not found: {file}")
                continue

            for question in subset["questions"]:
                test_info = question.get("test_info", [])
                if not test_info:
                    print(f"Warning: No test_info found for {question.get('target_object', 'unknown')}, skipping...")
                    continue

                first_appearance = question.get("first_appearance", "0:00")
                target_object = question.get("target_object", "unknown")
                inp = self.build_remind_prompt(question["question"])

                print(f"\n  Video: {video_name}, Object: {target_object}, First appearance: {first_appearance}")

                for i, test in enumerate(test_info):
                    if "response" in test:
                        continue

                    realtime = test["realtime"]
                    eval_type = test["type"]
                    start_time = 0
                    end_time = time_to_seconds(realtime)

                    print(
                        f"    [{i+1}/{len(test_info)}] Realtime: {realtime} (0s → {end_time}s), Expected: {'Yes' if eval_type == 1 else 'No'}"
                    )

                    time_s = time.time()
                    salience_map_path = test.get("salience_map_path", None)

                    response, response_time_sec = self.get_model_response(
                        model,
                        file,
                        inp,
                        start_time,
                        end_time,
                        max(0, start_time - 1),
                        False,
                        False,
                        salience_map_path,
                    )

                    time_e = time.time()
                    timecost = time_e - time_s

                    test["response"] = response
                    test["response_time_sec"] = response_time_sec
                    test["cost"] = timecost

                    print(f"       Response: {response}, Time: {response_time_sec:.2f}s")

                self.save_json(output_path, data, indent=4)

        print(f"✅ Evaluation complete! Results saved to {output_path}")
