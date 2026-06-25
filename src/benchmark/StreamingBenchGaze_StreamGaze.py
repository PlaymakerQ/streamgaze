from benchmark.Benchmark import Benchmark


class StreamingBenchGaze_StreamGaze(Benchmark):
    """
    Benchmark for StreamGaze format data - Present/Future Tasks
    Uses 60-second window [timestamp-60, timestamp]
    """

    def eval(self, data, model, output_path):
        """
        Evaluate model on StreamGaze format data for Present/Future tasks.

        Present/Future tasks use 60-second window before question timestamp.
        """
        self.eval_gaze_questions(data, model, output_path, desc="Evaluating Present/Future Tasks")

    def get_gaze_time_window(self, question_time):
        return max(question_time - 60, 0), question_time
