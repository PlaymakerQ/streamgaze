from benchmark.Benchmark import Benchmark


class StreamingBenchGaze_Past_StreamGaze(Benchmark):
    """
    Benchmark for StreamGaze format data - Past Tasks
    Uses full video from start to question timestamp
    """

    def eval(self, data, model, output_path):
        """
        Evaluate model on StreamGaze format data for Past tasks.

        Past tasks use the full video from 0 to question timestamp.
        """
        self.eval_gaze_questions(data, model, output_path, desc="Evaluating Past Tasks")

    def get_gaze_time_window(self, question_time):
        return 0, question_time
