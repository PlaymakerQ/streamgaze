from model.modelclass import Model


class Dummy(Model):
    def __init__(self):
        pass

    def Run(
        self,
        file,
        inp,
        start_time=None,
        end_time=None,
        question_time=None,
        omni=False,
        proactive=False,
        salience_map_path=None,
    ):
        if 'Answer only with "Yes" or "No"' in inp:
            return "No", 0.0

        return "A", 0.0

    def name(self):
        return "Dummy"
