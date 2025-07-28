class LLMTest:
    batch_id = 32768

    def __init__(self):
        pass

    @classmethod
    def get(cls) -> tuple[int, list[str]]:
        now_batch_id = cls.batch_id
        cls.batch_id += 17
        return now_batch_id, ["Where is the capital of France?", "What is the capital of England?"]

    @classmethod
    def __check_f1_score(cls, batch_id: int, index: int, answer: str) -> float:
        if index == 0 and answer == "Paris":
            return 1.0
        if index == 1 and answer == "London":
            return 1.0
        return 0.0

    @classmethod
    def __check_rogueL(cls, batch_id: int, index: int, answer: str) -> float:
        if index == 0 and answer == "Paris":
            return 1.0
        if index == 1 and answer == "London":
            return 1.0
        return 0.0

    @classmethod
    def f1_score(cls, batch_id: int, answers: list[str]) -> float:
        total_score = 0
        total_len = len(answers)
        for i, ans in enumerate(answers):
            total_score += cls.__check_f1_score(batch_id, i, ans)
        return total_score / total_len if total_len > 0 else 0.0

    @classmethod
    def rogueL(cls, batch_id: int, answers: list[str]) -> float:
        total_score = 0
        total_len = len(answers)
        for i, ans in enumerate(answers):
            total_score += cls.__check_rogueL(batch_id, i, ans)
        return total_score / total_len if total_len > 0 else 0.0

    @classmethod
    def score(cls, batch_id: int, answers: list[str]) -> dict[str, float]:
        return {
            "f1_score": cls.f1_score(batch_id, answers),
            "rogueL": cls.rogueL(batch_id, answers)
        }
