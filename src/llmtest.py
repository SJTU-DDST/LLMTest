class LLMTest:
    use_time = 0

    def __init__(self):
        pass

    @classmethod
    def get(cls) -> tuple[int, str]:
        now_token = cls.use_time
        cls.use_time += 1
        return now_token, "Where is the capital of France?"

    @classmethod
    def score(cls, token: int, answer: str) -> int:
        if answer == "Paris":
            return 100 - token
        return 0
