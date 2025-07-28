import re
import sys
from rouge_score import rouge_scorer
from .logger import logger

class LLMTest:
    batch_id = 32768

    def __init__(self):
        pass

    # TODO
    @classmethod
    def get(cls) -> tuple[int, list[str]]:
        now_batch_id = cls.batch_id
        cls.batch_id += 17
        return now_batch_id, cls.get_questions(now_batch_id)

    # TODO: link to dataset
    @staticmethod
    def get_questions(batch_id: int) -> list[str]:
        questions = {
            32768: ["Where is the capital of France?", "What is the capital of England?"],
        }
        if batch_id not in questions:
            raise ValueError(f"Question for batch ID {batch_id} not found.")
        return questions[batch_id]

    # TODO: link to dataset
    @staticmethod
    def get_truths(batch_id: int) -> list[list[str]]:
        truths = {
            32768: [["Paris", "It is Paris"], ["London", "It is London"]],
        }
        if batch_id not in truths:
            raise ValueError(f"Truths for batch ID {batch_id} not found.")
        return truths[batch_id]

    @staticmethod
    def __check_f1_score(answer: str, truths: list[str]) -> float:
        def normalize_text(s):
            s = s.lower()
            s = re.sub(r'\W+', ' ', s)
            return s.strip()
        pred = normalize_text(answer)
        max_f1 = 0.0
        for gold in truths:
            gold = normalize_text(gold)
            pred_toks = pred.split()
            gold_toks = gold.split()
            common = set(pred_toks) & set(gold_toks)
            if not common:
                continue
            prec = len(common) / len(pred_toks)
            rec = len(common) / len(gold_toks)
            if prec + rec == 0:
                f1 = 0
            else:
                f1 = 2 * prec * rec / (prec + rec)
            max_f1 = max(max_f1, f1)
        return max_f1

    @staticmethod
    def __check_rogue_l(answer: str, truths: list[str]) -> float:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        pred = answer.strip()
        max_score = 0.0
        for gold in truths:
            gold = gold.strip()
            score = scorer.score(gold, pred)['rougeL'].fmeasure
            max_score = max(max_score, score)
        return max_score

    @classmethod
    def f1_score(cls, batch_id: int, answers: list[str]) -> float:
        total_score = 0
        total_len = len(answers)
        for i, ans in enumerate(answers):
            truths = cls.get_truths(batch_id)[i]
            now_score = cls.__check_f1_score(ans, truths)
            total_score += now_score
            logger.debug(f"answer: {ans}, truths: {truths}")
            logger.info(f"Batch ID: {batch_id}, Index: {i}, F1 Score: {now_score}")
        result = total_score / total_len if total_len > 0 else 0.0
        logger.info(f"Total F1 Score for Batch ID {batch_id}: {result}")
        return result

    @classmethod
    def rogue_l(cls, batch_id: int, answers: list[str]) -> float:
        total_score = 0
        total_len = len(answers)
        for i, ans in enumerate(answers):
            truths = cls.get_truths(batch_id)[i]
            now_score = cls.__check_rogue_l(ans, truths)
            total_score += now_score
            logger.debug(f"answer: {ans}, truths: {truths}")
            logger.info(f"Batch ID: {batch_id}, Index: {i}, Rouge-L Score: {now_score}")
        result = total_score / total_len if total_len > 0 else 0.0
        logger.info(f"Total Rouge-L Score for Batch ID {batch_id}: {result}")
        return result

    @classmethod
    def score(cls, batch_id: int, answers: list[str]) -> dict[str, float]:
        return {
            "f1_score": cls.f1_score(batch_id, answers),
            "rogue_l": cls.rogue_l(batch_id, answers)
        }
