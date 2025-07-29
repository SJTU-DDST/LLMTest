import os
import re
from datasets import load_dataset
from rouge_score import rouge_scorer

from .logger import logger
from .dataset import DATASET_CONFIG


class LLMTest:

    def __init__(
        self, dataset_path="openai_humaneval", dataset_name=None
    ):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        if dataset_name is None:
            dataset_name = dataset_path
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        if dataset_path not in DATASET_CONFIG:
            raise ValueError(f"Dataset path '{dataset_path}' not found in DATASET_CONFIG.")
        if dataset_name not in DATASET_CONFIG[dataset_path]["__names__"]:
            raise ValueError(f"Dataset name '{dataset_name}' not found in DATASET_CONFIG for path '{dataset_path}'.")

        self.pos = 0
        self.batch_start_size_cache = {}

        CONFIG = DATASET_CONFIG[dataset_path]["__default__"] if dataset_name not in DATASET_CONFIG[dataset_path] else DATASET_CONFIG[dataset_path][dataset_name]
        self.test_class = CONFIG.get("test_class", "test")
        self.question_key = CONFIG.get("question_key", "question")
        self.answer_key = CONFIG.get("answer_key", "answer")
        self.choice_key = CONFIG.get("choice_key", "choices")
        self.should_add_answer_prompt = CONFIG.get("should_add_answer_prompt", False)
        self.have_different_answers = CONFIG.get("have_different_answers", False)
        self.is_choice = CONFIG.get("is_choice", False)
        self.is_multi_choice = CONFIG.get("is_multi_choice", False)
        self.choice_key_out = CONFIG.get("choice_key_out", True)

        logger.info(f"Loading dataset from '{dataset_path}' with name '{dataset_name}'")
        self.dataset: dict[str, dict] = load_dataset(dataset_path, dataset_name)
        logger.info("Dataset loaded successfully.")

        if self.test_class not in self.dataset:
            raise ValueError(f"Test class '{self.test_class}' not found in dataset '{dataset_name}'. Available classes: {list(self.dataset.keys())}")

        logger.debug(f"Dataset info: {self.dataset[self.test_class]}")

    def get(self, size: int | None = None) -> tuple[str, list[str]]:
        test_class = self.test_class
        pos = self.pos
        if size is None:
            size = len(self.dataset[test_class]) - pos
        if pos + size > len(self.dataset[test_class]):
            raise ValueError(f"Not enough data to get {size} items from position {pos}. Requested size: {size}, available: {len(self.dataset[test_class]) - pos}")
        batch_id = f"{test_class}:{pos}:{size}"
        self.batch_start_size_cache[batch_id] = (test_class, pos, size)
        self.pos += size
        return batch_id, self.get_questions(batch_id)

    def get_questions(self, batch_id: str) -> list[str]:
        if batch_id not in self.batch_start_size_cache:
            raise ValueError(f"Batch ID {batch_id} not found in cache. use get() first.")
        test_class, pos, size = self.batch_start_size_cache[batch_id]
        dataset_cut = self.dataset[test_class][pos:pos + size]
        questions = dataset_cut[self.question_key]
        if self.is_choice and self.choice_key_out:
            choice_prompt = dataset_cut[self.choice_key]
            LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            choice_prompt = [f"\nChoices: {'\t'.join([f'{LABELS[i]}, {c}' for i, c in enumerate(choices)])}" for choices in choice_prompt]
            questions = [f"{q} {c}" for q, c in zip(questions, choice_prompt)]
        if self.should_add_answer_prompt:
            questions = [f"{q}\nAnswer: " for q in questions]
        return questions

    def get_truths(self, batch_id: str) -> list[list[str]]:
        if batch_id not in self.batch_start_size_cache:
            raise ValueError(f"Batch ID {batch_id} not found in cache. use get() first.")
        test_class, pos, size = self.batch_start_size_cache[batch_id]
        dataset_cut = self.dataset[test_class][pos:pos + size]
        answers = dataset_cut[self.answer_key]
        if self.have_different_answers:
            return answers
        answers = [[ans] for ans in answers]
        return answers

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

    def f1_score(self, batch_id: str, answers: list[str]) -> float:
        total_score = 0
        total_len = len(answers)
        for i, ans in enumerate(answers):
            truths = self.get_truths(batch_id)[i]
            now_score = self.__check_f1_score(ans, truths)
            total_score += now_score
            logger.debug(f"answer: {ans}, truths: {truths}")
            logger.debug(f"Batch ID: {batch_id}, Index: {i}, F1 Score: {now_score}")
        result = total_score / total_len if total_len > 0 else 0.0
        logger.info(f"Total F1 Score for Batch ID {batch_id}: {result}")
        return result

    def rogue_l(self, batch_id: str, answers: list[str]) -> float:
        total_score = 0
        total_len = len(answers)
        for i, ans in enumerate(answers):
            truths = self.get_truths(batch_id)[i]
            now_score = self.__check_rogue_l(ans, truths)
            total_score += now_score
            logger.debug(f"answer: {ans}, truths: {truths}")
            logger.debug(f"Batch ID: {batch_id}, Index: {i}, Rouge-L Score: {now_score}")
        result = total_score / total_len if total_len > 0 else 0.0
        logger.info(f"Total Rouge-L Score for Batch ID {batch_id}: {result}")
        return result

    def score(self, batch_id: str, answers: list[str]) -> dict[str, float]:
        if self.is_choice and self.is_multi_choice:
            # TODO
            print("Multi-choice scoring is not implemented yet.")
            return { "accuracy": 0 }

        elif self.is_choice:
            # TODO
            print("Single-choice scoring is not implemented yet.")
            return { "accuracy": 0 }

        return {
            "f1_score": self.f1_score(batch_id, answers),
            "rogue_l": self.rogue_l(batch_id, answers)
        }
