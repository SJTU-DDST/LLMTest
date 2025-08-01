import os
import re
from datasets import load_dataset
from rouge_score import rouge_scorer

from .logger import logger
from .dataset import DATASET_CONFIG


class LLMTest:

    def __init__(
        self, dataset_path, dataset_name=None
    ):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        if dataset_path not in DATASET_CONFIG:
            raise ValueError(f"Dataset path '{dataset_path}' not found in DATASET_CONFIG.")
        if dataset_name not in DATASET_CONFIG[dataset_path]["__names__"]:
            raise ValueError(f"Dataset name '{dataset_name}' not found in DATASET_CONFIG for path '{dataset_path}'. Available names: {DATASET_CONFIG[dataset_path]['__names__']}")

        self.pos = 0
        self.batch_start_size_cache = {}

        CONFIG_KEY = "__default__" if dataset_name not in DATASET_CONFIG[dataset_path] else dataset_name
        CONFIG = DATASET_CONFIG[dataset_path].get(CONFIG_KEY, {})

        self.test_class = CONFIG.get("test_class", "test")

        self.question_key = CONFIG.get("question_key", "question")
        self.question_key_2 = CONFIG.get("question_key_2", None)

        self.answer_key = CONFIG.get("answer_key", "answer")
        self.choice_key = CONFIG.get("choice_key", "choices")
        self.choice_key_out = CONFIG.get("choice_key_out", True)
        self.should_add_answer_prompt = CONFIG.get("should_add_answer_prompt", False)

        # mutually exclusive
        self.have_different_answers = CONFIG.get("have_different_answers", False)
        self.many_question2_and_answers = CONFIG.get("many_question2_and_answers", False)

        # mutually exclusive
        self.is_guess_next = CONFIG.get("is_guess_next", False)
        self.is_choice = CONFIG.get("is_choice", False)
        self.is_multi_choice = CONFIG.get("is_multi_choice", False)



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
            raise ValueError(f"Batch ID '{batch_id}' not found in cache. use get() first.")
        test_class, pos, size = self.batch_start_size_cache[batch_id]
        dataset_cut = self.dataset[test_class][pos:pos + size]
        questions = dataset_cut[self.question_key]
        if self.is_guess_next:
            questions = [" ".join(q.split()[:-1]) + " " for q in questions]
        if self.question_key_2 is not None:
            questions_2 = dataset_cut[self.question_key_2]
            if self.many_question2_and_answers:
                questions_2 = ["?\n".join(q2) for q2 in questions_2]
            questions = [f"{q}\n{q2}" for q, q2 in zip(questions, questions_2)]
        if (self.is_choice or self.is_multi_choice) and self.choice_key_out:
            choice_prompt = dataset_cut[self.choice_key]
            LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            choice_prompt = [f"\nChoices: {'  '.join([f'{LABELS[i]}, {c}' for i, c in enumerate(choices)])}" for choices in choice_prompt]
            questions = [f"{q} {c}" for q, c in zip(questions, choice_prompt)]
        if self.should_add_answer_prompt:
            questions = [f"{q}\nAnswer: " for q in questions]
        return questions

    def get_truths(self, batch_id: str) -> list[list[str]]:
        if batch_id not in self.batch_start_size_cache:
            raise ValueError(f"Batch ID '{batch_id}' not found in cache. use get() first.")
        test_class, pos, size = self.batch_start_size_cache[batch_id]
        dataset_cut = self.dataset[test_class][pos:pos + size]
        if self.is_guess_next:
            answers = [q.split()[-1] for q in dataset_cut[self.question_key]]
            return [[ans] for ans in answers]
        answers = dataset_cut[self.answer_key]
        if self.many_question2_and_answers:
            answers = ["\n".join(ans) for ans in answers]
        elif self.have_different_answers:
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

    @staticmethod
    def __check_single_choice(answer: str, truths: list[str]) -> float:
        answer = answer.strip().lower()
        truths = [truth.strip().lower() if isinstance(truth, str) else chr(truth + ord('a')) for truth in truths]
        if len(answer) == 1:
            logger.debug(f"Good answer: '{answer}', truths: {truths}")
            if answer in truths:
                logger.debug(f"Matched")
                return 1.0
            else:
                logger.debug(f"Not match")
                return 0.0
        match = re.search(r"(?:answer|result|option|答案).*?([a-e])", answer)
        logger.debug(f"Use answer: '{answer}', truths: {truths}")
        if match:
            data = match.group(1)
            logger.debug(f"Extracted data: '{data}'")
            if data in truths:
                return 1.0
        else:
            data = answer[0]
            logger.debug(f"Not found, use first char: '{data}'")
            if data in truths:
                return 1.0
        return 0.0

    def f1_score(self, batch_id: str, answers: list[str]) -> float:
        total_score = 0
        total_len = self.batch_start_size_cache[batch_id][2]
        for i in range(total_len):
            truths = self.get_truths(batch_id)[i]
            now_score = self.__check_f1_score(answers[i], truths)
            total_score += now_score
            logger.debug(f"answer: {answers[i]}, truths: {truths}")
            logger.debug(f"Batch ID: {batch_id}, Index: {i}, F1 Score: {now_score}")
        result = total_score / total_len if total_len > 0 else 0.0
        logger.info(f"Total F1 Score for Batch ID {batch_id}: {result}")
        return result

    def rogue_l(self, batch_id: str, answers: list[str]) -> float:
        total_score = 0
        total_len = self.batch_start_size_cache[batch_id][2]
        for i in range(total_len):
            truths = self.get_truths(batch_id)[i]
            now_score = self.__check_rogue_l(answers[i], truths)
            total_score += now_score
            logger.debug(f"answer: {answers[i]}, truths: {truths}")
            logger.debug(f"Batch ID: {batch_id}, Index: {i}, Rouge-L Score: {now_score}")
        result = total_score / total_len if total_len > 0 else 0.0
        logger.info(f"Total Rouge-L Score for Batch ID {batch_id}: {result}")
        return result

    def single_choice_score(self, batch_id: str, answers: list[str]) -> float:
        total_score = 0
        total_len = self.batch_start_size_cache[batch_id][2]
        for i in range(total_len):
            truths = self.get_truths(batch_id)[i]
            now_score = self.__check_single_choice(answers[i], truths)
            total_score += now_score
            logger.debug(f"answer: {answers[i]}, truths: {truths}")
            logger.debug(f"Batch ID: {batch_id}, Index: {i}, Single Choice Score: {now_score}")
        result = total_score / total_len if total_len > 0 else 0.0
        logger.info(f"Total Single Choice Score for Batch ID {batch_id}: {result}")
        return result

    def score(self, batch_id: str, answers: list[str]) -> dict[str, float]:

        if self.is_multi_choice:
            # TODO
            print("Multi-choice scoring is not implemented yet.")
            return { "accuracy": 0.0 }

        elif self.is_choice:
            return { "accuracy": self.single_choice_score(batch_id, answers) }

        elif self.is_guess_next:
            return { "accuracy": self.f1_score(batch_id, answers) }

        return {
            "f1_score": self.f1_score(batch_id, answers),
            "rogue_l": self.rogue_l(batch_id, answers)
        }
