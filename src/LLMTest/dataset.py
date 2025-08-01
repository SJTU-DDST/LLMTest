DATASET_CONFIG = {
    "cais/mmlu": {
        "__names__": [
            'abstract_algebra', 'all', 'anatomy', 'astronomy', # 'auxiliary_train',
            'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science',
            'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts',
            'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history',
            'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics',
            'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
            'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging',
            'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
            'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
            'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
            'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
            'virology', 'world_religions'
        ],
        "__default__": {
            "test_class": "test",  # [default: "test"]
            # "features": ['question', 'subject', 'choices', 'answer'],
            "question_key": "question",  # [default: "question"]
            "answer_key": "answer",  # [default: "answer"]
            "choice_key": "choices",  # [default: "choices"]
            "choice_key_out": True,  # [default: True]  if True, choices are out of the question and should be appended to the question
            "should_add_answer_prompt": True,  # [default: False]  if True, append "\nAnswer: " to question
            "have_different_answers": False,  # [default: False]  if True, answers are a list of strings
            "is_choice": True,  # [default: False]
        },
    },

    "openai_humaneval": {
        "__names__": [None],
        "__default__": {
            "test_class": "test",
            # "features": ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'],
            "question_key": "prompt",
            "answer_key": "canonical_solution",
        },
    },

    "cimec/lambada": {
        "__names__": ['plain_text'],
        "__default__": {
            "test_class": "test",
            # "features": ['text', 'domain'],
            "question_key": "text",
            "is_guess_next": True,  # [default: False]  if True, the question is a text and the answer is the next word
        },
    },

    # this need to be download first using datasets<=3.6.0
    # datasets==4.0.0 will occur error
    "L4NLP/LEval": {
        "__names__": [
            # "coursera",  # multi + many
            # "gsm100",  # math (answer)
            # "quality",  # single + many
            # "tpo",  # single + many
            "topic_retrieval_longchat",  # f1/rl
            "financial_qa",  # f1/rl
            "gov_report_summ",  # f1/rl
            "legal_contract_qa",  # f1/rl
            "meeting_summ",  # f1/rl
            "multidoc_qa",  # f1/rl
            "narrative_qa",  # f1/rl
            "natural_question",  # f1/rl
            "news_summ",  # f1/rl
            "paper_assistant",  # f1/rl
            "patent_summ",  # f1/rl
            "review_summ",  # f1/rl
            "scientific_qa",  # f1/rl
            "tv_show_summ"  # f1/rl
        ],
        "__default__": {
            "test_class": "test",
            # "features": ['instructions', 'input', 'outputs', 'source', 'evaluation'],
            "question_key": "input",
            "question_key_2": "instructions",  # [default: None]
            "answer_key": "outputs",
            "should_add_answer_prompt": True,
            "many_question2_and_answers": True,  # [default: False]  if True, question_2 and answers are a list of strings, CANNOT use with have_different_answers
        },
    },
}