from LLMTest import LLMTest, change_log_level

# change_log_level("DEBUG")

def LLM(prompts):
    return ["Paris"] * len(prompts)

batch_id, prompts = LLMTest.get()
answers = LLM(prompts)
score = LLMTest.score(batch_id, answers)

print(batch_id, prompts)
print(answers)
print(score)
