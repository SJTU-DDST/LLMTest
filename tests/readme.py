from LLMTest import LLMTest

# from LLMTest import change_log_level
# change_log_level("DEBUG")

def LLM(prompts):
    return ["Paris"] * len(prompts)

tester = LLMTest()
batch_id, prompts = tester.get()
answers = LLM(prompts)
score = tester.score(batch_id, answers)

print(score)