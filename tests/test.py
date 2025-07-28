from LLMTest import LLMTest

def LLM(prompts):
    return "Paris"

batch_id, prompts = LLMTest.get()
answers = LLM(prompts)
score = LLMTest.score(batch_id, answers)

print(score)
