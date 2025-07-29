from LLMTest import LLMTest, change_log_level

def LLM(prompts):
    return ["Paris"] * len(prompts)

change_log_level("DEBUG")


tester = LLMTest("cais/mmlu", "high_school_biology")

try:
    batch_id, prompts = tester.get(1000)
except ValueError as e:
    print(f"Test error: {e}")
    print()
else:
    raise ValueError("It should not be able to get 1000 questions at once.")

batch_id, prompts = tester.get(10)
answers = LLM(prompts)
correct = tester.get_truths(batch_id)
score = tester.score(batch_id, answers)

print(batch_id)
print()
print(prompts)
print()
print(answers)
print()
print(correct)
print()
print(score)
print()


tester2 = LLMTest("openai_humaneval")

batch_id, prompts = tester2.get(10)
answers = LLM(prompts)
correct = tester2.get_truths(batch_id)
score = tester2.score(batch_id, answers)

print(batch_id)
print()
print(prompts)
print()
print(answers)
print()
print(correct)
print()
print(score)
print()