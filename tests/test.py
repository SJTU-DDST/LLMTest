from LLMTest import LLMTest, change_log_level

def LLM(prompts):
    return ["Paris", "the final answer is A"] * len(prompts)

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
print(batch_id)
print()
print(prompts)
print()
answers = LLM(prompts)
print(answers)
print()
correct = tester.get_truths(batch_id)
print(correct)
print()
score = tester.score(batch_id, answers)
print(score)
print()


tester = LLMTest("cimec/lambada", "plain_text")

batch_id, prompts = tester.get(10)
print(batch_id)
print()
print(prompts)
print()
answers = LLM(prompts)
print(answers)
print()
correct = tester.get_truths(batch_id)
print(correct)
print()
score = tester.score(batch_id, answers)
print(score)
print()


tester = LLMTest("openai_humaneval")

batch_id, prompts = tester.get(10)
print(batch_id)
print()
print(prompts)
print()
answers = LLM(prompts)
print(answers)
print()
correct = tester.get_truths(batch_id)
print(correct)
print()
score = tester.score(batch_id, answers)
print(score)
print()



# tester = LLMTest("L4NLP/LEval", "natural_question")

# batch_id, prompts = tester.get(10)
# print(batch_id)
# print()
# print(prompts)
# print()
# answers = LLM(prompts)
# print(answers)
# print()
# correct = tester.get_truths(batch_id)
# print(correct)
# print()
# score = tester.score(batch_id, answers)
# print(score)
# print()