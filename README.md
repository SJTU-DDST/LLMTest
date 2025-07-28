# LMTest

对 LLM 进行性能测试，包括 F1-Score，Rogue-L，困惑度等

## 安装方式

下载并放入 3rd 文件夹

```bash
git submodule add git@github.com:SJTU-DDST/LLMTest.git 3rd/llmtest
# git submodule update --init --recursive
```

安装（推荐均使用 uv）

```bash
# uv venv / uv sync
uv pip install -e 3rd/llmtest
```

## 使用方式

```python
from LLMTest import LLMTest

def LLM(prompts):
    return "Paris"

batch_id, prompts = LLMTest.get()
answers = LLM(prompts)
score = LLMTest.score(batch_id, answers)

print(score)
```