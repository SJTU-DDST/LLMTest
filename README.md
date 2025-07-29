# LLMTest

对 LLM 进行性能测试，包括 F1-Score，Rogue-L，困惑度等

以下内容均以 uv 为例

## 安装方式（快速，推荐）

```bash
uv pip install git+ssh://git@github.com/SJTU-DDST/LLMTest.git
```

## 安装方式（可修改）

下载并放入 3rd 文件夹

```bash
git submodule add git@github.com:SJTU-DDST/LLMTest.git 3rd/llmtest
# git submodule update --init --recursive
```

安装

```bash
# uv venv / uv sync
uv pip install -e 3rd/llmtest
```

## 使用方式

创建 `test.py`，写入

```python
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
```

```bash
uv run test.py
```

## 开发

```bash
uv pip install -e .
uv run tests/test.py
```
