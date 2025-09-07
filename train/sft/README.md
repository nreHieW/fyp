## Setup
```bash
bash setup.sh
```

## Train
```bash
cd LLaMA-Factory
llamafactory-cli train config.yaml
```

## VLLM 
```bash
vllm serve train/sft/LLaMA-Factory/saves/qwen3-4b-instruct-2507/full/sft --data-parallel-size 4
```

## LiveCodeBench
install uv todo 
```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
source .venv/bin/activate

uv pip install -e .
uv pip install datasets==3.6

```

To run the baseline, add the following to `LiveCodeBench/lcb_runner/lm_styles.py`

```python 
    ## Qwen3 4B Instruct
    LanguageModel(
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen3-4b-Instruct-2507",
        LMStyle.CodeQwenInstruct,
        datetime(2025, 7, 25),
        link="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
    ),
```
and run the following command
```bash
python -m lcb_runner.runner.main --model Qwen/Qwen3-4B-Instruct-2507 --scenario codegeneration --evaluate --release_version v6 --codegen_n 1
```