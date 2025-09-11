# Training

## SFT
First run the setup script which will install the dependencies and prepare the dataset. You will need to provide your Weights and Biases API key, and your HuggingFace API key.
```bash
bash setup.sh
```

Edit the `config.yaml` file to your desired settings and run 
```bash
uv run --prerelease=allow llamafactory-cli train config.yaml
```

To upload the model to HuggingFace, run
```bash
hf upload [HF HUB REPO NAME] saves/qwen3-4b-instruct-2507 .
```

## RL
[WIP]

First install and set up [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl). It is recommended to do this in a separate project directory to avoid conflicts with the current project dependencies.

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
cd prime-rl
prime login
uv sync --no-build-isolation-package flash-attn
uv sync
vf-install deepcoder-minimal-edit -p fyp/train/environments/deepcoder_minimal_edit
```

To run the RL training, run the following command. Adjust the config files and the number of GPUs as needed. Follow the instructions in the [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) repository for more details.
```bash
uv run rl --trainer @ ../fyp/train/configs/train.toml --orchestrator @ ../fyp/train/configs/orch.toml --inference @ ../fyp/train/configs/infer.toml --trainer_gpus 2 --inference_gpus 2
```

## Evaluation

#### Partial Edits
First launch the VLLM server and then run the evaluation script.
```bash
vllm serve nreHieW/partial-edits-sft-qwen3-4b-instruct-2507 --data-parallel-size 4

uv run partial_edits/generate_solutions.py --questions_path data/questions/corrupted_solutions_manual_easy_400.jsonl --store_token_info --model local/nreHieW/partial-edits-sft-qwen3-4b-instruct-2507
```

#### LiveCodeBench
A LiveCodeBench fork has been provided in this folder. This fork fixes several bugs in the original LiveCodeBench repository and adds support for relevant models. For more information on the bugs fixed, see [this blog post](https://blog.collinear.ai/p/lcb-bug-fixes).

First install the dependencies. 
```bash
uv pip install -e .
uv pip install datasets==3.6
```

To run the baseline, run the following command. This should give you a score of `0.3053435114503817`.
```bash
uv run python -m lcb_runner.runner.main --model Qwen/Qwen3-4B-Instruct-2507 --scenario codegeneration --evaluate --release_version v6 --n 1
```
To evaluate the trained model, run the following command.
```bash
uv run python -m lcb_runner.runner.main --model nreHieW/partial-edits-sft-qwen3-4b-instruct-2507 --scenario codegeneration --evaluate --release_version v6 --n 1
```

#### TODO 
- handle prompt length for prime rl 
