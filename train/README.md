# Training
The files in this folder are for training models. It is recommended to do this in a separate project directory to avoid conflicts with the current project dependencies.

## SFT
We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for SFT. First run the setup script which will install the dependencies and prepare the dataset. You will need to provide your Weights and Biases API key, and your HuggingFace API key.
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
We use [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) for RL. You will first need to install Astral's [uv](https://github.com/astral-sh/uv) package manager. We specifically use commit `5d7146b` of Prime-RL.

```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
cd prime-rl
git checkout 5b44d8283c08c9dae9003ae4e3c7138bde668c37
source ~/.bashrc
uv add wandb==0.25
uv run wandb login <YOUR WANDB API KEY>
uv pip install -e ../fyp/train/environments/deepcoder_minimal_edit/
```

To run the RL training, run the following command. Adjust the config files as needed. Follow the instructions in the [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) repository for more details.
```bash
uv run rl --trainer @ ../fyp/train/configs/train.toml --orchestrator @ ../fyp/train/configs/orch.toml --inference @ ../fyp/train/configs/infer.toml --inference_gpu_ids 0,1,2,3 --trainer_gpu_ids 4,5,6,7 --wandb.name 
```

## Evaluation

#### Partial Edits
First launch the VLLM server and then run the evaluation script.
```bash
vllm serve <MODEL NAME> --data-parallel-size 4

uv run partial_edits/generate_solutions.py --questions_path data/questions/corrupted_solutions_manual_easy_400.jsonl --store_token_info --model local/<MODEL NAME>
```

#### LiveCodeBench
A LiveCodeBench fork has been provided in this folder. This fork fixes several bugs in the original LiveCodeBench repository and adds support for relevant models. For more information on the bugs fixed, see [this blog post](https://blog.collinear.ai/p/lcb-bug-fixes). To add new models you will need to edit the [./lcb_runner/lm_styles.py](./lcb_runner/lm_styles.py) file.

First install the dependencies. 
```bash
uv pip install -e .
uv pip install datasets
```

To run the baseline, run the following command. This should give you a score of `0.32571428571428573`.
```bash
uv run python -m lcb_runner.runner.main --model Qwen/Qwen3-4B-Instruct-2507 --scenario codegeneration --evaluate --release_version v6 --n 1
```
To evaluate the trained model, run the following command.
```bash
uv run python -m lcb_runner.runner.main --scenario codegeneration --evaluate --release_version v6 --n 1 --model <MODEL NAME> 
```





