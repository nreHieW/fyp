# sudo apt-get update
# python -m pip install --upgrade pip wheel setuptools

# curl -LsSf https://astral.sh/uv/install.sh | sh
# source ~/.bashrc
# export PATH="/root/.cargo/bin:$PATH"


# git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
uv sync --extra torch --extra metrics --prerelease=allow
source .venv/bin/activate
# uv pip install -e ".[torch,metrics]" --no-build-isolation --prerelease=allow
uv pip install wheel
uv pip install deepspeed==0.14.4
uv pip install --no-build-isolation flash-attn==2.7.3
# uv pip install datasets 
# uv add vllm
uv pip install ray

cd ../
# rm -rf LLaMA-Factory/data/

uv run --with datasets prepare_dataset.py 
# mv ../configs/config.yaml LLaMA-Factory/

uv pip install wandb

uv run wandb login
export WANDB_PROJECT="partial-edits"

uv run huggingface-cli login

cd LLaMA-Factory
# uv run --prerelease=allow llamafactory-cli train config.yaml