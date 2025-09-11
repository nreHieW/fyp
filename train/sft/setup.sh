sudo apt-get update
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
export PATH="/root/.cargo/bin:$PATH"

python -m pip install --upgrade pip wheel setuptools

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
uv sync --extra torch --extra metrics --prerelease=allow
uv pip install -e ".[torch,metrics]" --no-build-isolation
uv pip install deepspeed==0.14.4
uv pip install --no-build-isolation flash-attn==2.7.3
uv pip install datasets 
uv pip install vllm
uv pip install ray

cd ../
rm -rf LLaMA-Factory/data/

python prepare_dataset.py 
mv ../configs/config.yaml LLaMA-Factory/

uv pip install wandb

uv run wandb login
export WANDB_PROJECT="partial-edits"

uv run huggingface-cli login

cd LLaMA-Factory
# uv run llamafactory-cli train config.yaml
# uv run --prerelease=allow llamafactory-cli train config.yaml