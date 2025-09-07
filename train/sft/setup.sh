sudo apt-get update
python -m pip install --upgrade pip wheel setuptools

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install deepspeed==0.14.4
pip install --no-build-isolation flash-attn==2.7.3
pip install datasets 
pip install vllm
pip install ray

cd ../
rm -rf LLaMA-Factory/data/

python prepare_dataset.py 
mv config.yaml LLaMA-Factory/

pip install wandb

wandb login
export WANDB_PROJECT="partial-edits"

huggingface-cli login

# cd LLaMA-Factory
# llamafactory-cli train config.yaml