git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install deepspeed
pip install flash-attn --no-build-isolation
pip install datasets 

cd ../
rm -rf LLaMA-Factory/data/

python prepare_dataset.py 
mv config.yaml LLaMA-Factory/

cd LLaMA-Factory
llamafactory-cli train config.yaml