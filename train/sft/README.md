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