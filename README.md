# Minimal Code Editing. 
This repository contains the code for the paper "When Models Edit Too Much: On the Fidelity of Minimal Code Edits".


## Usage
First you will need to install the uv package manager.

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync 
```

To run the evaluation scripts, you will need to first generate the solutions.
```bash
uv run partial_edits/generate_solutions.py --questions_path data/questions/corrupted_solutions_manual_easy_400.jsonl --store_token_info --model <MODEL NAME>
```

Then you can run the evaluation. You will first need to ensure that the Code Evaluator Docker is running. The DockerFile can be found in the [evaluator](./evaluator) folder.
```bash
docker build -t code-evaluator .
docker run -d --name code-evaluator -p 8000:8000 code-evaluator
```

Then you can run the evaluation and the results file will be saved in the [data/code_edits/results](./data/code_edits/results) folder.
```bash
uv run partial_edits/evaluate.py --sample_path data/code_edits/results_<MODEL NAME>_non_reasoning_400.jsonl
```

## Advanced Usage
### UI
The repository provides a simple UI for evaluating the models. The UI can be found in the [partial_edits/ui](./partial_edits/ui) folder. This can be used to open the evaluation results in a browser.

### Corruptions
The corruption script can be found in the [partial_edits/corrupt_data_manual.py](./partial_edits/corrupt_data_manual.py). The specific used corruption script can be found in the [partial_edits/utils/code_corruptor.py](./partial_edits/utils/code_corruptor.py) file.

### Models
The repository provides an interface to use various models from various providers. The models can be found in the [models](./models) folder. The interface can be found in the [models/__init__.py](./models/__init__.py) file.


### Training 
For training, refer to the [train](./train) folder and the corresponding `README.md` file.


## Citation
If you use this repository, please cite:

```bibtex
@misc{lim2026minimalcodeedits,
  title        = {When Models Edit Too Much: On the Fidelity of Minimal Code Edits},
  author       = {Lim, Wei Hern},
  year         = {2026},
  howpublished = {\url{https://github.com/nreHieW/fyp}},
  note         = {GitHub repository}
}
```
