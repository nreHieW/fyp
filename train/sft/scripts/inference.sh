#!/bin/bash
#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=00:00:30
#SBATCH --output=myjob_%j.log
#SBATCH --error=myjob_%j.log
##SBATCH --partition=i7-7700
echo "Running job!"
echo "We are running on $(hostname)"
echo "Job started at $(date)"
uv run get_samples_from_base.py
echo "Job ended at $(date)"