#!/bin/bash
#SBATCH --job-name=FCN_TRAIN
#SBATCH --account=PAS0536
#SBATCH --time=08:00:00
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --output=/users/PAS0536/subhransu/deployables/FCN/fcn_output.txt
module load cuda/11.8.0
source /fs/ess/PAS0536/Subhransu/miniconda3/bin/activate
export PYTHONNOUSERSITE=true
conda activate img-seg
cd /users/PAS0536/subhransu/deployables/FCN/
python train.py && python inference.py
