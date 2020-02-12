#!/bin/bash
#SBATCH --job-name=trainint_test
#SBATCH --nodes=1
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_short
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --mail-user=castella.sergi@gmail.com

module purge
module load 2019
module load CUDA/10.1.243
module load Python/3.7.5-foss-2018b

cp -r "$HOME"/semantic-compression/ "$TMPDIR"

# shellcheck disable=SC2164
cd "$TMPDIR"/semantic-compression/
source venv/bin/activate
RUN_NAME="base$(date +%d_%m_%Y_%H_%M_%S)";
python3 scripts/base_training.py --run-identifier ${RUN_NAME} -thr 1 --tensorboard-dir tensorboard --eval-periodicity 50 >> "output_${RUN_NAME}.txt"
cp "output_${RUN_NAME}.txt" "$HOME"/semantic-compression

RUN_NAME="comp$(date +%d_%m_%Y_%H_%M_%S)";
python3 scripts/comp_training.py --run-identifier ${RUN_NAME} -thr 0.8 --tensorboard-dir tensorboard --eval-periodicity 50 >> "output_${RUN_NAME}.txt"
cp "output_${RUN_NAME}.txt" "$HOME"/semantic-compression

cp "tensorboard/" "$HOME"/tensorboard
