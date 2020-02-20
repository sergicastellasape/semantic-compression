#!/bin/bash
#SBATCH --job-name=big-run
#SBATCH --nodes=1
#SBATCH --time=7:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=castella.sergi@gmail.com


module purge
module load 2019
module load CUDA/10.1.243
module load Python/3.7.5-foss-2018b

cp -r $HOME/semantic-compression/ $TMPDIR

# shellcheck disable=SC2164
cd $TMPDIR/semantic-compression/
source venv/bin/activate

RUN_NAME="base-$(date +%d_%m_%Y_%H_%M_%S)";
python3 scripts/base_training.py --run-identifier ${RUN_NAME} -thr=1 -lr=0.0001 --tensorboard-dir tensorboard --eval-periodicity 50 --wall-time 3000 --train-compression False --eval-compression False >> outputs/output_${RUN_NAME}.txt
cp outputs/output_${RUN_NAME}.txt $HOME/semantic-compression/outputs/

RUN_NAME="comp-09-$(date +%d_%m_%Y_%H_%M_%S)";
python3 scripts/base_training.py --run-identifier ${RUN_NAME} -thr=0.9  -lr=0.0001 --tensorboard-dir tensorboard --eval-periodicity 50 --wall-time 7000 --train-compression True --eval-compression True >> outputs/output_${RUN_NAME}.txt
cp outputs/output_${RUN_NAME}.txt $HOME/semantic-compression/outputs/

RUN_NAME="comp-08-$(date +%d_%m_%Y_%H_%M_%S)";
python3 scripts/base_training.py --run-identifier ${RUN_NAME} -thr=0.8  -lr=0.0001 --tensorboard-dir tensorboard --eval-periodicity 50 --wall-time 7000 --train-compression True --eval-compression True >> outputs/output_${RUN_NAME}.txt
cp outputs/output_${RUN_NAME}.txt $HOME/semantic-compression/outputs/

RUN_NAME="comp-07-$(date +%d_%m_%Y_%H_%M_%S)";
python3 scripts/base_training.py --run-identifier ${RUN_NAME} -thr=0.7  -lr=0.0001 --tensorboard-dir tensorboard --eval-periodicity 50 --wall-time 7000 --train-compression True --eval-compression True >> outputs/output_${RUN_NAME}.txt
cp outputs/output_${RUN_NAME}.txt $HOME/semantic-compression/outputs/

cp -r $TMPDIR/semantic-compression/tensorboard/ $HOME/semantic-compression
cp -r $TMPDIR/semantic-compression/assets/checkpoints/ $HOME/semantic-compression/assets