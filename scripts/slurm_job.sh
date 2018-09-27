#!/bin/bash
#SBATCH --array=0-255
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ActorCriticDyna
#SBATCH --time=00:15:00
#SBATCH --mem=512M
#SBATCH --output=/home/mzaheer/scratch/classic-control/data/job_output/ActorCriticDyna/%A%a.out
#SBATCH --error=/home/mzaheer/scratch/classic-control/data/job_output/ActorCriticDyna/%A%a.err
module load python35-scipy-stack/2017a
source ~/projects/def-whitem/mzaheer/rl-scratch-env/bin/activate
cd ../
python run.py --idx $SLURM_ARRAY_TASK_ID --config-file /home/mzaheer/scratch/classic-control/config_files/actor_critic_dyna.json
