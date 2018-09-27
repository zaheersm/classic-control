#!/usr/bin/env bash

# Get node
salloc --time=12:0:0 --cpus-per-task=48 --account=def-whitem --mem=4000M

# Load singularity
module load singularity/2.5

# Pull the image (if not already exists)
singularity pull --name cc-env.img shub://muhammadzaheer/singularity-recipes:classic-control

# Run jobs (Make sure data/parallel_output exists)
singularity exec -B /scratch/mzaheer/classic-control cc-env.img parallel --results data/parallel_output/ python run.py --idx {1} --config-file config_files/sw_actor_critic_dyna.json  ::: $(seq 0 1024)