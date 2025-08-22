#!/bin/bash
#SBATCH --job-name=data_generation
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriele.fronzoni@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --partition=rtx2080
#SBATCH --output=log_%j.txt
#SBATCH --chdir=/scratch.hpc/gabriele.fronzoni/tesi


venv/bin/python3 main_patch_generator.py "$MESH" "$LEVEL" --filter_iterations "$FILTER_ITER"


