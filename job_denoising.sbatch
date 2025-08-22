#!/bin/bash
#SBATCH --job-name=denoising
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriele.fronzoni@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --partition=rtx2080
#SBATCH --output=log_sbatch/log_denoising_%j.txt
#SBATCH --chdir=/scratch.hpc/gabriele.fronzoni/tesi
#SBATCH --gres=gpu:1

venv/bin/python3 mesh_denoising_main.py "$MESH" "$NOISE" "$ITERATIONS" "$DUMPING"
