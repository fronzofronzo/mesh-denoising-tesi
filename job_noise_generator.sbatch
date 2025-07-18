#!/bin/bash
#SBATCH --job-name=testing_fronzoni
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriele.fronzoni@studio.unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=31G
#SBATCH --partition=l40
#SBATCH --output=log_%j.txt
#SBATCH --chdir=/scratch.hpc/gabriele.fronzoni/tesi
#SBATCH --gres=gpu:1

venv/bin/python3 noise_generator.py "$MESH" "$NOISE" "$IMPULSIVE"
