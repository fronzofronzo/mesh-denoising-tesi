#!/bin/bash
#SBATCH --job-name=testing_fronzoni
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gabriele.fronzoni@studio.unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --partition=rtx2080
#SBATCH --output=log_%j.txt
#SBATCH --chdir=/scratch.hpc/gabriele.fronzoni/tesi


venv/bin/python3 mesh_reader_optimized.py "$MESH" "$LEVEL"

