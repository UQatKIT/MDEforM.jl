#!/bin/bash

#SBATCH --job-name=MDE
#SBATCH --time=20:00:00
#SBATCH --mem=10000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --output="MDE.out"
#SBATCH --error="MDE_err.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=df2994@kit.edu

export JULIA_NUM_THREADS=40
julia --project=. sim/simulation.jl