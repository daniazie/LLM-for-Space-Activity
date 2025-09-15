#!/usr/bin/bash

#SBATCH -J space
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 1-0
#SBATCH -o ./logs/slurm-%A.out
#SBATCH -e ./logs/slurm-err-%A.out

hostname

python - << 'EOF'
import time
i=0
while True:
    i=i+1
    print('>>', i)
    time.sleep(5)
EOF