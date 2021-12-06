#!/bin/bash
#SBATCH --job-name=job_test    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=heting.wang@ufl.edu     # Where to send mail, change it to your email address
#SBATCH --nodes=1	
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=4
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --mem=50gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:geforce:2
#SBATCH --time=02:00:00

pwd; hostname; date

module load cuda/11.1.0 python 

echo "Random unknown initialization for first 100 samples per class distillation on Fashion"

CUDA_LAUNCH_BLOCKING=1 python /blue/cis6940/heting.wang/TML2/dataset-distillation/main.py --mode distill_basic --dataset FASHION1000 --arch LeNet --batch_size 64
date