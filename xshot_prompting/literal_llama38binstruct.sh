#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH -o literal_llama38binstruct.out
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=80GB
#SBATCH --mail-user=zmi1@sheffield.ac.uk
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL


module load Anaconda3/2022.10
module load CUDA/10.1.243
source activate /mnt/parscratch/users/acq22zm/anaconda/.envs/semeval-t4
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache

LD_LIBRARY_PATH=""

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"


python llama3_1shot.py \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--eval_dataset ../prompting/dataset/literal_1032.csv \
--model_abr llama38binstruct \
--setting literal