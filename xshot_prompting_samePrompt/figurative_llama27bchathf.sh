#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=dcs-acad6
#SBATCH --reservation=dcs-acad6
#SBATCH --time=1-00:00:00
#SBATCH -o figurative_llama27bchathf_1shot.out
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


python llama2_1shot.py \
--model_name meta-llama/Llama-2-7B-chat-hf \
--eval_dataset ../prompting/dataset/figurative_1032.csv \
--model_abr llama27bchathf \
--setting figurative