#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=dcs-acad6
#SBATCH --reservation=dcs-acad6
#SBATCH --time=1-00:00:00
#SBATCH -o l_llama213bchat.out
#SBATCH --mail-user=zmi1@sheffield.ac.uk
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL


module load Anaconda3/2022.10
module load CUDA/10.1.243
source activate /mnt/parscratch/users/acq22zm/anaconda/.envs/semeval-t4
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache
export REPLICATE_API_TOKEN=r8_KeuYswdQLhCIjuwFwknqS9SItqZF6A221xzoO

LD_LIBRARY_PATH=""


python llamas_1shot.py \
--model_name meta/llama-2-13b-chat \
--eval_dataset ../../prompting/dataset/literal_1032.csv \
--model_abr llama213bchat \
--setting literal