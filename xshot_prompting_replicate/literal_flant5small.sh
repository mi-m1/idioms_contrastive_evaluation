#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --account=dcs-acad6
#SBATCH --reservation=dcs-acad6
#SBATCH --time=1-00:00:00
#SBATCH -o literal_flant5small.out
#SBATCH --mail-user=zmi1@sheffield.ac.uk
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL


module load Anaconda3/2022.10
module load CUDA/10.1.243
source activate /mnt/parscratch/users/acq22zm/anaconda/.envs/semeval-t4
export HF_HOME=/mnt/parscratch/users/acq22zm/.cache

LD_LIBRARY_PATH=""

# python flant5small.py \
# --model "google/flan-t5-xxl" \
# --seed 1234 \
# --eval_dataset "dataset/gpt4_t08_sentences_verifier_annotations_accepted_only.csv" \

python hf_1shot.py --model_name google/flan-t5-small --eval_dataset ../prompting/dataset/literal_1032.csv --model_abr flant5small --setting literal
