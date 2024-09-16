#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH -o figurative_flant5xl_1shot.out
#SBATCH --partition=gpu
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

# python flant5xl.py \
# --model "google/flan-t5-xxl" \
# --seed 1234 \
# --eval_dataset "dataset/gpt4_t08_sentences_verifier_annotations_accepted_only.csv" \

python hf_1shot.py --model_name google/flan-t5-xl --eval_dataset ../prompting/dataset/figurative_1032.csv --model_abr flant5xl --setting figurative
