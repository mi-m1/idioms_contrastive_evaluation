from transformers import AutoModelForCausalLM, AutoTokenizer



from tqdm import tqdm
import re
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
import argparse
import torch
import random
import numpy as np
from literal_flant5xxl import *

parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, help="Huggingface model ID")
parser.add_argument("--seed", help="Number of runs")
parser.add_argument("--eval_dataset", help="Path to eval dataset file")
parser.add_argument("--model_abr", help="model_abbreviation")
parser.add_argument("--setting", help="figurative or literal")
args = parser.parse_args()

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-hf")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")

message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
# inputs = {k: v.to('cuda') for k,v in inputs.items()}
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])


dataset_location = args.eval_dataset
df = pd.read_csv(dataset_location)


from tqdm import tqdm

def run_predictions(idioms, sentence, model, setting):
# def run_and_save_predictions(idioms, sentence, model, setting):
    idiom_pred = []
    issues = []
    idiom_sent_dict= list(zip(idioms, sentence))

    for idiom, sentence in tqdm(idiom_sent_dict):

        my_instruction = "Is the meaning of expression idiomatic or literal? If used idiomatically, answer 'i', if literally, answer 'l'."
        expression = f"Expression: {idiom}"
        sentence = f"Sentence: {sentence}"

        prompt = list(my_instruction + expression + sentence)

        inputs = tokenizer(message=prompt, return_tensors='pt', return_token_type_ids=False)
        response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    
        response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
        print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])

        idiom_pred.append(tuple([idiom, tokenizer.batch_decode(response, skip_special_tokens=True)[0]]))
    
    return idiom_pred, issues

