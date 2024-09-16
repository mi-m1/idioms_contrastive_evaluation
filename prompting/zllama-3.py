from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import os
import argparse
import torch
import random
import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Huggingface model ID")
parser.add_argument("--eval_dataset", help="Path to eval dataset file")
parser.add_argument("--model_abr", help="model_abbreviation")
parser.add_argument("--setting", help="figurative or literal")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name, token="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", device_map="auto", padding="left")
model = AutoModelForCausalLM.from_pretrained(args.model_name, token="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",device_map="auto", torch_dtype=torch.float16, resume_download=True,)

dataset_location = args.eval_dataset
df = pd.read_csv(dataset_location)
print(df.shape)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def get_answer_from_model(prompt):

    # strings_after_A= []

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=20,)
    string = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return string


def save_predictions(idioms_preds, setting, path, model_abr, run):
   filename = f"{path}{setting}_{model_abr}_{run}.csv"
   
   with open(filename,'w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['idiom','sentence'])
    for row in idioms_preds:
        csv_out.writerow(row)

def extract_answer(idiom, output_string):
    match = re.search(r'\[\s*[\'\"](i|l)[\'\"]\s*\]', output_string)
    if match:
        result = match.group(1)
        # print(f"this is result")
        # print(result)
        return result
    else:
        print(f"No match found for: {idiom}")

def run_predictions(idioms, sentence, model_abr, setting):
# def run_and_save_predictions(idioms, sentence, model, setting):
  idioms_preds_p1 = []
  idioms_preds_p2 = []
  idioms_preds_p3 = []

  issues = []
  idiom_sent_dict= list(zip(idioms, sentence))

  for idiom, sentence in tqdm(idiom_sent_dict):

    prompt1 = f"<s>[INST] <<SYS>>\nIs the expression used literally or figuratively? Only answer with lowercase 'i' for figurative or 'l' for literal only. Generate a Python list containing the letter.\n<</SYS>>\n The expression is '{idiom}'. The sentence is '{sentence}'[/INST]"
    prompt2 = f"<s>[INST] <<SYS>>\nGiven a contextual sentence and a expression, tell me if the expression is used figurative or literally. Either generate 'i' if figurative, or generate 'l' if literal. Generate a Python list containing the letter.\n<</SYS>>\n The expression is '{idiom}'. The sentence is '{sentence}'[/INST]"
    prompt3 = f"<s>[INST] <<SYS>>\nDoes the expression hold a figurative or literal meaning in the following contextual sentence? Generate 'i' for figurative meaning, or 'l' for literal meaning. Generate a Python list containing the letter.\n<</SYS>>\n The expression is '{idiom}'. The sentence is '{sentence}'[/INST]"
    # prompt2 = f"""
    # # [INST] You are an assistant, who can only generate one letter. Given a contextual sentence and a expression, tell me if the expression is used figurative or literally. Either generate "i" if figurative, or generate "l" if literal.
    # # expression: '{idiom}'
    # # sentence; '{sentence}'
    # # Generate a Python list containing the letter.[/INST]"""

    # prompt3 = f"""
    # [INST] You are a native speaker of English, who can only generate one letter. Does the expression hold a figurative or literal meaning in the following contextual sentence? Generate "i" for figurative meaning, or "l" for literal meaning.
    # expression: '{idiom}'
    # sentence; '{sentence}'
    # Generate a Python list containing the letter.[/INST]"""

    answer1 = get_answer_from_model(prompt1)
    answer2 = get_answer_from_model(prompt2)
    answer3 = get_answer_from_model(prompt3)

    answer1_extracted = extract_answer(idiom, answer1)
    answer2_extracted = extract_answer(idiom, answer2)
    answer3_extracted = extract_answer(idiom, answer3)

    idioms_preds_p1.append((idiom, answer1_extracted))
    idioms_preds_p2.append((idiom, answer2_extracted))
    idioms_preds_p3.append((idiom, answer3_extracted))

    # prediction = int(prediction != 'i')
    # print(f"The model has predicted: \t{prediction}")

    # idiom_pred.append(tuple([idiom, prediction[0]]))
  print(len(idioms_preds_p1))
  print(len(idioms_preds_p2))
  print(len(idioms_preds_p3))

  path = f"predictions/"
  save_predictions(idioms_preds_p1, setting, path, args.model_abr, "p1")
  save_predictions(idioms_preds_p2, setting, path, args.model_abr, "p2")
  save_predictions(idioms_preds_p3, setting, path, args.model_abr, "p3")


  return {"p1": idioms_preds_p1, "p2": idioms_preds_p2, "p3": idioms_preds_p3}

def get_scores(preds_tup, labels):

  preds = [item[1] for item in preds_tup]

  prfs = precision_recall_fscore_support(labels, preds, labels=["i", "l"])

  results = {
    'precision': prfs[0],
    'recall': prfs[1],
    "fscore":prfs[2],
    "support":prfs[3],
    "macro f1": f1_score(labels, preds, average="macro", labels=["i", "l"]),
    "accuracy": accuracy_score(labels, preds)
}
  return results


def write_to_csv(results, filename):
  df = pd.DataFrame([results])

  df.to_csv(filename)


predictions_all_prompts = run_predictions(df.Idiom, df.Sentence, args.model_abr, args.setting)
print(predictions_all_prompts)

for run, predictions in predictions_all_prompts.items():
   
  if args.setting == "literal":
      labels = ["l"] * (len(df))
  else:
    labels = ["i"] * (len(df))

  scores = get_scores(predictions, labels)

  write_to_csv(scores, f"results/{args.setting}_{args.model_abr}_{run}.csv")




# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token="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", device_map="auto")
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, token="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", device_map="auto")



# prompt = "<s>[INST] <<SYS>>\n Is the expression used literally or figuratively? Only answer with lowercase 'i' for figurative or 'l' for literal only. Generate a Python list containing the letter.\n<</SYS>>\n The expression is 'spilled the beans'. The sentence is 'He managed to accidentally spill the beans whilst making dinner.'[/INST]"

# # prompt = "Is the expression 'spill the beans' used literally or figuratively in the sentece 'she managed to spill the beans all over the kitchen floor.' Generate 'i' for figurative, and 'l' for literal. Do not say anything else."

# inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
# # model.to(device)

# outputs = model.generate(**inputs, max_new_tokens=20)
# string = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(string)
# print(type(string))


# match = re.search(r'\[\s*[\'\"](i|l)[\'\"]\s*\]', string)

# # Check if there is a match and print the result
# if match:
#     result = match.group(1)
#     print(f"this is result")
#     print(result)
# else:
#     print("No match found")
