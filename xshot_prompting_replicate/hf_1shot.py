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
import csv
from useful_functions import get_random_item_from_list, get_random_oneshot_example

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Huggingface model ID")
# parser.add_argument("--seed", help="Number of runs")
parser.add_argument("--eval_dataset", help="Path to eval dataset file")
parser.add_argument("--model_abr", help="model_abbreviation")
parser.add_argument("--setting", help="figurative or literal")
args = parser.parse_args()


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, device_map="auto", resume_download=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, resume_download=True)



dataset_location = args.eval_dataset
df = pd.read_csv(dataset_location)
print(df.shape)
# number_of_runs = 3

fig_data = "/mnt/parscratch/users/acq22zm/ae/prompting/dataset/figurative_1032.csv"
lit_data = "/mnt/parscratch/users/acq22zm/ae/prompting/dataset/literal_1032.csv"


def get_answer_from_model(prompt):
  inputs = tokenizer(prompt, return_tensors="pt")
  inputs = inputs.to('cuda')
  outputs = model.generate(**inputs)
  prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  return prediction[0]

def save_predictions(idioms_preds, setting, path, model_abr, run):
   filename = f"{path}{setting}_{model_abr}_{run}.csv"
   
   with open(filename,'w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['idiom','sentence'])
    for row in idioms_preds:
        csv_out.writerow(row)

def process_output(output):
    # output = output.lower().replace("'", "")

    # regex to check if predictions file has the right format, i.e., letter i or l after comma
    # ^(?!.*(,i|,l)).*$

    output = output.lower()
    if 'figurative' in output or 'figuratively' in output or ": i" in output:
        output = "i"
    elif 'literal' in output or 'literally' in output or ": l" in output:
        output = "l"
    elif output == "f":
        output = "i"
     
    output = output.replace("'", "")
    output = output.replace("`", "")
    output = output.replace("‘", "")
    output = output.replace("’", "")

    return output


def run_predictions(idioms, sentence, model_abr, setting):
# def run_and_save_predictions(idioms, sentence, model, setting):
  idioms_preds_p1 = []
  idioms_preds_p2 = []
  idioms_preds_p3 = []

  issues = []
  idiom_sent_tpl= list(zip(idioms, sentence))

  for idiom, sentence in tqdm(idiom_sent_tpl):

    # random_idiom, oneshot_fig_example, oneshot_lit_example = get_random_oneshot_example(idiom, 2343242, fig_data, lit_data)

    my_instruction = "Is the meaning of expression idiomatic or literal? If used idiomatically, answer 'i', if literally, answer 'l'."
    expression = f"Expression: {idiom}"
    sent = f"Sentence: {sentence}"

    # oneshot_example = f"""\nHere is an example: The expression '{random_idiom}' occurs figuratively in the sentence: '{oneshot_fig_example}', and literally in the setence: '{oneshot_lit_example}'. """
    oneshot_example = f"""\nHere is an example: The expression 'play with fire' occurs figuratively in the sentence: 'The war took away the unfortunate necessity , as Unionists saw it , to play with fire in the national interest , but it did not materially alter their view of themselves.', and literally in the setence: 'Despite the danger, he decided to play with fire, poking the embers with a stick.'. """

    prompt1 = my_instruction + expression + sent + oneshot_example
    prompt2 = f"In the sentence '{sentence}', is the expression '{idiom}' being used figuratively or literally? Respond with 'i' for figurative and 'l' for literal.{oneshot_example}"
    prompt3 = f"How is the expression '{idiom}' used in this context: '{sentence}'. Output 'i' if the expression holds figurative meaning, output 'l' if the expression holds literal meaning.{oneshot_example}"

    answer1 = get_answer_from_model(prompt1)
    answer2 = get_answer_from_model(prompt2)
    answer3 = get_answer_from_model(prompt3)

    # answer1 = 

    idioms_preds_p1.append((idiom, answer1))
    idioms_preds_p2.append((idiom, answer2))
    idioms_preds_p3.append((idiom, answer3))

    # prediction = int(prediction != 'i')
    # print(f"The model has predicted: \t{prediction}")

    # idiom_pred.append(tuple([idiom, prediction[0]]))
  print(len(idioms_preds_p1))
  print(len(idioms_preds_p2))
  print(len(idioms_preds_p3))

  path = f"raw_outputs_flant5/"
  save_predictions(idioms_preds_p1, setting, path, args.model_abr, "p1")
  save_predictions(idioms_preds_p2, setting, path, args.model_abr, "p2")
  save_predictions(idioms_preds_p3, setting, path, args.model_abr, "p3")


  return {"p1": idioms_preds_p1, "p2": idioms_preds_p2, "p3": idioms_preds_p3}



def get_scores(preds_tup, labels):

  preds = [item[1] for item in preds_tup]

  prfs = precision_recall_fscore_support(labels, preds, labels=["i", "l",])

  results = {
    'precision': prfs[0],
    'recall': prfs[1],
    "fscore":prfs[2],
    "support":prfs[3],
    "macro f1": f1_score(labels, preds, average="macro", labels=["i", "l",]),
    "accuracy": accuracy_score(labels, preds)
}
  return results

def write_to_csv(results, filename):
  df = pd.DataFrame([results])

  df.to_csv(filename)


number_of_runs = 3
runs = list(range(1, number_of_runs+1))
setting = args.setting

all_results = []

# run = args.seed
# for run in runs:

predictions_all_prompts = run_predictions(df.Idiom, df.Sentence, args.model_abr, args.setting)
print(predictions_all_prompts)

# for run, predictions in predictions_all_prompts.items():

#   outputs_modified = [(idiom, 'o' if category not in ['i', 'l'] else category) for idiom, category in predictions]

#   if setting=="literal":
#     true_labels = ["l"] * len(outputs_modified)
#   else:
#     true_labels = ["i"] * len(outputs_modified)

#   scores = get_scores(outputs_modified, true_labels)
#   all_results.append(scores)

#   write_to_csv(scores, f"results_isd/{setting}_{args.model_abr}_{run}.csv")



# # outputs, issues = run_predictions(df.Idiom, df.Sentence, model, "literal")
# # outputs = [item[1] for item in outputs if ]
# print(f"Length of outputs_modified: {len(outputs_modified)}")
# save_predictions(outputs_modified, "literal", "predictions/", args.model_abr, run)

# print(issues)













