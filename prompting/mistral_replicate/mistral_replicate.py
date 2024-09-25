import os
from getpass import getpass
from tqdm import tqdm
import argparse
import replicate
import pandas as pd
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Replicate model ID")
parser.add_argument("--eval_dataset", help="Path to eval dataset file")
parser.add_argument("--model_abr", help="model_abbreviation")
parser.add_argument("--setting", help="figurative or literal")
args = parser.parse_args()


# REPLICATE_API_TOKEN = getpass()
# os.environ["REPLICATE_API_TOKEN"] = "r8_KeuYswdQLhCIjuwFwknqS9SItqZF6A221xzoO"


def prompt_model(prompt):
   for event in replicate.stream(
    args.model_name,
    input={"prompt": prompt},):
        return str(event).replace("\n", "")

dataset_location = args.eval_dataset

df = pd.read_csv(dataset_location)


def save_predictions(idioms_preds, setting, path, model_abr, run):
   filename = f"{path}{setting}_{model_abr}_{run}.csv"
   
   with open(filename,'w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['idiom','sentence'])
    for row in idioms_preds:
        csv_out.writerow(row)



def run_predictions(idioms, sentence, model_abr, setting):
    idioms_preds_p1 = []
    idioms_preds_p2 = []
    idioms_preds_p3 = []
    idiom_sent_dict= list(zip(idioms, sentence))

    for idiom, sentence in tqdm(idiom_sent_dict):
       
        raw_prompt1 = f"Is the expression '{idiom}' used figuratively or literally in the sentence: '{sentence}'. Answer 'i' for figurative, 'l' for literal. Put your answer after 'output: '"
        raw_prompt2 = f"In the sentence '{sentence}', is the expression '{idiom}' being used figuratively or literally? Respond with 'i' for figurative and 'l' for literal. Put your answer after 'output: '"
        raw_prompt3 = f"How is the expression '{idiom}' used in this context: '{sentence}'. Output 'i' if the expression holds figurative meaning, output 'l' if the expression holds literal meaning. Put your answer after 'output: '"
    
        raw_output_p1 = prompt_model(raw_prompt1)
        raw_output_p2 = prompt_model(raw_prompt2)
        raw_output_p3 = prompt_model(raw_prompt3)

        print(f"raw_output_p1: {raw_output_p1}")
        print(f"raw_output_p2: {raw_output_p2}")
        print(f"raw_output_p3: {raw_output_p3}")

        idioms_preds_p1.append((idiom, raw_output_p1))
        idioms_preds_p2.append((idiom, raw_output_p2))
        idioms_preds_p3.append((idiom, raw_output_p3))
    
    path = f""
    save_predictions(idioms_preds_p1, setting, path, model_abr, "p1")
    save_predictions(idioms_preds_p2, setting, path, model_abr, "p2")
    save_predictions(idioms_preds_p3, setting, path, model_abr, "p3")

    return {"p1": idioms_preds_p1, "p2": idioms_preds_p2, "p3": idioms_preds_p3}

predictions_all_prompts = run_predictions(df.Idiom, df.Sentence, args.model_abr, args.setting)
