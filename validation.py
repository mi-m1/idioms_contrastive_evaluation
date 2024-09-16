from tqdm import tqdm
import re
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import jsonlines
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse


model_id = "google/flan-t5-xxl"

i_data_file = "validation_dataset/validation20_i.csv"
l_data_file = "validation_dataset/validation20_l.csv"

df_i = pd.read_csv(i_data_file)
df_l = pd.read_csv(l_data_file)


def convert_to_idiom_sentence_pairs(df):
  pairs = []

  for index, row in df.iterrows():
    idiom = row['Idiom']

    sentences = [row["S1"], row["S2"], row["S3"]]

    for sentence in sentences:

      pairs.append((idiom, sentence.strip()))

  return pairs

pairs_i = convert_to_idiom_sentence_pairs(df_i)
pairs_l = convert_to_idiom_sentence_pairs(df_l)

print(pairs_l)
print(len(pairs_l))


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# model = AutoModelForSeq2SeqLM.from_pretrained(args.model, device_map="auto", resume_download=True)
# tokenizer = AutoTokenizer.from_pretrained(args.model, resume_download=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto", resume_download=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, resume_download=True)

def prompt_model(pairs, label_for_pairs):
  idioms, preds, labels, sentences = [], [], [], []

  for idiom, sentence in tqdm(pairs):

    my_instruction = "Is the meaning of expression idiomatic or literal? If idiomatic, answer 'i', if literal, answer 'l'."
    expression = f" Expression: {idiom}."
    sentence = f" Sentence: {sentence}"

    prompt = my_instruction + expression + sentence

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to("cuda")
    outputs = model.generate(**inputs)
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    idioms.append(idiom)
    preds.append(prediction[0])
    labels.append(label_for_pairs)
    sentences.append(sentence)

  return idioms, preds, labels, sentences, label_for_pairs

def save_information_to_csv(idioms, preds, labels, sentences, label):
  
    df = pd.DataFrame(
        {
        "idioms": idioms,
        "preds": preds,
        "labels": labels,
        "sentences": sentences,
        }
    )

    try:
        os.mkdir(f"results/val_{label}_{model}_outputs.csv")
    except FileExistsError as e:
        pass

    df.to_csv(f"results/val_{label}_{model}_outputs.csv")


    return labels, preds, label

def save_metrics_to_csv(labels, preds, label):
    prfs = precision_recall_fscore_support(labels, preds, average=None, labels=["i", "l"])
    results = {
        'precision': prfs[0],
        'recall': prfs[1],
        "fscore":prfs[2],
        "support":prfs[3],
        "macro f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds)
    }

    print(results)

    pd.DataFrame(results).to_csv(f"results/val_{label}_{model}_scores.csv")



save_metrics_to_csv(save_information_to_csv(prompt_model(pairs_i, "i")))