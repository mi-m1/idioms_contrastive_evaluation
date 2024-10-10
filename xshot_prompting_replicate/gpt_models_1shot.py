
from tqdm import tqdm
import re
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import os
from openai import OpenAI
import argparse
import csv

import os
os.environ['OPENAI_API_KEY'] = "sk-proj-qqjGfSUcVxvSzEOeHBzeT3BlbkFJrwkt5SSwLCZgNvaAcg7G"

from useful_functions import get_random_item_from_list, get_random_oneshot_example

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Huggingface model ID")
# parser.add_argument("--seed", help="Number of runs")
parser.add_argument("--eval_dataset", help="Path to magpie dataset file")
parser.add_argument("--setting", help="figurative or literal")
parser.add_argument("--model_abr", help="Abbreviation of model")
args = parser.parse_args()


class OpenAIPrompter:
    def __init__(self, key, model,):
        """
        Initialize the OpenAIPrompter class.

        Parameters:
        - api_key (str): Your OpenAI API key.
        - model (str): The model to use for the prompts. Default is "gpt-3.5-turbo".
        """
        self.key = key
        self.model = model

    # def prompt(self, prompt_text, max_tokens, temperature, top_p, n):
    def prompt(self, prompt_text):
        """
        Send a prompt to the OpenAI model and return the response.

        Parameters:
        - prompt_text (str): The text prompt to send to the model.
        - max_tokens (int): The maximum number of tokens to generate in the response.
        - temperature (float): Sampling temperature to use. Higher values means the model will take more risks.
        - top_p (float): Nucleus sampling parameter. The model will consider the smallest set of tokens with cumulative probability top_p.
        - n (int): Number of completions to generate for the prompt.

        Returns:
        - response (str): The generated response from the model.
        """

        # client = OpenAI(self.api_key)
        client=OpenAI(api_key = self.key)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                              "type": "text",
                              "text": prompt_text,
                            }
                        ]

                    }
                ]
                # max_tokens=max_tokens,
                # temperature=temperature,
                # top_p=top_p,
                # n=n
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {e}"


fig_data = "/mnt/parscratch/users/acq22zm/ae/prompting/dataset/figurative_1032.csv"
lit_data = "/mnt/parscratch/users/acq22zm/ae/prompting/dataset/literal_1032.csv"

df = pd.read_csv(args.eval_dataset)


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


def run_and_save_predictions(idioms, sentence, model, setting, model_abr):

  # model = "gpt-3.5-turbo"
  api_key = "sk-proj-qqjGfSUcVxvSzEOeHBzeT3BlbkFJrwkt5SSwLCZgNvaAcg7G"
  prompter = OpenAIPrompter(key=api_key, model=model)

  idiom_sent_dict= list(zip(idioms, sentence))

  print(len(idiom_sent_dict))

  idioms_preds_p1 = []
  idioms_preds_p2 = []
  idioms_preds_p3 = []

  for idiom, sentence in tqdm(idiom_sent_dict):

    # random_idiom, oneshot_fig_example, oneshot_lit_example = get_random_oneshot_example(idiom, 2343242, fig_data, lit_data)
    # oneshot_example = f"""\nHere is an example: The expression '{random_idiom}' occurs figuratively in the sentence: '{oneshot_fig_example}', and literally in the setence: '{oneshot_lit_example}'. """

    oneshot_example = f"""Here is an example: The expression 'play with fire' occurs figuratively in the sentence: 'The war took away the unfortunate necessity , as Unionists saw it , to play with fire in the national interest , but it did not materially alter their view of themselves.', so you would output "i",  and literally in the sentence: 'Despite the danger, he decided to play with fire, poking the embers with a stick.', so you would output "l". """
   
    raw_output1 = prompter.prompt(f"Is the expression '{idiom}' used figuratively or literally in the sentence: '{sentence}'. Answer 'i' for figurative, 'l' for literal.{oneshot_example}")
    raw_output2 = prompter.prompt(f"In the sentence '{sentence}', is the expression '{idiom}' being used figuratively or literally? Respond with 'i' for figurative and 'l' for literal.{oneshot_example}")
    raw_output3 = prompter.prompt(f"How is the expression '{idiom}' used in this context: '{sentence}'. Output 'i' if the expression holds figurative meaning, output 'l' if the expression holds literal meaning.{oneshot_example}")

    # response1 = process_output(raw_output1)
    # response2 = process_output(raw_output2)
    # response3 = process_output(raw_output3)


    idioms_preds_p1.append((idiom, raw_output1))
    idioms_preds_p2.append((idiom, raw_output2))
    idioms_preds_p3.append((idiom, raw_output3))

  # print(idom_preds)
  print(len(idioms_preds_p1))
  print(len(idioms_preds_p2))
  print(len(idioms_preds_p3))

  # model_abr = model.replace("-", "")
  # model_abr = model_abr.replace(".", "")

  path = f"raw_outputs_gpts_1shot/"
  save_predictions(idioms_preds_p1, setting, path, model_abr, "p1")
  save_predictions(idioms_preds_p2, setting, path, model_abr, "p2")
  save_predictions(idioms_preds_p3, setting, path, model_abr, "p3")

  return {"p1": idioms_preds_p1, "p2": idioms_preds_p2, "p3": idioms_preds_p3}

# predictions_dict = run_and_save_predictions(df.Idiom, df.Sentence, "gpt-3.5-turbo")

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


# runs = list(range(1, number_of_runs+1))
model = args.model
model_abr = model.replace("-", "")
model_abr = model_abr.replace(".", "")



predictions_all_prompts = run_and_save_predictions(df.Idiom, df.Sentence, model, args.setting, args.model_abr)

# for run, predictions in predictions_all_prompts.items():
   
#   if args.setting == "literal":
#       labels = ["l"] * (len(df))
#   else:
#     labels = ["i"] * (len(df))

#   scores = get_scores(predictions, labels)

#   write_to_csv(scores, f"results_isd/{args.setting}_{model_abr}_{run}.csv")

