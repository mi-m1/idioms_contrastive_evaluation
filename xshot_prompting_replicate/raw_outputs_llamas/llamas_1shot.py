import os
from getpass import getpass
from tqdm import tqdm
import argparse
import replicate
import pandas as pd
import csv
from useful_functions import get_random_item_from_list, get_random_oneshot_example
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Huggingface model ID")
parser.add_argument("--eval_dataset", help="Path to eval dataset file")
parser.add_argument("--model_abr", help="model_abbreviation")
parser.add_argument("--setting", help="figurative or literal")
args = parser.parse_args()



fig_data = "/mnt/parscratch/users/acq22zm/ae/prompting/dataset/figurative_1032.csv"
lit_data = "/mnt/parscratch/users/acq22zm/ae/prompting/dataset/literal_1032.csv"


dataset_location = args.eval_dataset
df = pd.read_csv(dataset_location)
print(df.shape)


def get_answer_from_model(prompt):

    output = replicate.run(
      args.model_name,
      input={"prompt": prompt}
    )
    
    return ''.join(output)



def save_predictions(idioms_preds, setting, path, model_abr, run):
   filename = f"{path}{setting}_{model_abr}_{run}.csv"
   
   with open(filename,'w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['idiom','sentence'])
    for row in idioms_preds:
        csv_out.writerow(row)



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

def run_predictions(idioms, sentence, model_abr, setting):
# def run_and_save_predictions(idioms, sentence, model, setting):
    idioms_preds_p1 = []
    idioms_preds_p2 = []
    idioms_preds_p3 = []
    idiom_sent_dict= list(zip(idioms, sentence))

    for idiom, sentence in tqdm(idiom_sent_dict):

        random_idiom, oneshot_fig_example, oneshot_lit_example = get_random_oneshot_example(idiom, 2343242, fig_data, lit_data)
        # oneshot_example = f"""\nHere is an example: The expression '{random_idiom}' occurs figuratively in the sentence: '{oneshot_fig_example}', and literally in the setence: '{oneshot_lit_example}'. """
        oneshot_example = f"""Here is an example: The expression 'play with fire' occurs figuratively in the sentence: 'The war took away the unfortunate necessity , as Unionists saw it , to play with fire in the national interest , but it did not materially alter their view of themselves.', so you would output "i",  and literally in the sentence: 'Despite the danger, he decided to play with fire, poking the embers with a stick.', so you would output "l". """

        my_instruction = "Is the meaning of expression idiomatic or literal? If used idiomatically, answer 'i', if literally, answer 'l'. Only generate the letter after 'output: '"
        expression = f"Expression: {idiom}"
        sent = f"Sentence: {sentence}"

        prompt1 = my_instruction + expression + sent + oneshot_example
        prompt2 = f"In the sentence '{sentence}', is the expression '{idiom}' being used figuratively or literally? Respond with 'i' for figurative and 'l' for literal.{oneshot_example} Only generate the letter after 'output: '"
        prompt3 = f"How is the expression '{idiom}' used in this context: '{sentence}'. Output 'i' if the expression holds figurative meaning, output 'l' if the expression holds literal meaning.{oneshot_example} Only generate the letter after 'output: '"
    
        answer1 = get_answer_from_model(prompt1)
        answer2 = get_answer_from_model(prompt2)
        answer3 = get_answer_from_model(prompt3)

      
        idioms_preds_p1.append((idiom, answer1))
        idioms_preds_p2.append((idiom, answer2))
        idioms_preds_p3.append((idiom, answer3))


    path = f""
    save_predictions(idioms_preds_p1, setting, path, model_abr, "p1")
    save_predictions(idioms_preds_p2, setting, path, model_abr, "p2")
    save_predictions(idioms_preds_p3, setting, path, model_abr, "p3")


    # return {"p1": idioms_preds_p1}
    return {"p1": idioms_preds_p1, "p2": idioms_preds_p2, "p3": idioms_preds_p3}

all_results = []
predictions_all_prompts = run_predictions(df.Idiom, df.Sentence, args.model_abr, args.setting)

# for run, predictions in predictions_all_prompts.items():

#   preds_cleaned = [(idiom, 'o' if category not in ['i', 'l'] else category) for idiom, category in predictions]

#   if args.setting=="literal":
#     true_labels = ["l"] * len(preds_cleaned)
#   else:
#     true_labels = ["i"] * len(preds_cleaned)

#   scores = get_scores(preds_cleaned, true_labels)
#   all_results.append(scores)

#   write_to_csv(scores, f"results_isd/{args.setting}_{args.model_abr}_{run}.csv")



