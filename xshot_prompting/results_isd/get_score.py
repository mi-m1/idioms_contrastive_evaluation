import csv
from os import listdir
# from os.path import isfile, join
import pandas as pd
from statistics import mean 

def get_score(metric, file_path):
    df = pd.read_csv(file_path)
    score = df[metric].values[0]

    return score
        
# onlyfiles = [f for f in listdir(".") if isfile(join(".", f))]
onlyfiles = [f for f in listdir(".") if f != "get_score.py"]

print(onlyfiles)
print(len(onlyfiles))

f1_three_runs_figurative = {}
f1_three_runs_literal = {}


for file in onlyfiles:

    # e.g., figurative_llama38binstruct_p1.csv
    parts = file.split("_")

    setting = parts[0]
    model = parts[1]
    run = parts[2].split(".")[0] # remove .csv bit
    
    macro_f1_score = get_score("macro f1", file)
    print(f"{file}: {macro_f1_score}")

    if setting == "figurative":
        if model not in f1_three_runs_figurative:
            f1_three_runs_figurative[model] = []

        f1_three_runs_figurative[model].append(macro_f1_score)

    elif setting == "literal":

        if model not in f1_three_runs_literal:
            f1_three_runs_literal[model] = []

        f1_three_runs_literal[model].append(macro_f1_score)

print(f1_three_runs_literal)
print(f1_three_runs_figurative)


def get_average(dictionary_of_runs):

    avg = {}

    for model, runs in dictionary_of_runs.items():

        avg[model] = mean(runs)

    sorted_dict = {key: value for key, value in sorted(avg.items())}
    return sorted_dict


avg_lit = get_average(f1_three_runs_literal)
        
avg_fig = get_average(f1_three_runs_figurative)


print("avg_fig:\n", avg_fig)
print("avg_lit:\n",avg_lit)
