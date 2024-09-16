import pandas as pd
import csv
from os import listdir

from statistics import mean
from tqdm import tqdm


preds_files = [f for f in listdir("../predictions/") if f != "get_score.py"]
print(preds_files)

tp = 0

model_abrs = ["gpt4o",
            "gpt35turbo",
            "flant5xxl",
            "flant5xl",
            "flant5large",
            "flant5small",              
            "llama38binstruct",
            "llama27bchathf",
            "mistral7binstructv0.3",
            "gpt4",
]
model_files = {}

for file_path in preds_files:
    parts = file_path.split('_')

    setting = parts[0]
    model = parts[1]  # e.g., 'flant5xxl'
    run = parts[2].split(".")[0] # remove .csv bit

    if model not in model_files:
        model_files[model] = []

        
        
sorted_model_files = {key: value for key, value in sorted(model_files.items())}
print(f"This is sorted_model_files:\n{sorted_model_files}\n")




for file_path in preds_files:

    

    print(file_path)
    setting = file_path.split("_")[0]

    df = pd.read_csv("../predictions/"+file_path)

    instances = list(df.idiom)
    idiom_count = {i:instances.count(i) for i in instances}
    print(idiom_count)
    
    # if setting == "figurative":

    perfect_idioms_literal = []
    perfect_idioms_figurative = []

    if setting == "literal":
        for idiom, count in idiom_count.items():
            sub_df = df[(df.sentence == "l") & (df.idiom == idiom)]
            # print(sub_df.shape)
            # print(sub_df.shape[0])

            if sub_df.shape[0] == count:
                perfect_idioms_literal.append(idiom)

    elif setting == "figurative":
        for idiom, count in idiom_count.items():
            sub_df = df[(df.sentence == "i") & (df.idiom == idiom)]

            if sub_df.shape[0] == count:
                perfect_idioms_figurative.append(idiom)

    # print(perfect_idioms_literal)
    # print(perfect_idioms_figurative)




    # else:
    #     for idiom in set(instances):
    #         df[(df.sentence == "i") & (df.idiom == idiom)]
    
    
    break