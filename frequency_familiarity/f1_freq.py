import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt




# predictions_dir = "../prompting/all_cleaned_predictions"
predictions_dir = "../all_cleaned_predictions"

models = [
    "gpt4o_",
    "gpt35turbo_",
    "flant5xxl_",
    "flant5xl_",
    "flant5large_",
    "flant5small_",
    "llama31405binstruct_",
    "llama370binstruct_",
    "llama38binstruct_",
    "llama270bchat_",
    "llama213bchat_",
    "llama27bchat_",
    # "mistral7binstructv0.3_",
    "gpt4_",
]

#{"gpt40_":[[f_p1,l_p1],[f_p2,l_p2],[f_p3,l_p3]]}
file_map = {}
for model in models:
    file_map[model] = [
        [f"figurative_{model}p1.csv", f"literal_{model}p1.csv"],
        [f"figurative_{model}p2.csv", f"literal_{model}p2.csv"],
        [f"figurative_{model}p3.csv", f"literal_{model}p3.csv"],
    ]

print(file_map)


def calculate_f1_scores(df,pred_col_name,correctLabel_col_name,kind_of_average):
    # Initialize an empty dictionary to store F1 scores
    f1_scores = {}

    # Group by idiom
    grouped = df.groupby('idiom')

    # Iterate over each idiom group
    for idiom, group in grouped:
        # Extract the predicted labels and correct labels
        preds = group[pred_col_name]
        correct_labels = group[correctLabel_col_name]
        
        # Calculate F1 score for the current idiom
        # You can specify the average='binary' if the labels are binary (0,1),
        # or use 'micro', 'macro', 'weighted' if they are multiclass.
        # f1 = f1_score(correct_labels, preds, average='binary')
        f1 = f1_score(correct_labels, preds, average=kind_of_average)

        # Store the F1 score for the idiom
        f1_scores[idiom] = f1

    # f1_scores = {k: [v] for k, v in f1_scores.items()}
    return f1_scores

def draw_plot(model_name, f1_scores_df, freq_df,x_axis_var, y_axis_var):

    merged_df = pd.merge(f1_scores_df,freq_df, on="idiom")
    print(merged_df)

    sns.set_theme()
    sns.scatterplot(data=merged_df, x=x_axis_var, y=y_axis_var)
    plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # sns_plot = sns.pairplot(df, hue='species', height=2.5)
    plt.savefig(f'plots_macro/{model_name}.png')
    plt.clf()

# frequency_values
freq_df = pd.read_csv("frequency_revised.csv")
num_of_hits = freq_df["number of hits"]

# 6 files as one big file
for model, pair_for_each_run in file_map.items():

    six_files_paths = [item for row in pair_for_each_run for item in row]
    print(f"six_files_paths: {six_files_paths}")
    
    big_df = []

    for file_path in six_files_paths:
        setting = file_path.split("_")[0]
        df = pd.read_csv(predictions_dir+"/"+file_path)

        if setting == "figurative":
            df["correct_label"] = ["i"] * 1033
        elif setting == "literal":
            df["correct_label"] = ["l"] * 1033

        big_df.append(df)

    # print(big_df)

    dtfr = pd.concat(big_df, axis=0, ignore_index=True)
    dtfr["number of hits"] = num_of_hits
    print(dtfr)

    f1_scores = calculate_f1_scores(dtfr, "pred", "correct_label", "macro")
    f1_scores_df = pd.DataFrame(list(f1_scores.items()), columns=['idiom', 'f1'],)
    print(f1_scores_df)

    # draw_plot
    draw_plot(model, f1_scores_df, freq_df, "number of hits", "f1")
    # draw_plot(f1_scores_df, freq_df, "number of hits per million tokens", "f1")
    


    # f1_scores_df = pd.DataFrame.from_dict(f1_scores, orient="index", columns=["f1 (micro)"])
    # print(len(f1_scores))
    # draw_plot(dtfr, "number of hits", "f1")



    # break

    


# runs = ["p1", "p2", "p3"]

# pairs = {}

# combinations= list(product(models, runs))
# print(f"number of combinations: {combinations}")
# print(f"combinations: {combinations}")

# for x in combinations:
#     print(f"{x[0]}{x[1]}")

#     pairs[f"{x[0]}{x[1]}"] = [f"figurative_{x[0]}{x[1]}.csv", f"literal_{x[0]}{x[1]}.csv"]

# print(pairs)


# for model_run, pair in pairs.items():

#     # make a big file
#     df_fig = pd.read_csv(pair[0])
#     df_lit = pd.read_csv(pair[1])
    
#     df = pd.concat([df_fig, df_lit])
#     df["correct_label"] = ["i"]*1033 + ["l"]*1033





# for filename in os.listdir(predictions_dir):
#     if filename.endswith(".csv"):
#         print(filename)