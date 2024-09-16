import pandas as pd
import csv
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import glob
import statistics 

model_abrs = ["flant5small",
              "flant5xxl",
              "flant5xl",
              "flant5large",
              "gpt4",
              "gpt4o",
              "gpt35turbo",
              "mistral7binstructv0.3",
              "llama38binstruct",
              "llama27bchathf"]

# expressions_division (list) contains 4 lists of expressions

groups_paths = ["group1.csv", "group2.csv", "group3.csv", "group4.csv"]
groups = []

for g in groups_paths:

    with open(g, "r") as f:

        content = f.readlines()
        content = [x.strip() for x in content]
        content = [x.replace(",", "") for x in content]

        groups.append(content)

print(groups)

def get_scores_for_selection(dataframe):

    print(f"this is dataframe: {dataframe}")
    print(f"this is true_labels \t{dataframe.label}")
    true_labels = dataframe.label
    preds = dataframe.sentence

    prfs = precision_recall_fscore_support(true_labels, preds, labels=["i", "l"])

    results = {
    'precision': prfs[0],
    'recall': prfs[1],
    "fscore":prfs[2],
    "support":prfs[3],
    "f1": f1_score(true_labels, preds, labels=["i", "l"], average=None),
    "macro f1": f1_score(true_labels, preds, average="macro", labels=["i", "l"]),
    "accuracy": accuracy_score(true_labels, preds),
    # "per class accuracy": accuracy_score(true_labels, preds, method)
    }

    return pd.DataFrame.from_dict(results)




def write_df_to_csv(df, output_filename):

    df.to_csv(output_filename, index=False)
    

# for filename in glob.glob("../prompting/collect_results/merged_preds/merged_*.csv"):

#     df = pd.read_csv(filename)

#     for g in expressions_division:

#         # df.loc[df['column_name'].isin(g)]

#         df_for_group = df.loc[df['idiom'].isin(g)]

#         results_for_group = get_scores_for_selection(df_for_group)
        
#         output_filename = f"results/{}"

#         # write_df_to_csv(results_for_group, )
def get_results_for_single_file(group, path, output_filename, save):

    df = pd.read_csv(path)

    df['idiom'] = df['idiom'].apply(str)
    df['sentence'] = df['sentence'].apply(str)
    df['label'] = df['label'].apply(str)

    print(f"df:")
    print(df.head)
    

    print(f"this is group: \t{group}")
    # print(df.loc[df['idiom'].isin(group)])
    df_for_group = df[df['idiom'].isin(group)]

    # print(f"df_for_group: {df_for_group}")
    # print(f"type: {type(df_for_group)}")
    # print(f"{df_for_group.columns}")
    # print(df_for_group.head)
    results_for_group = get_scores_for_selection(df_for_group)

    
    # output_filename = f"results/{}"
    if save == True:
        write_df_to_csv(results_for_group, output_filename)
    else:
        pass

    return df_for_group, results_for_group




model_runs = {}
for model in model_abrs:

    list_of_files = []

    for filename in glob.glob("../prompting/collect_results/merged_preds/merged_*.csv"):

        if model in filename:
            list_of_files.append(filename)
    
    model_runs[model] = list_of_files

    # print(model_runs)

    # break

model_paths_organised = {}

for model,paths in model_runs.items():

    # print(f"key:{key}")
    # print(f"value: {value}")
    # 

    paths = sorted(paths)
    print(paths)

    model_paths_organised[model] = paths

def overall_mean_of_all_three(all_three_runs, model, group_num):

    df_concat = pd.concat((all_three_runs[0], all_three_runs[1], all_three_runs[2]))

    average = df_concat.mean()

    average_df = average.to_frame().T

    average_df.columns = df_concat.columns

    average_df.to_csv(f"results_overall_average/{group_num}_{model}.csv", index=False)

    # df_concat = df_concat.groupby(df_concat.index).agg({
    #     '': 'mean',
    #     ""
    # })

def combined_of_all_three_run_and_average_per_class_and_std(all_three_runs, model, group_num):
    df_concat = pd.concat((all_three_runs[0], all_three_runs[1], all_three_runs[2]))
    df_concat.to_csv(f"results_combined_3runs/{group_num}_{model}.csv", index=False)

    class_i = df_concat.iloc[::2].reset_index(drop=True)  # Odd rows ==> idiomatic class
    # print(class_i)
    
    class_l = df_concat.iloc[1::2].reset_index(drop=True)  # Even rows ==> literal class
    # print(class_l)

    class_i.mean().to_frame().T.to_csv(f"results_average_per_class/{group_num}_figurative_{model}.csv", index=False)
    class_l.mean().to_frame().T.to_csv(f"results_average_per_class/{group_num}_literal_{model}.csv", index=False)


    class_i.std().to_frame().T.to_csv(f"results_std_per_class/{group_num}_figurative_{model}.csv", index=False)
    class_l.std().to_frame().T.to_csv(f"results_std_per_class/{group_num}_literal_{model}.csv", index=False)

def sd_of_all_three_runs(all_three_runs, model, group_num, metric, setting):

    df_concat = pd.concat((all_three_runs[0], all_three_runs[1], all_three_runs[2]))

    score = list(df_concat[metric])

    # standard_deviation = 

    # print(score)

    return score




    


    

for model, sorted_paths in model_paths_organised.items():

    all_three_runs_g1 = []
    all_three_runs_g2 = []
    all_three_runs_g3 = []
    all_three_runs_g4 = []

    for path in sorted_paths:

        run = path.split("_")[-1][-6:-4]

        print(run)

        selection_g1, results_g1 = get_results_for_single_file(groups[0], path, f"results_each_run/g1_{model}_{run}.csv", save=True)
        write_df_to_csv(selection_g1, f"results_predictions/g1_{model}_{run}.csv")
        all_three_runs_g1.append(results_g1)

        selection_g2, results_g2 = get_results_for_single_file(groups[1], path, f"results_each_run/g2_{model}_{run}.csv", save=True)
        write_df_to_csv(selection_g2, f"results_predictions/g2_{model}_{run}.csv")
        all_three_runs_g2.append(results_g2)

        selection_g3, results_g3 = get_results_for_single_file(groups[2], path, f"results_each_run/g3_{model}_{run}.csv", save=True)
        write_df_to_csv(selection_g3, f"results_predictions/g3_{model}_{run}.csv")
        all_three_runs_g3.append(results_g3)


        selection_g4, results_g4 = get_results_for_single_file(groups[3], path, f"results_each_run/g4_{model}_{run}.csv", save=True)
        write_df_to_csv(selection_g4, f"results_predictions/g4_{model}_{run}.csv")
        all_three_runs_g4.append(results_g4)
        # write_df_to_csv(selection_g4, f"results_predictions/g4_{model}_{run}.csv")
        # all_three_runs_g4.append(results_g4)


        # results_g2 = get_results_for_single_file(groups[1], path, f"results_each_run/g2_{model}_{run}", save=True)
        # results_g3 = get_results_for_single_file(groups[2], path, f"results_each_run/g3_{model}_{run}", save=True)
        # results_g4 = get_results_for_single_file(groups[3], path, f"results_each_run/g4_{model}_{run}", save=True)
    
    # overall_mean_of_all_three(all_three_runs_g1, model, "g1")
    # overall_mean_of_all_three(all_three_runs_g2, model, "g2")
    # overall_mean_of_all_three(all_three_runs_g3, model, "g3")
    # overall_mean_of_all_three(all_three_runs_g4, model, "g4")

    combined_of_all_three_run_and_average_per_class_and_std(all_three_runs_g1, model, "g1")
    combined_of_all_three_run_and_average_per_class_and_std(all_three_runs_g2, model, "g2")
    combined_of_all_three_run_and_average_per_class_and_std(all_three_runs_g3, model, "g3")
    combined_of_all_three_run_and_average_per_class_and_std(all_three_runs_g4, model, "g4")


    # s = sd_of_all_three_runs(all_three_runs_g1, model, "g1", "f1", "x")
    # print(s)
    # print(model)
    
    





