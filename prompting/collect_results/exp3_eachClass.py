import pandas as pd
import glob

storage_counts = {} # "flant5large": [[p1],[p2],[p3]]
storage_scores = {}

paths_lit_fig = [[],[]] # = [[lit_paths], [fig_paths]]

for filename in glob.glob("collected_preds/*.csv"):
    
    if "figurative_" in filename:

        paths_lit_fig[0].append(filename)

    else:

        paths_lit_fig[1].append(filename)

paths_lit_fig = [sorted(x) for x in paths_lit_fig]

# print(paths_lit_fig[0])


model_abrs = ["flant5small",
              "flant5xxl",
              "flant5xl",
              "flant5large",
              "gpt4_",
              "gpt4o",
              "gpt35turbo",
              "mistral7binstructv0.3",
              "llama38binstruct",
              "llama27bchathf"]


def return_triples(list_of_merged):

    triples_list = []
    for model in model_abrs:

        ls = []
        for path in list_of_merged:

            if model in path:
                ls.append(path)

        triples_list.append(ls)

    return triples_list

print(f"\t{paths_lit_fig[0]}\n\t{paths_lit_fig[1]}")

for i,x in enumerate(paths_lit_fig):
    
    # if "figurative" in x[0]:
    #     print("figurative!")
    # else:
    #     print("literal")

    for model_set_paths in return_triples(x):

        model_name = model_set_paths[0].split("_")[-2]

        ls_counts = []
        ls_understandingScores = []

        for index,path in enumerate(model_set_paths):
            print(path)

            df_1 = pd.read_csv(model_set_paths[index])

            list_of_idioms = list(df_1.idiom)

            count_instances = {i:list_of_idioms.count(i) for i in list_of_idioms}

            num_unique_idioms = len(set(list_of_idioms))

            preds_equal_label = df_1.query("sentence == label")

            ls_preds_equal_label = list(preds_equal_label.idiom)

            count_instances_preds_equal_label = {i:ls_preds_equal_label.count(i) for i in ls_preds_equal_label}


            intersection_dict = dict(set(count_instances.items()).intersection(count_instances_preds_equal_label.items()))
            # diff_dict = dict(count_instances_preds_equal_label.items() - count_instances.items())

            # tp_count = 

            tp_count = len(intersection_dict)

            ls_counts.append(tp_count)

                ##### measures

            understanding_score = tp_count / num_unique_idioms
            ls_understandingScores.append(understanding_score)


        storage_counts[model_name] = ls_counts
        storage_scores[model_name] = ls_understandingScores

    print(storage_counts)
    print(storage_scores)





