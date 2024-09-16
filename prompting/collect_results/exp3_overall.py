import pandas as pd
import glob


storage_counts = {} # "flant5large": [[p1],[p2],[p3]]
storage_scores = {}
storage_idioms = {}

list_of_merged = []

for filename in glob.glob("merged_preds/*.csv"):

    list_of_merged.append(filename)

list_of_merged = sorted(list_of_merged)


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

triples_list = []
for model in model_abrs:

    ls = []
    for path in list_of_merged:

        if model in path:
            ls.append(path)

    triples_list.append(ls)

print(triples_list)


for model_set_paths in triples_list:

    model_name = model_set_paths[0].split("_")[-2]
    # setting

    ls_counts = []
    ls_understandingScores = []
    ls_idioms = []

    for index, path in enumerate(model_set_paths):

        print(path)
        df_1 = pd.read_csv(model_set_paths[index])
        # df_2 = pd.read_csv(model_set_paths)
        # df_3 = pd.read_csv(model_set_paths)

        # print(df_1.query("sentence == label"))

        list_of_idioms = list(df_1.idiom)

        count_instances = {i:list_of_idioms.count(i) for i in list_of_idioms}
        # print(count_instances)
        # print(f"count_instances: {count_instances}")
        # print(len(count_instances))

        num_unique_idioms = len(set(list_of_idioms))


        

        ### sentence === label

        preds_equal_label = df_1.query("sentence == label")
        # print(preds_equal_label.shape)

        ls_preds_equal_label = list(preds_equal_label.idiom)
        # print(len(set(ls_preds_equal_label)))

        count_instances_preds_equal_label = {i:ls_preds_equal_label.count(i) for i in ls_preds_equal_label}
        # print(f"\t{count_instances_preds_equal_label}")
        # print(f"len: {len(count_instances_preds_equal_label)}")

        # print(len(count_instances_preds_equal_label.keys()))

        intersection_dict = dict(set(count_instances.items()).intersection(count_instances_preds_equal_label.items()))
        # diff_dict = dict(count_instances_preds_equal_label.items() - count_instances.items())

        # tp_count = 

        tp_count = len(intersection_dict)
        # print(tp_count)

        ls_counts.append(tp_count)

        ls_idioms.append(intersection_dict.keys())

        ##### measures

        understanding_score = tp_count / num_unique_idioms
        ls_understandingScores.append(understanding_score)


    storage_counts[model_name] = ls_counts
    storage_scores[model_name] = ls_understandingScores
    storage_idioms[model_name]= ls_idioms

    # break

# print(storage_counts)
# print(storage_scores)
# print(storage_idioms)

def find_common(list1, list2, list3):
    # Convert lists to sets
    set1 = set(list1)
    set2 = set(list2)
    set3 = set(list3)
     
    # Use list comprehension to find common elements
    # in all three sets and return as a list
    return [elem for elem in set1 if elem in set2 and elem in set3]

top3_models = ["flant5xxl", "gpt35turbo", "mistral7binstructv0.3"]

df_f = pd.read_csv("frequency_results_396.csv")
# print(df_frequencies["melting pot"]["number of hits"])
# filtered_df = df_f[df_f['expression'] == "melting pot"]
# print(filtered_df)
# number_of_hits = filtered_df['number of hits']
# print(print(type(number_of_hits)))
# print(number_of_hits.astype(int))

# print(df_f.loc[df_f['expression'] == 'melting pot','number of hits'].values[0])

# x = filtered_df.get("number of hits")
# print(x)
# print(int(x))

idiom_freq = []
for model in top3_models:

    idiom_group = {}

    idioms_three = storage_idioms[model]

    print(len(idioms_three))

    overlapped_idioms = list(set(idioms_three[0]) & set(idioms_three[1]) & set(idioms_three[2]))
    # overlapped_idioms = min(idioms_three[0], idioms_three[1], idioms_three[2])
    # overlapped_idioms = min(idioms_three)

    print(len(overlapped_idioms))

    print(overlapped_idioms)


    for idiom in overlapped_idioms:

        # filtered_df = df_f[df_f['expression'] == idiom]
        print(idiom)
        number_of_hits = df_f.loc[df_f['expression'] == idiom,'number of hits'].values
        print(number_of_hits)

        try:
            number_of_hits = number_of_hits[0]
        except:
            pass

     
        # print(df_f.loc[df_f['expression'] == 'melting pot','number of hits'].values[0])
        

        # number_of_hits = int(number_of_hits)

        if number_of_hits <= 11:
            group = 1
        elif number_of_hits < 137:
            group = 2
        elif number_of_hits < 1578:
            group = 3
        else:
            group = 4
        
        idiom_group[idiom] = group

    idiom_freq.append(idiom_group)

print(idiom_freq)
print(len(idiom_freq))











# for model_set_paths in triples_list: #['merged_preds/merged_gpt35turbo_p1.csv', 'merged_preds/merged_gpt35turbo_p2.csv', 'merged_preds/merged_gpt35turbo_p3.csv']

#     model_name = model_set_paths[0].split("_")[-2]

#     ls_counts = []
#     ls_understandingScores = []

#     for path in model_set_paths:

#         df_1 = pd.read_csv(model_set_paths[0])
#         # df_2 = pd.read_csv(model_set_paths)
#         # df_3 = pd.read_csv(model_set_paths)

#         # print(df_1.query("sentence == label"))

#         list_of_idioms = list(df_1.idiom)

#         count_instances = {i:list_of_idioms.count(i) for i in list_of_idioms}
#         # print(count_instances)
#         # print(f"count_instances: {count_instances}")
#         # print(len(count_instances))

#         num_unique_idioms = len(set(list_of_idioms))

#         ### sentence === label

#         preds_equal_label = df_1.query("sentence == label")
#         # print(preds_equal_label.shape)

#         ls_preds_equal_label = list(preds_equal_label.idiom)
#         # print(len(ls_preds_equal_label))

#         count_instances_preds_equal_label = {i:ls_preds_equal_label.count(i) for i in ls_preds_equal_label}
#         # print(f"\t{count_instances_preds_equal_label}")
#         # print(f"len: {len(count_instances_preds_equal_label)}")



#         intersection_dict = dict(set(count_instances.items()).intersection(count_instances_preds_equal_label.items()))
#         # diff_dict = dict(count_instances_preds_equal_label.items() - count_instances.items())

#         # tp_count = 

#         tp_count = len(intersection_dict)

#         ls_counts.append(tp_count)

#         ##### measures

#         understanding_score = tp_count / num_unique_idioms
#         ls_understandingScores.append(understanding_score)


#     storage_counts[model_name] = ls_counts
#     storage_scores[model_name] = ls_understandingScores

#     # break

# print(storage_counts)
# print(storage_scores)





