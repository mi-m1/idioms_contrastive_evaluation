import pandas as pd
import glob
import statistics 

def get_consistency(predictions_file_all_classes, label_class):

    df = pd.read_csv(predictions_file_all_classes)

    true_label = df[df["label"] == label_class]

    # print(true_label)

    list_of_idioms = list(df.idiom)

    count_instances = {i:list_of_idioms.count(i) for i in list_of_idioms}
    count_instances = {k: v//2 for k,v in count_instances.items()} # becuase of the file contains instaces from both senses

    num_unique_idioms = len(set(list_of_idioms))

    preds_equal_label = true_label.query("sentence == label")

    ls_preds_equal_label = list(preds_equal_label.idiom)

    count_instances_preds_equal_label = {i:ls_preds_equal_label.count(i) for i in ls_preds_equal_label}
    # print(count_instances_preds_equal_label)

    intersection_dict = dict(set(count_instances.items()).intersection(count_instances_preds_equal_label.items()))

    tp_count = len(intersection_dict)

    
    score = tp_count/num_unique_idioms


    return tp_count, score

# # overall true = robustness
# get_tp_count(filename, overall=True)
# # overall false = get per class scores, consistency
# get_tp_count(filename, overall=False,)

def get_scores_for_group(group, model_list):


    for model in model_list:

        all_three_runs_counts_fig = []
        all_three_runs_score_fig = []

        all_three_runs_counts_lit = []
        all_three_runs_score_lit = []
    

        for filename in glob.glob(f"results_predictions/{group}_{model}*.csv"):
            # r_count, robustness = get_consistency(filename, label_class="all")
            c_count_i, class_i_consistency = get_consistency(filename, label_class="i")
            c_count_l, class_l_consistency = get_consistency(filename, label_class="l")

            all_three_runs_counts_lit.append(c_count_l)
            all_three_runs_score_lit.append(class_l_consistency)

            all_three_runs_counts_fig.append(c_count_i)
            all_three_runs_score_fig.append(class_i_consistency)
        
            # print(filename)
            # print(r_count, robustness)
            # print(c_count_i, class_i_consistency)
            # print(c_count_l, class_l_consistency)

        print(model)

        # print(f"{all_three_runs_score_fig},")
        # print(f"{all_three_runs_score_lit},")
        print(f"Mean fig: {statistics.mean(all_three_runs_score_fig)} ± {statistics.stdev(all_three_runs_score_fig)}")
        print(f"Mean lit: {statistics.mean(all_three_runs_score_lit)} ± {statistics.stdev(all_three_runs_score_fig)}")
        # break

model_abrs = ["gpt4o",
            "gpt35turbo",
            "flant5xxl",
            "flant5xl",
            "flant5large",
            "flant5small",              
            "llama38binstruct",
            "llama27bchathf",
            "mistral7binstructv0.3",
            "gpt4_",
]

# get_scores_for_group("g1", model_abrs)
# get_scores_for_group("g2", model_abrs)
# get_scores_for_group("g3", model_abrs)
get_scores_for_group("g4", model_abrs)



