import pandas as pd
from tqdm import tqdm
from statistics import mean, stdev
from os import listdir


# models 
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


# list of results files
preds_files = [f for f in listdir("../predictions/") if f != "get_score.py"]
# print(preds_files)

# function: given model, pair fig_p1 with lit_p1
def get_file_pair(model, list_of_files):

    # all six files for model
    results_for_model = [f for f in list_of_files if model in f] 

    p1 = sorted([f for f in results_for_model if "p1" in f])
    p2 = sorted([f for f in results_for_model if "p2" in f])
    p3 = sorted([f for f in results_for_model if "p3" in f])

    pairs = [p1, p2, p3]

    return pairs
print(get_file_pair("gpt35turbo", preds_files))

def get_perfect_idioms(file_path,):

    ## read predictions file
    df = pd.read_csv("../predictions/"+file_path)

    ## use file name to check if the setting i.e., the true label for file
    setting = file_path.split("_")[0]

    ## check how many instances of each idiom
    ## store in dictionary: idiom_count e.g., {"eager beaver": 2, 'panda car': 1}
    instances = list(df.idiom)
    idiom_count = {i:instances.count(i) for i in instances}

    perfect_idioms = []
    if setting == "figurative":
        for idiom, count in idiom_count.items():
            sub_df = df[(df.sentence == "i") & (df.idiom == idiom)]

            if sub_df.shape[0] == count:
                perfect_idioms.append(idiom)

    elif setting == "literal":
        for idiom, count in idiom_count.items():
            sub_df = df[(df.sentence == "l") & (df.idiom == idiom)]

            if sub_df.shape[0] == count:
                perfect_idioms.append(idiom)

    return perfect_idioms, len(perfect_idioms)

print(get_perfect_idioms('figurative_gpt35turbo_p1.csv'))

# [['figurative_gpt35turbo_p1.csv', 'literal_gpt35turbo_p1.csv'], ['figurative_gpt35turbo_p2.csv', 'literal_gpt35turbo_p2.csv'], ['figurative_gpt35turbo_p3.csv', 'literal_gpt35turbo_p3.csv']]



def write_consistency_stats_for_run(directory, model, run, num_pif, num_pil, tp_shared, fig_consistency, lit_consistency, strict_consistency_score_for_run,):
    
    df = pd.DataFrame({'no. perfect idioms (fig)': [num_pif],
                    'no. perfect idioms (lit)': [num_pil],
                  'intersection': [tp_shared],
                  "figurative consistency":[fig_consistency],
                  "literal consistency":[lit_consistency],
                  "strict consistency score": [strict_consistency_score_for_run]})
    
    file_output_name = f"{directory}{model}_{run}.csv"
    df.to_csv(file_output_name, index=False)  


def write_strict_consistency_stats_for_model(directory, model, fig_mean, fig_stdev, lit_mean, lit_stdev, strict_mean_score, strict_stdev_score):
    df = pd.DataFrame({
        "figurative (mean)": [fig_mean],
        "figurative (stdev)": [fig_stdev],
        "literal (mean)": [lit_mean],
        "literal (stdev)": [lit_stdev],
        "strict consistency (mean)": [strict_mean_score],
        "strict consistency (stdev)": [strict_stdev_score],
    })

    file_output_name = f"{directory}{model}.csv"

    df.to_csv(file_output_name, index=False)

def write_consistency_stats():
    pass



# store the averages for each model, across three runs
averages_strict_consistency = {}

# # loop through models to get pairs of runs
for model in model_abrs:

    pairs_list = get_file_pair(model, preds_files)
    # print(pairs_list)

    # each pair is also a run

    strict_consistency_all_runs = []
    fig_consistency_all_runs = []
    lit_consistency_all_runs = []


    # for pair in run [fig_model_p1, lit_model_p1] in pairs_list, get perfect idioms for each file
    for pair in pairs_list:

        # get run

        run0 = pair[0].split("_")[2].split(".")[0]
        run1 = pair[1].split("_")[2].split(".")[0]

        assert run0 == run1, "run not the same!"

        # perfect idioms figurative
        perfect_idioms_fig, num_pif= get_perfect_idioms(pair[0])

        # perfect idioms literal
        perfect_idioms_lit, num_pil = get_perfect_idioms(pair[1])

        # print(len(perfect_idioms_fig), len(perfect_idioms_lit))
        # print(num_pif, num_pil)


        ### strict consistency measure

        # intersection of idioms that model achieved perfection for figurative and literal settings
        shared_perfect_idioms = set(perfect_idioms_lit).intersection(perfect_idioms_fig)

        tp_shared = len(shared_perfect_idioms)
        # print(tp_shared)

        # strict consistency score for run
        strict_consistency_score_for_run = (tp_shared / 402) * 100
        # print(strict_consistency_score_for_run)


        # add strict consistency of each run to list
        strict_consistency_all_runs.append(strict_consistency_score_for_run)


    

        ### consistency measure

        fig_consistency = (num_pif / 402) * 100
        lit_consistency = (num_pil / 402) * 100

        # add strict consistency of each run to list
        fig_consistency_all_runs.append(fig_consistency)
        lit_consistency_all_runs.append(lit_consistency)

        # save both consistency scores for run
        write_consistency_stats_for_run("../results_consistency_each_run/", model, run0, num_pif, num_pil, tp_shared, fig_consistency, lit_consistency, strict_consistency_score_for_run)
        
        # print(f"strict consistency: {strict_consistency_score_for_run}")
        # print(f"fig consistency: {fig_consistency} \t lit consistency: {lit_consistency}")




    # strict consistency averaged and stdev
    mean_strict_consistency = mean(strict_consistency_all_runs)
    stdev_strict_consistency = stdev(strict_consistency_all_runs)

    mean_fig_consistency = mean(fig_consistency_all_runs)
    stdev_fig_consistency = stdev(fig_consistency_all_runs)

    mean_lit_consistency = mean(lit_consistency_all_runs)
    stdev_lit_consistency = stdev(lit_consistency_all_runs)

    # write both consistency scores averaged and stdev
    write_strict_consistency_stats_for_model("../results_consistency_averaged/", model, mean_fig_consistency, stdev_fig_consistency, mean_lit_consistency, stdev_lit_consistency, mean_strict_consistency, stdev_strict_consistency)
    
    print(f"{model}\t{mean_fig_consistency}\t{stdev_fig_consistency}\t{mean_lit_consistency}\t{stdev_lit_consistency}\t{mean_strict_consistency}\t{stdev_strict_consistency}")

    with open("results_for_gsheet.csv", "a") as f:
        f.write(f"{model}\t{mean_fig_consistency}\t{stdev_fig_consistency}\t{mean_lit_consistency}\t{stdev_lit_consistency}\t{mean_strict_consistency}\t{stdev_strict_consistency}\n")
    






