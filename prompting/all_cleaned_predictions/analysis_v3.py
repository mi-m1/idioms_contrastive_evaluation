import os
from itertools import product
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm import tqdm
# from collections import defaultdict
from statistics import fmean,stdev

def save_dict_as_csv(dictionary_to_save, filename):

    # for key, value in dictionary_to_save.items():
    for key, value in dictionary_to_save.items():
        if isinstance(value, list):
            df = pd.DataFrame.from_dict(dictionary_to_save)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame.from_dict([dictionary_to_save])
            df.to_csv(filename, index=False)

class ModelEvaluator:
    def __init__(self, predictions_dir):
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

        settings = ["figurative", "literal"]
        runs = ["p1", "p2", "p3"]

        self.predictions_dir = predictions_dir
        self.models = models
        self.settings = settings
        self.runs = runs

        self.f1_fig = {}
        self.f1_lit = {}
        self.f1_overall = {}

    def fig_lit_pairs_for_each_run(self):

        pairs = {}

        combinations= list(product(self.models, self.runs))
        print(f"number of combinations: {combinations}")
        print(f"combinations: {combinations}")

        for x in combinations:
            print(f"{x[0]}{x[1]}")

            pairs[f"{x[0]}{x[1]}"] = [f"figurative_{x[0]}{x[1]}.csv", f"literal_{x[0]}{x[1]}.csv"]

        print(pairs)

        return pairs

    def get_scores(self, pairs, average_of_three=None):

        scores = {}
        for model_run, file_pair in pairs.items():

            print(model_run, file_pair)

            fig_file = file_pair[0]
            lit_file = file_pair[1]

            fig_preds = pd.read_csv(fig_file)["pred"]
            lit_preds = pd.read_csv(lit_file)["pred"]

            true_labels = ["i"] * 1033 + ["l"] * 1033
            preds = list(fig_preds) + list(lit_preds) 

            cr = classification_report(true_labels, preds, digits=3, output_dict=True,)
            # print(cr)
            
            scores[model_run] = cr
            # break
        
        if average_of_three:

            f1_per_class_average_of_three = {}
            stdev_per_class_average_of_three = {}

            for model in self.models:
                f1_fig = []
                f1_lit = []
                for model_run, results in scores.items():
                    if model in model_run:
                        print(model_run)

                        f1_fig.append(scores[model_run]["i"]["f1-score"])
                        f1_lit.append(scores[model_run]["l"]["f1-score"])

                f1_per_class_average_of_three[model] = [fmean(f1_fig), fmean(f1_lit)]
                stdev_per_class_average_of_three[model] = [stdev(f1_fig), stdev(f1_lit)]

            return scores, f1_per_class_average_of_three,stdev_per_class_average_of_three
        else:
            pass
        return scores
    
    def loose_consistency(self, pairs, average_of_three=None):

        model_lc = {}

        for model_run, file_pair in pairs.items():
            lc_for_each_file = []

            for file in file_pair:
                print(f"file: {file}")

                setting = file.split("_")[0]

                if setting == "figurative":
                    correct_label = "i"
                elif setting == "literal":
                    correct_label = "l"

                predictions_file_df = pd.read_csv(file)

                predictions_file_df['is_correct'] = predictions_file_df['pred'] == correct_label

                idiom_groups = predictions_file_df.groupby('idiom')['is_correct'].all()
                correct_predictions = idiom_groups.sum()

                lc_accuracy = (correct_predictions / len(idiom_groups))*100
                lc_for_each_file.append(lc_accuracy)
            
            model_lc[model_run] = lc_for_each_file

        if average_of_three == True:
            averages = {}
            stdevs = {}
            for model_run, lc_values in model_lc.items():
                model_name = model_run.split('_')[0]  # Get the model name from the key
                
                if model_name not in averages:
                    averages[model_name] = [[], []]  # Initialize sums for both classes
                    counts = 0  # Initialize count
                
                # Accumulate the results for both classes
                averages[model_name][0].append(lc_values[0])  # Sum for class 1
                averages[model_name][1].append(lc_values[1])  # Sum for class 2
                counts += 1  # Increment the count
            print(f"averages in progress: {averages}")
            print(type(averages))

            # Calculate averages
            for model, three_run_results in averages.items():
                averages[model] = [fmean(three_run_results[0]), fmean(three_run_results[1])]
                stdevs[model] = [stdev(three_run_results[0]), stdev(three_run_results[1])]
            print(f"averages: {averages}")

            return model_lc, averages,stdevs

        else:
            return model_lc
        
        
    def strict_consistency(self, pairs, average_of_three=None):

        model_sc = {}

        for model_run, pair in pairs.items():

            # make a big file
            df_fig = pd.read_csv(pair[0])
            df_lit = pd.read_csv(pair[1])

            df = pd.concat([df_fig, df_lit])

            df["correct_label"] = ["i"]*1033 + ["l"]*1033
            # print(df)

            # calculate sc

            # 1) check preds against the correct label
            df['is_correct'] = df['pred'] == df["correct_label"]
            # print(df)

            # 2) count the ones the models got all correct in both settings                
            idiom_groups = df.groupby('idiom')['is_correct'].all()
            # print(idiom_groups)
            correct_predictions = idiom_groups.sum()
            # print(correct_predictions)

            sc_accuracy = (correct_predictions / len(idiom_groups))
            # print(sc_accuracy)

            model_sc[model_run] = sc_accuracy

        if average_of_three == True:
            
            sc_average_of_three = {}
            sc_stdev_of_three = {}

            for model in self.models:
                sc_value_of_three_runs = []
                for model_run, results in model_sc.items():
                    if model in model_run:
                        sc_value_of_three_runs.append(model_sc[model_run])

                sc_average_of_three[model] = fmean(sc_value_of_three_runs)*100
                sc_stdev_of_three[model] = stdev(sc_value_of_three_runs)*100

            return model_sc, sc_average_of_three,sc_stdev_of_three
        else:
            return model_sc

    
        pass
            # print(f"model_run:{model_run}, file_pair:{file_pair}")
            # df_fig = pd.read_csv(file_pair[0])
            # df_lit = pd.read_csv(file_pair[1])
            # df_merged = df_fig.append(df_lit, ignore_index=True)

            # print(df_merged.shape)


            




evaluator = ModelEvaluator(predictions_dir=".")
# print(evaluator.models)

# print(evaluator.fig_lit_pairs_for_each_run())

filepath_pairs = evaluator.fig_lit_pairs_for_each_run()
print(f"filepath_pairs: {filepath_pairs}")

scores,avg_of_three,stdev_of_three = evaluator.get_scores(filepath_pairs,average_of_three=True)

print(f"\n{scores}\n")

print(avg_of_three)

print(f"Model Name\tFig (F1)\tLit (F1)")
for model, per_class_f1 in avg_of_three.items():
    print(f"{model}:\t{per_class_f1[0]*100}\t{per_class_f1[1]*100}")
print("\n")

print(f"Model Name\tFig (F1 stdev)\tLit (F1 stdev)")
for model,per_class_stdev in stdev_of_three.items():
    print(f"{model}:\t{per_class_stdev[0]*100}\t{per_class_stdev[1]*100}")

# sanity check
# print(f"{scores['gpt4o_p1']['i']['f1-score']}")
# print(f"{scores['gpt4o_p2']['i']['f1-score']}")
# print(f"{scores['gpt4o_p3']['i']['f1-score']}")

# print(f"{scores['gpt4o_p1']['l']['f1-score']}")
# print(f"{scores['gpt4o_p2']['l']['f1-score']}")
# print(f"{scores['gpt4o_p3']['l']['f1-score']}")

print(f"\n")
print(f"loose consistency")
lc_scores, avg_lc, stdev_lc = evaluator.loose_consistency(filepath_pairs, average_of_three=True)
print(lc_scores)
print(avg_lc)

print(f"lc_values for each run")
for model, lc_values in lc_scores.items():
    print(f"{model}:\t{lc_values[0]}\t{lc_values[1]}")

print(f"\nlc_values for average")
for model, lc_values in avg_lc.items():
    print(f"{model}:\t{lc_values[0]}\t{lc_values[1]}")

print(f"\nlc_values for stdev")
for model, lc_values in stdev_lc.items():
    print(f"{model}:\t{lc_values[0]}\t{lc_values[1]}")

save_dict_as_csv(lc_scores, "results/lc_scores.csv")
save_dict_as_csv(avg_lc, "results/lc_avg.csv")
save_dict_as_csv(stdev_lc, "results/lc_stdev.csv")
# save_dict_as_csv()

##### STRICT CONSISTENCY ######ß
print(evaluator.strict_consistency(filepath_pairs))

sc_scores, avg_sc, stdev_sc = evaluator.strict_consistency(filepath_pairs, average_of_three=True)
print(f"\nsc_values for average")
for model, sc_values in avg_sc.items():
    print(f"{model}:\t{sc_values}")

print(f"\nsc_values for stdev")
for model, sc_values in stdev_sc.items():
    print(f"{model}:\t{sc_values}")


save_dict_as_csv(sc_scores, "results/sc_scores.csv")
save_dict_as_csv(avg_sc, "results/sc_avg.csv")
save_dict_as_csv(stdev_sc, "results/sc_stdev.csv")


