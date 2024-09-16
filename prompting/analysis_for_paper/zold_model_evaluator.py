import os
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import statistics as stats

class ModelEvaluator:

    def __init__(self, predictions_dir):
        self.predictions_dir = predictions_dir
        self.f1_results = {}

        self.loose_consistency_results = {}
        self.strict_consistency_results = {}

        models = [
            "gpt35turbo_",
            "gpt4_",
            "gpt4o_",
            # "llama27bchathf_",
            # "llama38binstruct_",
            "flant5small_",
            "flant5large_",
            "flant5xl_",
            "flant5xxl_",
            # "mistral7binstructv0.3_",
        ]

        self.models = models
    def load_predictions(self, filepath):
        return pd.read_csv(filepath)


    def calculate_f1_score(self, true_labels, predictions):
        return f1_score(true_labels, predictions, average="macro")
    
    def get_f1_result(self):

        for filename in os.listdir(self.predictions_dir):
            if filename.endswith(".csv"):
                #extract setting, model name and run ID from the filename

                parts = filename.split("_")
                setting = parts[0] # also the true label
                model_name = parts[1]
                run_id = parts[2].split(".")[0] # get rid of the .csv

                model_name_and_runID = model_name + "_" + run_id

                if model_name_and_runID not in self.f1_results:
                    self.f1_results[model_name_and_runID] = {"figurative": [], "literal": []}

                filepath = os.path.join(self.predictions_dir, filename)
                predictions_df = self.load_predictions(filepath)
                
                #true labels
                if setting == "figurative":
                    true_labels = ["i"] * len(predictions_df)

                elif setting == "literal":
                    true_labels = ["l"] * len(predictions_df)

                # predictions = predictions_df["pred"] # DONE: need to check this in the predictions files
                predictions_df["pred"].apply(str)

                predictions = predictions_df["pred"]
                
                try:
                    f1 = self.calculate_f1_score(true_labels, predictions)

                except TypeError:
                    print(setting, model_name, run_id, )
                else:
                    self.f1_results[model_name_and_runID][setting].append(f1)
        
        return self.f1_results
    
    def average_scores(self, results, setting, decimal_place):

        model_results = {}
        averages = {}

        
        for model in self.models:

            try:
                p1 = results[f"{model}p1"][setting]
                p2 = results[f"{model}p2"][setting]
                p3 = results[f"{model}p3"][setting]

                three_results = [p1,p2,p3] # [[], [], []]
                flattened_three_results = [val for sublist in three_results for val in sublist]
            
                model_results[model] = flattened_three_results
            
            except KeyError:
                print(model)

        print(f"model_results: {model_results}")

        for model, values in model_results.items():
            print(model, values)

            mean = stats.mean(values)
            print(mean)

            stdev = stats.stdev(values)

            rounded_mean = round(mean, decimal_place)
            rounded_stdev = round(stdev, decimal_place)

            rounded_mean_and_stdev = f"{rounded_mean}±{rounded_stdev}"
            averages[model] = rounded_mean_and_stdev

            # mean_and_stdev = f"{round(mean, decimal_place)}±{round(stdev, decimal_place)}"
            # print(mean_and_stdev)
            # averages[model] = mean_and_stdev
        
        print(f"averages: {averages}")

        return model_results, averages
    

    
    def calculate_loose_consistency(self, predictions_file_df, setting):
        
        if setting == "figurative":
            correct_label = "i"

        elif setting == "literal":
            correct_label = "l"

        predictions_file_df['is_correct'] = predictions_file_df['pred'] == correct_label

        idiom_groups = predictions_file_df.groupby('idiom')['is_correct'].all()
        correct_predictions = idiom_groups.sum()

        lc_accuracy = (correct_predictions / len(idiom_groups))*100

        return lc_accuracy
        


    def get_loose_consistency(self):
        for filename in os.listdir(self.predictions_dir):
            if filename.endswith(".csv"):
                #extract setting, model name and run ID from the filename

                parts = filename.split("_")
                setting = parts[0] # also the true label
                model_name = parts[1]
                run_id = parts[2].split(".")[0] # get rid of the .csv

                model_name_and_runID = model_name + "_" + run_id

                if model_name_and_runID not in self.loose_consistency_results:
                    self.loose_consistency_results[model_name_and_runID] = {"figurative": [], "literal": []}

                filepath = os.path.join(self.predictions_dir, filename)
                predictions_df = self.load_predictions(filepath)
                
                try:
                    lc = self.calculate_loose_consistency(predictions_df, setting)

                except TypeError:
                    print(setting, model_name, run_id, )
                else:
                    self.loose_consistency_results[model_name_and_runID][setting].append(lc)
        
        return self.loose_consistency_results
    

    def calculate_strict_consistency(self, figurative_filepath, literal_filepath):
        # Load the data
        df_literal = pd.read_csv(figurative_filepath)
        df_figurative = pd.read_csv(literal_filepath)

        # Define the correct label for each setting
        literal_correct_label = 'l'  # Example correct label for literal setting
        figurative_correct_label = 'i'  # Example correct label for figurative setting

        # Add a column to check if the prediction matches the correct label
        df_literal['is_correct'] = df_literal['pred'] == literal_correct_label
        df_figurative['is_correct'] = df_figurative['pred'] == figurative_correct_label

        # Group by idiom and check if all predictions in each setting are correct
        literal_correct = df_literal.groupby('idiom')['is_correct'].all()
        figurative_correct = df_figurative.groupby('idiom')['is_correct'].all()

        # Combine the results
        combined_correctness = pd.DataFrame({
            'literal_correct': literal_correct,
            'figurative_correct': figurative_correct
        })

        # Check if an idiom is correct in both settings
        combined_correctness['both_correct'] = combined_correctness['literal_correct'] & combined_correctness['figurative_correct']

        # Count the number of idioms where all predictions are correct in both settings
        correct_both_settings = combined_correctness['both_correct'].sum()

        # Display the number of idioms correctly predicted in both settings
        print(f'The model got {correct_both_settings} idioms completely correct in both settings out of {len(combined_correctness)} idioms.')

        return (correct_both_settings/len(combined_correctness))*100

    

    def get_strict_consistency(self,):

        file_pairs = {}
        

        for filename in os.listdir(self.predictions_dir):
            if filename.endswith(".csv"):
                parts = filename.split("_")
                setting = parts[0] # also the true label
                model_name = parts[1]
                run_id = parts[2].split(".")[0] # get rid of the .csv

                model_name_and_runID = model_name + "_" + run_id

                if model_name_and_runID not in file_pairs:
                    file_pairs[model_name_and_runID] = {"figurative": [], "literal": []}

                # print(file_pairs)
                
                file_pairs[model_name_and_runID][setting] =  str(os.path.join(self.predictions_dir, filename))
        
        # print(file_pairs)

        strict_consistency_each_run = {}

        for model_and_id, dct in file_pairs.items():

            # model_name = model_and_id.split("_")[0]

            sc = self.calculate_strict_consistency(dct["figurative"], dct["literal"])

            strict_consistency_each_run[model_and_id] = sc

        # for model_and_id, sc in strict_consistency_each_run.items():
        
        # print(strict_consistency_each_run)
        for model in self.models:
            try:
                p1 = strict_consistency_each_run[f"{model}p1"]
                p2 = strict_consistency_each_run[f"{model}p2"]
                p3 = strict_consistency_each_run[f"{model}p3"]

                three_results = [p1,p2,p3]
                self.strict_consistency_results[model] = three_results

            except KeyError:
                print(model)

        return self.strict_consistency_results

    def sc_average_scores(self, decimal_place):
        averages = {}
        for model, threes in self.strict_consistency_results.items():
            mean = stats.mean(threes)
            std = stats.stdev(threes)

            rounded_mean = round(mean, decimal_place)
            rounded_stdev = round(std, decimal_place)

            rounded_mean_and_stdev = f"{rounded_mean}±{rounded_stdev}"
            averages[model] = rounded_mean_and_stdev

        return averages






evaluator = ModelEvaluator(predictions_dir="../old_predictions_unstandardised")
f1_results = evaluator.get_f1_result()
fig_average_f1_scores, mean_f1_fig= evaluator.average_scores(f1_results, "figurative", 2)
lit_average_f1_scores, mean_f1_lit= evaluator.average_scores(f1_results, "literal", 2)

print(f"f1_results:\t{f1_results}")
# print("fig threes:", mean_f1_fig)
# print("lit three:", mean_f1_lit)

lc_results = evaluator.get_loose_consistency()
# print(f"loose_accuracy results diction:\n\t{lc_results}")

# fig_lc_threes, fig_mean_lc = evaluator.average_scores(lc_results, "figurative", 4)
lit_lc_threes, lit_mean_lc = evaluator.average_scores(lc_results, "literal", 4)

# print(f"fig_lc_threes: {fig_lc_threes}\n\tmean_lc_fig: {fig_mean_lc}")
# print(f"lit_lc_threes: {lit_lc_threes}\n\tmean_lc_lit: {lit_mean_lc}")

sc_results = evaluator.get_strict_consistency()
print(f"\nsc_results:{sc_results}")

sc_averages = evaluator.sc_average_scores(4)
print(sc_averages)

# 0.3264