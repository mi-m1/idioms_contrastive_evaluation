import pandas as pd
from tqdm import tqdm
import re
import os
from sklearn.metrics import f1_score

predictions_dir = "../../xshot_prompting_samePrompt/cleaned_model_outputs"

def calculate_f1_score(file_path, true_label, average='macro'):
    """
    Reads a CSV file, extracts true and predicted labels, and calculates the F1 score.

    Parameters:
    - file_path (str): Path to the CSV file.
    - true_label_col (str): Name of the column containing the true labels.
    - predicted_label_col (str): Name of the column containing the predicted labels.
    - average (str): Averaging method for F1 score ('micro', 'macro', 'weighted', etc.).

    Returns:
    - float: The F1 score.
    """
    # Step 1: Load the CSV file using pandas
    df = pd.read_csv(file_path)
    
    # Step 2: Extract the true and predicted labels
    true_labels = [true_label] * 1033
    predicted_labels = df["pred"]
    
    # Step 3: Calculate the F1 score
    f1 = f1_score(true_labels, predicted_labels, average=average)
    
    return f1

# # Example usage
# file_path = 'your_file.csv'  # Replace with actual file path
# true_label_col = 'true_label'  # Replace with actual true label column name
# predicted_label_col = 'predicted_label'  # Replace with actual predicted label column name

# f1 = calculate_f1_score(file_path, true_label_col, predicted_label_col)
# print(f'F1 Score: {f1}')


for filename in os.listdir(predictions_dir):
    # if filename.endswith(".csv") and "mistral" not in filename:
    if filename.endswith(".csv") and "mistral" in filename:
        # print(f"filename: {filename}")

        parts = filename.split("_")
        setting = parts[0] # also the true label
        print(f"setting: {setting}")

        if setting == "literal":
            true_label = "l"
        else:
            true_label == "i"

        path_to_file = predictions_dir + "/" + filename
        print(calculate_f1_score(path_to_file, true_label,))







