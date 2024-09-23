from tqdm import tqdm
import re
import pandas as pd
import os

tqdm.pandas()

predictions_dir = "."

def extract_label(text):

    match = re.findall(r"output: (i|l)", text, re.IGNORECASE)

    if match:
        return match[0]
    else:
        return "u"


for filename in os.listdir(predictions_dir):
    
    if "llama38binstruct" in filename and filename.endswith(".csv"):
        print(filename)

        folder_name = "cleaned_llama38binstruct"

        try:
            save_path = os.path.join(predictions_dir, folder_name) 
            os.mkdir(save_path)

        except FileExistsError as e:
            pass

        df = pd.read_csv(filename)

        df['response'] = df['sentence'].progress_apply(extract_label)
        df.to_csv(f'{folder_name}/{filename}', index=False, columns=["idiom","response"])


    if "llama370binstruct" in filename and filename.endswith(".csv"):
        print(filename)

        folder_name = "cleaned_llama370binstruct"

        try:
            save_path = os.path.join(predictions_dir, folder_name) 
            os.mkdir(save_path)

        except FileExistsError as e:
            pass

        df = pd.read_csv(filename)

        df['response'] = df['sentence'].progress_apply(extract_label)
        df.to_csv(f'{folder_name}/{filename}', index=False, columns=["idiom","response"])

    if "llama31405binstruct" in filename and filename.endswith(".csv"):
        print(filename)

        folder_name = "cleaned_llama31405binstruct"

        try:
            save_path = os.path.join(predictions_dir, folder_name) 
            os.mkdir(save_path)

        except FileExistsError as e:
            pass

        df = pd.read_csv(filename)

        df['response'] = df['sentence'].progress_apply(extract_label)
        df.to_csv(f'{folder_name}/{filename}', index=False, columns=["idiom","response"])
    
