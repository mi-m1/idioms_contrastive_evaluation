import pandas as pd
from tqdm import tqdm
import os
import re

tqdm.pandas()

def extract_label(text):
    
    match = re.findall(r"output: (i|l)", text, re.IGNORECASE)

    if match:
        return match[0]
    else:
        return "check"




predictions_dir = "."
for filename in os.listdir(predictions_dir):

    if "llama270bchat" in filename and filename.endswith(".csv"):
        # print(filename)

        try:
            save_path = os.path.join(predictions_dir, "cleaned_llama270bchat") 
            os.mkdir(save_path)
        except FileExistsError as e:
            pass

        df = pd.read_csv(filename)

        df['response'] = df['sentence'].progress_apply(extract_label)
        df.to_csv(f'cleaned_llama270bchat/{filename}', index=False, columns=["idiom","response"])

