import os
from tqdm import tqdm
import re
import pandas as pd

tqdm.pandas()

def extract_label(text):
    
    match = re.findall(r"output: (i|l)", text, re.IGNORECASE)
    if match:
        return match[0]
    else:
        # return "check"
        pass

    # output: (i)
    match = re.findall(r"output: \((i|l)\)", text, re.IGNORECASE)
    if match:
        return match[0]
    else:
        # return "check"
        pass
    
    match = re.findall(r"'(i|l)'", text, re.IGNORECASE)
    if match:
        return match[0]
    else:
        # return "check"
        pass
    
    # ""(i|l)""
    match = re.findall(r'"(i|l)"', text, re.IGNORECASE)
    if match:
        return match[0]
    else:
        # return "check"
        pass
    
    match = re.findall(r"output: (f)", text, re.IGNORECASE)
    if match:
        return "i"
    else:
        return "u"
    

# TODO: finish this code for llama213b
predictions_dir = "."
for filename in os.listdir(predictions_dir):

    if "llama213bchat" in filename and filename.endswith(".csv"):
        # print(filename)

        try:
            save_path = os.path.join(predictions_dir, "cleaned_llama213bchat") 
            os.mkdir(save_path)
        except FileExistsError as e:
            pass

        df = pd.read_csv(filename)

        df['response'] = df['sentence'].progress_apply(extract_label)
        df.to_csv(f'cleaned_llama213bchat/{filename}', index=False, columns=["idiom","response"])

