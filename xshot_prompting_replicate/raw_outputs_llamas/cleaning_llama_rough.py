import pandas as pd
from tqdm import tqdm
import re

tqdm.pandas()

def extract_label_llama27b(text):
    print(text)

    match = re.findall('output "(i|l)"', text, re.IGNORECASE)

    if len(match) == 1:
        print(f"match: {match}")
        return match[0]
    
    match = re.findall(r"Output: (i|l|I|L)", text,)

    text = text.strip()
    if match:
        print(f"match2: {match}")
        return match[0]
    

    match = re.findall(r'"(i|l)"', text)
    if match:
        print(f"match3: {match}")
        return match



df = pd.read_csv("figurative_llama27bchat_p1.csv")

df['response'] = df["sentence"].progress_apply(extract_label_llama27b)