from tqdm import tqdm
import pandas as pd
import re
import os

tqdm.pandas()

predictions_dir = "."

def extract_label_gpt4(text):

    if text == "'i'":
        return "i"
    elif text == "'l'":
        return "l"
    
    elif not re.match(r'^(i|l)$', text, re.IGNORECASE) and "'i'" in text:
        return "i"
    elif not re.match(r'^(i|l)$', text, re.IGNORECASE):
        return "u"
    else:
        return text
    
    
def extract_label_gpt35turbo(text):
    # figurative (i)
    # 'l'
    # figuratively
    # f
    text = text.lower()

    if "figurative" in text or "figuratively" in text or "(i)" in text or "Figurative" in text or "Figuratively" in text:
        return "i"
    elif "literal" in text or "literally" in text:
        return "l"
    
    elif text == "'i'" or text == "f" or text ==" i":
        return "i"
    elif text == "'l'":
        return "l"
    elif text == "lf":
        return "u"
    
    elif not re.match(r'^(i|l)$', text, re.IGNORECASE):
        return "u"
    else:
        return text
    

    



for filename in os.listdir(predictions_dir):

    if "gpt4_" in filename and "cleaned_" not in filename:
        print(filename)


        df = pd.read_csv(filename)

        df['cleaned_text'] = df['Sentence'].progress_apply(extract_label_gpt4)
        # print(df)

        df.to_csv(f'cleaned_{filename}', index=False, columns=["Idiom","cleaned_text",])
    
    if "gpt35turbo_" in filename and "cleaned_" not in filename:
        print(filename)

        df = pd.read_csv(filename)

        df['cleaned_text'] = df['Sentence'].progress_apply(extract_label_gpt35turbo)

        df.to_csv(f"cleaned_{filename}", index=False, columns=["Idiom", "cleaned_text"])
    
    if "gpt4o_" in filename and "cleaned_" not in filename:
        print(filename)

        df = pd.read_csv(filename)

        df['cleaned_text'] = df['Sentence'].progress_apply(extract_label_gpt4)

        df.to_csv(f"cleaned_{filename}", index=False, columns=["Idiom", "cleaned_text"])

    

        