import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re

def check_for_keywords(text):
    if "literal" in text or "literally" in text:
        prediction = "l"
        return prediction 
    elif "figurative" in text or "figuratively" in text:
        prediction = "i"
        return prediction

    else:
        return None


def find_output_label(text):
    pattern = r"output:\s*(i|l)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)  # Return the captured group
    return None

def find_output_label_in_quotation_marks(text):
    # the output is "(.*?)"

    pattern = r'Therefore, the output is "(.*?)"'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)  # Return the captured group
    return None
    
def find_output_is(text):
    pattern = r'the output is \'(i|l)\''
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)  # Return the captured group
    return None

def find_generate_the_letter(text):
    pattern = r"generate the letter \'(i|l)\'"

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)  # Return the captured group
    return None

def extract_label(text):
    # print(f"this is text: {text}")

    after_inst = text.split("[/INST]  ")[-1]
    print(f"after_inst: {after_inst}")


    print(f"output_label: {find_output_label(after_inst)}")


    if find_output_label(after_inst) == None:
        print(f"output_label_in_quotes: {find_output_label_in_quotation_marks(after_inst)}")
        return find_output_label_in_quotation_marks(after_inst)
    
    

    # elif find_output_label_in_quotation_marks(after_inst) == None and find_output_label(after_inst) == None:
    #     print(f"none found!")
    #     # if "literally" in after_inst and "figuratively" in after_inst:
    #     #     print("u")
    #     #     return "u"
    #     # elif "literal" in after_inst and "figurative" in after_inst:
    #     #     print("u")
    #     #     return "u"
    #     # elif "literal" in after_inst or "literally" in after_inst:
    #     #     print("l")
    #     #     return "l"
    #     # elif "figurative" in after_inst or "figuratively" in after_inst:
    #     #     print(f"i")
    #     #     return "i"
    #     # else:
    #     #     print("u")
    #     #     return "u"

    else:
        return find_output_label(after_inst)


predictions_dir = "."

for filename in os.listdir(predictions_dir):
    if "llama27bchathf_" in filename and "cleaned_" not in filename:
        print(filename)

        df = pd.read_csv(filename)
        
        df['cleaned_text'] = df['sentence'].apply(extract_label)
        # df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])

        break
