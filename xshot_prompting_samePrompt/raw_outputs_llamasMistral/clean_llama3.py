import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re

predictions_dir = "."

def check_for_keywords(text):
    if "literal" in text or "literally" in text:
        prediction = "l"
        return prediction 
    elif "figurative" in text or "figuratively" in text or "idiomatic" in text:
        prediction = "i"
        return prediction
    else:
        return None
    
def extract_label_p1(text):

    lines = text.split("\n")
    # print(f"lines: {lines}")
    last_line = lines[-1].strip()
    # print(f"last_line: {last_line}")

    if last_line == 'i':
        return last_line
    elif last_line == 'l':
        return last_line
    else:
        print(f"None")
        return "u"
    

def extract_label_p2(text):

    lines = text.split("\n")
    # print(f"lines: {lines[-1]}")
    
    last_line = lines[-1].strip()
    
    pattern = r"is used (literally|figuratively)"
    match = re.search(pattern, last_line, re.IGNORECASE)
    if match:
        if match.group(1) == "literally":
            return "l"
        elif match.group(1) == "figuratively":
            return "i"
    else:
        print(f"None")

    pattern_single_letter = r"(i|l)"
    match = re.search(pattern_single_letter, last_line, re.IGNORECASE)
    if match:
        if match.group(1) == "l":
            return "l"
        elif match.group(1) == "i":
            return "i"
    else:
        print(f"None")

    pattern_answer_colon = r"answer: (i|l)"
    match = re.search(pattern_answer_colon, last_line, re.IGNORECASE)
    if match:
        if match.group(1) == "l":
            return "l"
        elif match.group(1) == "i":
            return "i"
    else:
        print(f"None")    
       

    if (last_line == "i") or (last_line == "I"):
        return "i"
    elif (last_line == "l") or (last_line == "L"):
        return "l"
    else:
        print("None")


    if check_for_keywords(last_line) is not None:
        return check_for_keywords(last_line)

    return "u"

def extract_label_p3(text):

    lines = text.split("\n")
    print(f"lines: {lines[-1]}")
    last_line = lines[-1].strip()

    pattern = r"holds a (literal|figurative) meaning"
    match = re.search(pattern, last_line, re.IGNORECASE)
    if match:
        if match.group(1) == "literal":
            return "l"
        elif match.group(1) == "figurative":
            return "i"
    else:
        print(f"None")

    pattern_answer_colon = r"answer: (i|l)"
    match = re.search(pattern_answer_colon, last_line, re.IGNORECASE)
    if match:
        if match.group(1) == "l":
            return "l"
        elif match.group(1) == "i":
            return "i"
    else:
        print(f"None")    


    if ": L" in last_line or ": l" in last_line or " 'l'" in last_line or " ''l''" in last_line or ' "L"' in last_line:
        return "l"
    elif ": I" in last_line or ": i" in last_line or " 'i'" in last_line or ' "i"' in last_line:
        return "i"
    else:
        print(f"None")

    
    # check second to last line
    second_last_line = lines[-3].strip()
    print(f"2nd_last_line: {second_last_line}")

    # if ": L" in second_last_line or ": l" in second_last_line or " 'l'" in second_last_line or " ''l''" in second_last_line or ' "L"' in second_last_line:
    #     return "l"
    # elif ": I" in second_last_line or ": i" in second_last_line or " 'i'" in second_last_line or ' "i"' in second_last_line:
    #     return "i"
    # else:
    #     print(f"None")

    if check_for_keywords(last_line):
        return check_for_keywords(last_line)
    else:
        if check_for_keywords(second_last_line) is not None:
            return check_for_keywords(second_last_line)
    
    return "u"


    # lines: I would answer with the letter 'i'.





for filename in os.listdir(predictions_dir):
    if "llama38binstruct_p1" in filename and "cleaned_" not in filename:
        print(filename)

        df = pd.read_csv(filename)
        
        df['cleaned_text'] = df['sentence'].apply(extract_label_p1)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])
    elif "llama38binstruct_p2" in filename and "cleaned_" not in filename:
        print(filename)

        df = pd.read_csv(filename)
        
        df['cleaned_text'] = df['sentence'].apply(extract_label_p2)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])        

    elif "llama38binstruct_p3" in filename and "cleaned_" not in filename:

        df = pd.read_csv(filename)
        df['cleaned_text'] = df['sentence'].apply(extract_label_p3)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])   
        


print(extract_label_p2(
    """system

You are an assistant.user

expression: 'carry the can'
 sentence: 'After the party, I had to carry the can of recycling out to the curb.' QUESTION: Given a contextual sentence and a expression, tell me if the expression is used figurative or literally. Either generate the letter 'i' if figurative, or generate the letter 'l' if literal.
Here is an example: The expression 'play with fire' occurs figuratively in the sentence: 'The war took away the unfortunate necessity, as Unionists saw it, to play with fire in the national interest, but it did not materially alter their view of themselves.', and literally in the setence: 'Despite the danger, he decided to play with fire, poking the embers with a stick.'.assistant

I'd be happy to help!

For the given sentence: ""After the party, I had to carry the can of recycling out to the curb.""

The expression ""carry the can"" is used literally in this sentence. Therefore, the answer is:

L
"""
))