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

def find_output_would_be(text):
    pattern = r'Therefore, the output would be ""(i|l)"".'

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)  # Return the captured group
    return None

def find_generate_double_quotes_letter_as_output(text):
    pattern = r'Therefore, I will generate ""(i|l)"" as the output.'

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)  # Return the captured group
    return None


def find_generate_quotes_letter_as_output(text):
    pattern = r'Therefore, I will generate "(i|l)" as the output'

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)  # Return the captured group
    return None

def extract_label(text):
    # pass

    after_inst = text.split("[/INST]  ")[-1]
    print(f"after_inst: {after_inst}")

    result = find_output_label(after_inst)
    if result is not None:
        print("A")
        return result

    result = find_output_label_in_quotation_marks(after_inst)
    if result is not None:
        print("B")
        return result

    result = find_output_is(after_inst)
    if result is not None:
        print("C")
        return result

    result = find_generate_the_letter(after_inst)
    if result is not None:
        print("D")
        return result
    
    result = find_output_would_be(after_inst)
    if result is not None:
        print("E")
        return result
    
    result = find_generate_double_quotes_letter_as_output(after_inst)
    if result is not None:
        print("F")
        return result
    
    result = find_generate_quotes_letter_as_output(after_inst)
    if result is not None:
        print("F")
        return result
    
    result = check_for_keywords(after_inst)
    if result is not None:
        print("F")
        return result
    
    return "u"



predictions_dir = "."

for filename in os.listdir(predictions_dir):
    if "llama27bchathf" in filename and "cleaned_" not in filename:
        print(filename)

        df = pd.read_csv(filename)
        
        df['cleaned_text'] = df['sentence'].apply(extract_label)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])

        # break


text = """[INST] <<SYS>>
You are an assistant.
<</SYS>>

expression: 'play with fire'
 sentence: 'The stunts became more daring and , in one episode , Crawford found himself literally playing with fire , when what looked like fireworks were attached to the back of his jacket and suddenly had to start exploding .' QUESTION: Given the sentence and the expression, tell me if the expression is used figurative or literally. Either generate 'output: i' if figurative, or generate the letter 'output: l' if literal. Only generate the letter after 'output: '. 
Here is an example: The expression 'nest egg' occurs figuratively in the sentence: 'The survey looking into the business impacts of the pandemic also found seven in 10 businesses impacted by holiday spending reported lower revenues compared to previous holiday seasons -- depriving the businesses of the nest egg typically used to navigate the tougher winter months.', and literally in the setence: 'I climbed the tree to get a closer look, and to my surprise, I found a nest egg tucked away in the branches.'. [/INST]  Understood! For the sentence you provided, the expression ""play with fire"" is used figuratively. Therefore, the output is:

output: i"""

print(f"extracted label:{extract_label(text)}")