import pandas as pd
import re
import os
from tqdm import tqdm

tqdm.pandas()

def extract_response(text):
    response = text.split(r"[/INST]")[-1]
    return response

def check_if_all_elem_identical(lst):
    return all(i == lst[0] for i in lst)

def extract_label_p1(text):
    response = extract_response(text)
    # print(f"response: {response}")

    match = re.findall(r'[O|o]utput: (i|l)', response,)

    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0]
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0]

    match_double_quotes = re.findall(r'\"(i|l)\"', response, re.IGNORECASE)
    print(f"match_double_quotes: {match_double_quotes}")

    if match_double_quotes:
        return match_double_quotes[0]
    else:
        # return "u"
        # print(f"response: {response}")
        pass
    # check the last line of the response

    last_line = response.split("\n")[-1]

    count_figurative = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("figurative"), last_line))
    count_figuratively = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("figuratively"), last_line))
    count_literal = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("literal"), last_line))
    count_literally = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("literally"), last_line))

    print(f"f_counts: {count_figurative + count_figuratively}")
    print(f"l_counts: {count_literal + count_literally}")

    if (count_figurative + count_figuratively) > (count_literal + count_literally):
        print(f"{last_line}")
        return "i"
    elif (count_figurative + count_figuratively) < (count_literal + count_literally):
        print(f"f{last_line}\n")
        return "l"
    else:
        print("check")
        return "u"

def extract_label_p2(text):
    response = extract_response(text)
    # return response

    match = re.findall(r'[O|o]utput: (i|l)', response,)

    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0]
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0]
    
    count_figurative = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("figurative"), response))
    count_figuratively = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("figuratively"), response))
    count_literal = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("literal"), response))
    count_literally = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("literally"), response))

    print(f"f_counts: {count_figurative + count_figuratively}")
    print(f"l_counts: {count_literal + count_literally}")

    if (count_figurative + count_figuratively) > (count_literal + count_literally):
        print(f"{response}")
        return "i"
    elif (count_figurative + count_figuratively) < (count_literal + count_literally):
        print(f"f{response}\n")
        return "l"
    else:
        print("check")
        return "check"

def extract_label_p3(text):
    pass


predictions_dir = "."

for filename in os.listdir(predictions_dir):
    if filename.endswith(".csv") and "cleaned_" not in filename and "llama27bchathf_p1" in filename and "inter_" not in filename:
    # if filename.endswith(".csv") and "testing" in filename and "inter_" not in filename:
        print(filename)
        df = pd.read_csv(filename)
        
        df['response'] = df['sentence'].progress_apply(extract_label_p1)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","response"])

    if filename.endswith(".csv") and "cleaned_" not in filename and "llama27bchathf_p2" in filename and "inter_" not in filename:
        print(filename)
        df = pd.read_csv(filename)
        
        df['response'] = df['sentence'].progress_apply(extract_label_p2)
        df.to_csv(f'inter_{filename}', index=False, columns=["idiom","response"])

    else:
        pass
    
    # if filename.endswith(".csv") and "cleaned_" not in filename and "llama27bchathf_p2" in filename and "inter_" not in filename:
    #     print(filename)
    #     df = pd.read_csv(filename)
        
    #     df['response'] = df['sentence'].progress_apply(extract_label_p3)
    #     df.to_csv(f'inter_{filename}', index=False, columns=["idiom","response"])
    


print(extract_label_p1("""Sure! The expression ""in the pink"" is a common idiom that can have both figurative and literal meanings, depending on the context in which it is used.
"  Sure, I'd be happy to help!

The expression ""head for the hills"" is a figurative idiom that means to escape or flee quickly, often in a state of panic or danger. Therefore, I would output ""i"" for the idiom being used figuratively.
"""))

print(extract_label_p1("I apologise I can't produce the right answer."))

print(extract_label_p1('""off the hills"" and some more text here ""i""'))

print(extract_label_p1('''Great, I'm happy to help!

The expression ""go for the jugular"" is figurative, so I will generate ""l"" as the output.'''))

print(extract_label_p1("Output: i During her output: l ashd output: i"))