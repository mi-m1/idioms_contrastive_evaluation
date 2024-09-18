import pandas as pd
import re
import os



# Function to clean text and extract label
def extract_label(text):
    # # Step 1: Remove text between [INST] and [/INST]
    # cleaned_text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL).strip()
    # # return cleaned_text
    # print(cleaned_text)

    match = re.findall(r'[O|o]utput:\s([a-zA-Z])"', text, flags=re.IGNORECASE)
    print(f"match: {match}")   

    if match:
        return match[0]
    else:
        return "u"

    # return match 


    # TODO: findall output:\s*[a-zA-Z]\n
    # if len(match)>1:
        
    #     label_part = match[0].split("\n\n")[-1]
    #     label = label_part.split(":")[-1].strip()

    #     if "i" in label or "figurative" in label or "figuratively" in label:
    #         return "i"
    #     elif "l" in label or "literal" in label or "literally" in label:
    #         return "l"
    #     else:
    #         return "u"
        
    # elif len(match) == 1:
        
    #     label_part = match[0]
    #     label = label_part.split(":")[-1].strip()
    #     if "i" in label or "figurative" in label or "figuratively" in label:
    #         return "i"
    #     elif "l" in label or "literal" in label or "literally" in label:
    #         return "l"
    #     else:
    #         return "u"
        
    # elif re.search(r'\bfigurative\b|\bfiguratively\b', cleaned_text, flags=re.IGNORECASE):
    #     return 'i'  # Return 'i' if figurative language is detected
    # elif re.search(r'\bliteral\b|\bliterally\b', cleaned_text, flags=re.IGNORECASE):
    #     return 'l'  # Return 'l' if literal language is detected
    # elif re.search(r"\bidiom\b", cleaned_text, flags=re.IGNORECASE):
    #     return "u"
    # else:
    #     return "u"


df = pd.read_csv("/mnt/parscratch/users/acq22zm/ae/prompting/raw_output/figurative_llama38binstruct_p3.csv")

df['cleaned_text'] = df['sentence'].apply(extract_label)

print(df["cleaned_text"])
df.to_csv(f'cleaned_output.csv', index=False, columns=["idiom","cleaned_text"])



text = """system

You are a language expert.user

expression: 'all hell broke loose'
 sentence: 'Then all hell broke loose.' QUESTION: Is the expression figurative or literal? Generate 'output: i' if the idiom is used figuratively, or generate 'output: l' if the expression is used literally. Only generate the letter after 'output: '.assistant

output: i"
"""

x = extract_label(text)
print(f"x: {x}")


print(f"hi")