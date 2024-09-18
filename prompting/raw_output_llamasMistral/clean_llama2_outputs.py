import pandas as pd
import re
import os



# Function to clean text and extract label
def extract_label(text):
    # Step 1: Remove text between [INST] and [/INST]
    cleaned_text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL).strip()
    # return cleaned_text
    print(cleaned_text)
    match = re.findall(r'output:\s*[a-zA-Z].*', cleaned_text, flags=re.IGNORECASE)
    print(f"match: {match}")    

    if len(match)>1:
        
        label_part = match[0].split("\n\n")[-1]
        label = label_part.split(":")[-1].strip()

        if "i" in label or "figurative" in label or "figuratively" in label:
            return "i"
        elif "l" in label or "literal" in label or "literally" in label:
            return "l"
        else:
            return "u"
        
    elif len(match) == 1:
        
        label_part = match[0]
        label = label_part.split(":")[-1].strip()
        if "i" in label or "figurative" in label or "figuratively" in label:
            return "i"
        elif "l" in label or "literal" in label or "literally" in label:
            return "l"
        else:
            return "u"
        
    elif re.search(r'\bfigurative\b|\bfiguratively\b', cleaned_text, flags=re.IGNORECASE):
        return 'i'  # Return 'i' if figurative language is detected
    elif re.search(r'\bliteral\b|\bliterally\b', cleaned_text, flags=re.IGNORECASE):
        return 'l'  # Return 'l' if literal language is detected
    elif re.search(r"\bidiom\b", cleaned_text, flags=re.IGNORECASE):
        return "u"
    else:
        return "u"

predictions_dir = "."

for filename in os.listdir(predictions_dir):
    if filename.endswith(".csv"):

        df = pd.read_csv(filename)

        
        df['cleaned_text'] = df['sentence'].apply(extract_label)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])







# df = pd.read_csv("/mnt/parscratch/users/acq22zm/ae/prompting/raw_output/figurative_llama27bchathf_p1.csv")
# text = """[INST] <<SYS>>
# You are a language expert.
# <</SYS>>

# expression: 'join the club'
#  sentence: 'If you 're confused , join the club !' QUESTION: Is the expression figurative or literal? Generate 'output: i' if the idiom is used figuratively, or generate 'output: l' if the expression is used literally. Only generate the letter after 'output: '. [/INST]  Sure! The expression ""join the club"" is generally used figuratively, to convey the idea of joining a group of people who share a common experience or problem. So, in this case, I would output:

# Output: z"
# """

    

# # Apply the clean_text function to each row in the 'text' column
# df['cleaned_text'] = df['sentence'].apply(extract_label)

# print(df["cleaned_text"])

# df.to_csv('output.csv', index=False, columns=["idiom","cleaned_text"])


# text1 = """[INST] <<SYS>>
# You are a language expert.
# <</SYS>>

# expression: 'join the club'
#  sentence: 'If you 're confused , join the club !' QUESTION: Is the expression figurative or literal? Generate 'output: i' if the idiom is used figuratively, or generate 'output: l' if the expression is used literally. Only generate the letter after 'output: '. [/INST]  Sure! The expression ""join the club"" is generally used figuratively, to convey the idea of joining a group of people who share a common experience or problem. So, in this case, I would output:

# Output: z"
# """

# x = extract_label(text1)
# print(f"label extracted: {x}")

# text2 = """[INST] <<SYS>>
# You are a language expert.
# <</SYS>>

# expression: 'join the club'
#  sentence: 'If you 're confused , join the club !' QUESTION: Is the expression figurative or literal? Generate 'output: i' if the idiom is used figuratively, or generate 'output: l' if the expression is used literally. Only generate the letter after 'output: '. [/INST]  Sure! The expression ""join the club"" is generally used figuratively, to convey the idea of joining a group of people who share a common experience or problem. So, in this case, I would:

# Output: z"
# """

# x = extract_label(text2)
# print(f"label extracted: {x}")

# # x = "Zutput: i"
# # def extract_label(cleaned_text):
# #     # Step 2: Find the label after "output: "
# #     match = re.search(r'Zutput:\s([a-zA-Z])', cleaned_text, flags=re.IGNORECASE)
# #     print(f"match: {match}")
    
# #     if match:
# #         # Return the label if found
# #         return match.group(1)
# #     else:
# #         # Step 3: Check for keywords if no direct match is found
# #         if re.match(r'figurative|figuratively', cleaned_text, re.IGNORECASE):
# #             return 'i'
# #         elif re.match(r'literal|literally', cleaned_text, re.IGNORECASE):
# #             return 'l'
# #         else:
# #             return None  # If no match or keywords are found, return None

# # print(extract_label(x))