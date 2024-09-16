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
    if filename.endswith(".csv") and "llama27bchathf" in filename:

        df = pd.read_csv(filename)

        print(filename)
        df['cleaned_text'] = df['sentence'].apply(extract_label)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])

        break