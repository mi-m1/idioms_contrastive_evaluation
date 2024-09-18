import pandas as pd
import re
import os

def extract_label(text):
    # print(f"this is text: {text}")

    last_line = text.split("\n")[-1]
    # print(f"this is last line: {last_line}")

    prediction = last_line.split("output: ")
    print(f"this is prediction: {prediction}")

    prediction_letter = last_line.split("output: ")[-1]
    print(f"this is prediction_letter: {prediction_letter}")


    if len(prediction_letter) >= 2:

        if "(figurative)" in prediction_letter and "i" in prediction_letter:
            print(prediction_letter)
            return "i"
        elif "(literal)" in prediction_letter and "l" in prediction_letter:
            print(prediction_letter)

            return "l"
        else:
            return "u"

    else:
        return prediction_letter


    # pass

    # match = re.findall(r"output: (i|'i'|'l'|l)\"", text, flags=re.IGNORECASE)
    # print(f"match: {match}")   

    # if match:
    #     return match[0]

predictions_dir = "."

for filename in os.listdir(predictions_dir):
    if "mistral7binstructv0.3_" in filename and "cleaned_" not in filename:

        print(filename)

        df = pd.read_csv(filename)

        
        df['cleaned_text'] = df['sentence'].apply(extract_label)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])
        # break

# output: (i|'i'|'l'|l)"