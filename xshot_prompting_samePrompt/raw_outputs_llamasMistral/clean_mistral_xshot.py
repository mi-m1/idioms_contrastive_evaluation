import pandas as pd
import re
import os

def check_for_keywords(text):
    if "literal" in text or "literally" in text:
        prediction = "l"
        return prediction 
    elif "figurative" in text or "figuratively" in text:
        prediction = "i"
        return prediction

    else:
        return None




def extract_label(text):
    # print(f"this is text: {text}")

    last_line = text.split("\n")[-1]
    print(f"\nthis is last line: {last_line}")



    last_part_of_last_line = last_line.split("  ")[-1]
    print(f"this is last_part: {last_part_of_last_line}")

    if "output:" in last_part_of_last_line:

        # output: i|l
        # output: i|l (sense)
        prediction_letter = last_part_of_last_line.split("output: ")[-1]
        # print(f"prediction_letter: {prediction_letter}")

        if len(prediction_letter) >= 2:

            if "(figurative)" in prediction_letter and "i" in prediction_letter:
                return "i"
            elif "(literal)" in prediction_letter and "l" in prediction_letter:
                return "l"

        else:
            return prediction_letter
            
    else:

        prediction = check_for_keywords(last_part_of_last_line)

        if prediction == None:
            print("prediction: u")
            return "u"
        else:
            print(f"prediction: {prediction}")
            return prediction




predictions_dir = "."

for filename in os.listdir(predictions_dir):
    if "literal_mistral7binstructv0.3_p3" in filename and "cleaned_" not in filename:

        print(filename)

        df = pd.read_csv(filename)

        
        df['cleaned_text'] = df['sentence'].apply(extract_label)
        df.to_csv(f'cleaned_{filename}', index=False, columns=["idiom","cleaned_text"])
        # break

# output: (i|'i'|'l'|l)"