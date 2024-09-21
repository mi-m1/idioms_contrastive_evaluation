import os
from tqdm import tqdm
import pandas as pd
import re
from openai import OpenAI

tqdm.pandas()

class OpenAIPrompter:
    def __init__(self, key, model,):
        """
        Initialize the OpenAIPrompter class.

        Parameters:
        - api_key (str): Your OpenAI API key.
        - model (str): The model to use for the prompts. Default is "gpt-3.5-turbo".
        """
        self.key = key
        self.model = model

    # def prompt(self, prompt_text, max_tokens, temperature, top_p, n):
    def prompt(self, prompt_text):
        """
        Send a prompt to the OpenAI model and return the response.

        Parameters:
        - prompt_text (str): The text prompt to send to the model.
        - max_tokens (int): The maximum number of tokens to generate in the response.
        - temperature (float): Sampling temperature to use. Higher values means the model will take more risks.
        - top_p (float): Nucleus sampling parameter. The model will consider the smallest set of tokens with cumulative probability top_p.
        - n (int): Number of completions to generate for the prompt.

        Returns:
        - response (str): The generated response from the model.
        """

        # client = OpenAI(self.api_key)
        client=OpenAI(api_key = self.key)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                              "type": "text",
                              "text": prompt_text,
                            }
                        ]

                    }
                ]
                # max_tokens=max_tokens,
                # temperature=temperature,
                # top_p=top_p,
                # n=n
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred: {e}"


def extract_response(text):
    response = text.split(r"[/INST]")[-1]
    return response

def check_if_all_elem_identical(lst):
    return all(i == lst[0] for i in lst)


def extract_label(text):

    match = re.findall(r'[O|o]utput: (i|l|L|I)', text,)

    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0].lower()
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0].lower()
    
    match = re.findall(r"[O|o]utput: '(i|l)'", text,)

    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0]
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0]
    
    match = re.findall(r"[O|o]utput: \((i|l)\)", text,)

    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0]
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0]

    match_double_quotes = re.findall(r'\"(i|l)\"', text, re.IGNORECASE)
    print(f"match_double_quotes: {match_double_quotes}")

    if match_double_quotes:
        return match_double_quotes[0]
    else:
        # return "u"
        # return text
        # print(f"response: {response}")
        pass

    match = re.findall(r"(i|l) \(literal|figurative\)", text,)

    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0]
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0]
    

    match = re.findall(r"Output:\n (i|l)", text,)

    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0]
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0]
    
    match = re.findall(r"Output : (i|l)", text,)

    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0]
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0]
    

    match = re.findall(r"in the sentence as (figurative|literal)", text,)

    if match:
        if match[0] == "figurative":
            return "i"
        elif match[0] == "literal":
            return "l"
    else:
        pass


    match = re.findall(r"in the given sentence is used (figuratively|literally)", text,)

    if match:
        if match[0] == "figuratively":
            return "i"
        elif match[0] == "literally":
            return "l"
    else:
        pass

    # in the sentence you provided is used figuratively

    match = re.findall(r"in the sentence you provided is used (figuratively|literally)\.", text,)

    if match:
        if match[0] == "figuratively":
            return "i"
        elif match[0] == "literally":
            return "l"
    else:
        pass

    # (figuratively|figurative|literal|literally)\.

    match = re.findall(r"(figuratively|figurative|literal|literally)\.", text,)

    if match:
        if match[0] == "figuratively" or match[0] == "figurative":
            return "i"
        elif match[0] == "literally" or match[0] == "literal":
            return "l"
    else:
        pass

    # I would answer 'i' for figurative
    match = re.findall(r"would answer '(i|l)' for", text,)

    if match:
        return match[0]
    

    # in the given sentence as figurative (i)
    match = re.findall(r"in the given sentence as (figurative|literal)", text,)

    if match:
        if match[0] == "figurative":
            return "i"
        elif match[0] == "literal":
            return "l"
    else:
        pass

# is used figuratively in the sentence.

    match = re.findall(r"is used (figuratively|literally) in the sentence\.", text,)

    if match:
        if match[0] == "figuratively":
            return "i"
        elif match[0] == "literally":
            return "l"
    else:
        pass
    

    # I would answer (l) for literal.
    match = re.findall(r"I would answer \((i|l)\)", text,)

    if match:
        return match[0]
    
    
    match = re.findall(r' is used (figuratively|literally)\.', text, re.IGNORECASE)
    print("got a match here!")
    if match:
        if match[0] == "figuratively":
            return "i"
        elif match[0] == "literally":
            return "l"

    # output:\n\n(i|l) \(figurative\)
    match = re.findall(r'Output:\n\ni \(figurative\)', text,)
    # print("got a match here!")
    if match:
        return "i"
    else:
        print(f"text_here: {text}")

    # Regex to match the letter "i" after "Output:"
    pattern = r"(?<=Output:\s*)(i|l) \(figurative|literal\)"

    # Search for the pattern in the text
    match = re.search(pattern, text, re.IGNORECASE)

    # Check if a match is found
    if match:
        print(f"Matched letter: {match.group()}")
    else:
        print("No match found")




    # check how many times the words figurative and literal appear in the response
    count_figurative = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("figurative"), text))
    count_figuratively = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("figuratively"), text))
    count_literal = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("literal"), text))
    count_literally = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("literally"), text))

    if (count_figurative + count_figuratively) > (count_literal + count_literally):
        print(f"text: {text}")
        print(f"$i")
        return "i"
    elif (count_figurative + count_figuratively) < (count_literal + count_literally):
        print(f"text: {text}\n")
        print(f"$l")
        return "l"
    elif (count_figurative + count_figuratively) == (count_literal + count_literally):
        print("check")
        return "u"
    else:
        pass

    print(f"f_counts: {count_figurative + count_figuratively}")
    print(f"l_counts: {count_literal + count_literally}")

    match = re.findall(r"in the sentence as (figurative|literal)", text,)

    if match[0] == "figurative":
        return "i"
    elif match[0] == "literal":
        return "l"

predictions_dir = "."

for filename in os.listdir(predictions_dir):

    if "figurative_llama27bchat" in filename and filename.endswith(".csv"):
        
        print(filename)

        try:
            save_path = os.path.join(predictions_dir, "cleaned_llama27bchat") 
            os.mkdir(save_path)

        except FileExistsError as e:
            pass

        
        df = pd.read_csv(filename)
        df['response'] = df['sentence'].progress_apply(extract_label)
        df.to_csv(f'cleaned_llama27bchat/{filename}', index=False, columns=["idiom","response"])


