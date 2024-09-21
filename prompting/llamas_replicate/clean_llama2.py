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

    match = re.findall(r"[O|o]utput: (i|l|'l'|'i')", text, re.IGNORECASE)
    if len(match) >= 2 and check_if_all_elem_identical(match):
        return match[0].lower()
    elif len(match) >= 2:
        return "u"
    elif len(match) == 1:
        return match[0].lower()
    
    # match = re.findall(r"[O|o]utput: '(i|l)'", text,)
    # if len(match) >= 2 and check_if_all_elem_identical(match):
    #     return match[0]
    # elif len(match) >= 2:
    #     return "u"
    # elif len(match) == 1:
    #     return match[0]
    else:
        api_key = "sk-proj-qqjGfSUcVxvSzEOeHBzeT3BlbkFJrwkt5SSwLCZgNvaAcg7G"
        prompter = OpenAIPrompter(key=api_key, model="gpt-4")

        answer_from_gpt = prompter.prompt(f"Extract the answer from the model. If the model thinks the answer is figurative or 'i', return the letter 'i'. If the model thinks the answer is literal, return the letter 'l'. If the model cannot give a conclusive answer due to safe-guarding, return the letter 'u'. This is the answer: {text}")

        print(f"answer_from_gpt:{answer_from_gpt}")


def extract_label_p3(text):
    match = re.findall(r"output: ('i'|'l'|l|i)", str(text), re.IGNORECASE)
    if match:
        return match[0].replace("'", "")
    else:
        return "check"
    
def extract_label_p2(text):

    match = re.findall(r"output: ('i'|'l'|l|i)", str(text), re.IGNORECASE)
    if match:
        return match[0].replace("'", "").lower()
    else:
        pass

    match = re.findall(r"output:\n\n(i) \(figurative\)\n", text, re.IGNORECASE)
    if match:
        return match[0].lower()
    else:
        # return "check"
        pass
    
    match = re.findall(r'"(i|l)"', text)
    if match:
        return match[0]
    else:
        # return "check"
        pass

    match = re.findall(r"Output:\n(i|l) \(", text, re.IGNORECASE)
    if match:
        return match[0]
    else:
        return "check"
    

    
def get_gpt_to_extract_label(text, extracted_label):
    if extracted_label == "check":
        answer_from_gpt = prompter.prompt(f"Extract the answer from the text. If the text suggests the answer is figurative or 'i', return the letter 'i'. If the text suggests the answer is literal, return the letter 'l'. If the text cannot give a conclusive answer due to safe-guarding, return the letter 'u'. This is the answer: {text}")

        print(f"answer_from_gpt:{answer_from_gpt}")
        # return f"afg_{answer_from_gpt}"
        return answer_from_gpt
        
    else:
        return extracted_label

predictions_dir = "."
api_key = "sk-proj-qqjGfSUcVxvSzEOeHBzeT3BlbkFJrwkt5SSwLCZgNvaAcg7G"
prompter = OpenAIPrompter(key=api_key, model="gpt-4")

for filename in os.listdir(predictions_dir):

    if "llama27bchat_p3" in filename and filename.endswith(".csv"):
        print(filename)

        try:
            save_path = os.path.join(predictions_dir, "cleaned_llama27bchat") 
            os.mkdir(save_path)

        except FileExistsError as e:
            pass

        df = pd.read_csv(filename)
        df['response'] = df['sentence'].progress_apply(extract_label_p3)
        df.to_csv(f'cleaned_llama27bchat/{filename}', index=False, columns=["idiom","response"])


    if "llama27bchat_p2" in filename and filename.endswith(".csv"):
        print(filename)
        df = pd.read_csv(filename)
        df['response'] = df['sentence'].progress_apply(extract_label_p2)

        df['response_checked'] = df.progress_apply(lambda x: get_gpt_to_extract_label(x["sentence"], x["response"]), axis=1)
        df.to_csv(f'cleaned_llama27bchat/{filename}', index=False, columns=["idiom","response_checked"])
        # df.to_csv(f"withCheckedMarkings_llama27bchat/{filename}", index=False, columns=["idiom", "response", "response_checked",])

    if "llama27bchat_p1" in filename and filename.endswith(".csv"):
        print(filename)
        df = pd.read_csv(filename)
        df['response'] = df['sentence'].progress_apply(extract_label_p2)

        df['response_checked'] = df.progress_apply(lambda x: get_gpt_to_extract_label(x["sentence"], x["response"]), axis=1)
        df.to_csv(f'cleaned_llama27bchat/{filename}', index=False, columns=["idiom","response_checked"])
        # df.to_csv(f"withCheckedMarkings_llama27bchat/{filename}", index=False, columns=["idiom", "response", "response_checked",])


