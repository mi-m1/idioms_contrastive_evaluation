import re
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import os

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


    
def extract_label_llama213b(text):
    match = re.findall("Output: (i|l)", text, re.IGNORECASE)

    if match:
        return match[0]
    else:
        api_key = "sk-proj-qqjGfSUcVxvSzEOeHBzeT3BlbkFJrwkt5SSwLCZgNvaAcg7G"
        prompter = OpenAIPrompter(key=api_key, model="gpt-4")

        answer_from_gpt = prompter.prompt(f"Extract the answer from the model's response. If the model thinks the answer is figurative or 'i', return the letter 'i'. If the model thinks the answer is literal, return the letter 'l'. If the model cannot give a conclusive answer, return the letter 'u'. If the model can't decide between 'i' and 'l', then return 'u'. This is the model's response: {text}")
        answer_from_gpt = answer_from_gpt.replace("'", "")
        answer_from_gpt = answer_from_gpt.replace('"', '')
        
        print(f"answer_from_gpt:{answer_from_gpt}")

        return f"answer_from_gpt:{answer_from_gpt}"
        

predictions_dir = "."

for filename in os.listdir(predictions_dir):

    if filename.endswith(".csv") and "llama213bchat" in filename:
        folder_name = "cleaned_llama213bchat_1shot"
        try:
            save_path = os.path.join(predictions_dir, folder_name) 
            os.mkdir(save_path)
        except FileExistsError as e:
            pass

        print(filename)

        df = pd.read_csv(filename)
        df['response'] = df["sentence"].progress_apply(extract_label_llama213b)

        df.to_csv(f'{folder_name}/{filename}', index=False, columns=["idiom","response"])