from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import csv
import re
from tqdm import tqdm
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Huggingface model ID")
parser.add_argument("--eval_dataset", help="Path to eval dataset file")
parser.add_argument("--model_abr", help="model_abbreviation")
parser.add_argument("--setting", help="figurative or literal")
args = parser.parse_args()



# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(args.model_name, token="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", device_map="auto", padding="left")
model = AutoModelForCausalLM.from_pretrained(args.model_name, token="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",device_map="auto", torch_dtype=torch.float16, resume_download=True,)
print("loaded model and tokeniser")
model.to('cuda')


#chat template
messages = [
{"role": "system", "content": "You're a language expert who can only generate one letter."},
{"role": "user", "content": "expression: 'spill the beans'\n sentence: 'he managed to spill the beans in the kitchen.' QUESTION: Is the expression figurative or literal? Generate the letter 'i' if the idiom is used figuratively, or generate 'l' if the expression is used literally."},
],

apply_chat_format = tokenizer.apply_chat_template(messages, tokenize=False)
# print(apply_chat_format[0])
# string = tokenizer.decode(apply_chat_format[0], skip_special_tokens=True)
# print(string)


dataset_location = args.eval_dataset
df = pd.read_csv(dataset_location)
print(df.shape)

# model_abr = "literal_llama_testing"

# setting = "literal"

def get_answer_from_model(prompt):

    # strings_after_A= []

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=70, pad_token_id=tokenizer.eos_token_id)
    string = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return string


# def extract_answer(idiom, output_string):
#     match = re.search(r'\[\s*[\'\"](i|l)[\'\"]\s*\]', output_string)
#     if match:
#         result = match.group(1)
#         # print(f"this is result")
#         # print(result)
#         return result
#     else:
#         print(f"No match found for: {idiom}")

# def apply_prompt_format(messages):
#    format = tok.apply_chat_templates(messages, tokenize=False)
#    return format

def save_predictions(idioms_preds, setting, path, model_abr, run):
   filename = f"{path}{setting}_{model_abr}_{run}.csv"
   
   with open(filename,'w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['idiom','sentence'])
    for row in idioms_preds:
        csv_out.writerow(row)

def process_output(output):
    # output = output.lower().replace("'", "")
    output = output.lower()
    if 'figurative' in output or 'figuratively' in output or ": i" in output:
        output = "i"
    elif 'literal' in output or 'literally' in output or ": l" in output:
        output = "l"
    elif output == "f":
        output = "i"
     
    output = output.replace("'", "")
    output = output.replace("`", "")
    output = output.replace("‘", "")
    output = output.replace("’", "")

    return output

def run_predictions(idioms, sentence, model_abr, setting):
# def run_and_save_predictions(idioms, sentence, model, setting):
    idioms_preds_p1 = []
    idioms_preds_p2 = []
    idioms_preds_p3 = []
    idiom_sent_dict= list(zip(idioms, sentence))

    for idiom, sentence in tqdm(idiom_sent_dict):

        #chat template
        # messages1 = [
        # {"role": "system", "content": "You're a language expert who can only generate one letter. Your task is to interpret the sentence, and generate a letter 'i' if the idiom is used figuratively, or generate 'l' if the expression is used literally."},
        # {"role": "user", "content": f"expression: '{idiom}'\n sentence: '{sentence}' QUESTION: Is the expression figurative or literal?"},
        # ],
        my_instruction = "Is the meaning of the expression idiomatic or literal? If used idiomatically, answer 'i', if literally, answer 'l'."
        expression = f"Expression: {idiom}"
        sent = f"Sentence: {sentence}"

        messages1 = [
        # {"role": "system", "content": "You are a language expert."},
        # {"role": "user", "content": f"expression: '{idiom}'\n sentence: '{sentence}' QUESTION: Is the expression figurative or literal? Generate the letter 'i' if the idiom is used figuratively, or generate 'l' if the expression is used literally. Only generate the letter."},
        {"role": "user", "content": f"{my_instruction + expression + sent}"},
        ],
        prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False)

        messages2 = [
        # {"role": "system", "content": "You are an assistant."},
        # {"role": "user", "content": f"expression: '{idiom}'\n sentence: '{sentence}' QUESTION: Given a contextual sentence and a expression, tell me if the expression is used figurative or literally. Either generate the letter 'i' if figurative, or generate the letter 'l' if literal."},
        {"role": "user", "content": f"In the sentence '{sentence}', is the expression '{idiom}' being used figuratively or literally? Respond with 'i' for figurative and 'l' for literal."},
        ],
        prompt2 = tokenizer.apply_chat_template(messages2, tokenize=False)

        messages3 = [
        # {"role": "system", "content": "You are a native speaker of English."},
        # {"role": "user", "content": f"expression: '{idiom}'\n sentence: '{sentence}' QUESTION: Does the expression hold a figurative or literal meaning in the contextual sentence? Generate a letter 'i' for figurative meaning, or 'l' for literal meaning."},
        {"role": "user", "content": f"How is the expression '{idiom}' used in this context: '{sentence}'. Output 'i' if the expression holds figurative meaning, output 'l' if the expression holds literal meaning."}
        ],
        prompt3 = tokenizer.apply_chat_template(messages3, tokenize=False)


        # print(apply_chat_format)
        # # string = tokenizer.decode(apply_chat_format[0], skip_special_tokens=True)
        # # print(string)

        # prompt1 = f"""<s>[INST] <<SYS>>
        # Is the expression used literally or figuratively? Only answer with lowercase 'i' for figurative or 'l' for literal only. 
        # Generate a Python list containing the letter.
        # <</SYS>>
        # The expression is '{idiom}'. The sentence is '{sentence}'[/INST]"""

        # prompt1 = f"""<s>[INST] <<SYS>>\nQ: Predict whether the expression given is being used idiomatically or literally. Only output one letter i for idiomatic, l for literal.\n<</SYS>>\nExpression: {idiom}. Context: {sentence}. A: [/INST]<s>
        #             """
        # prompt2 = f"<s>[INST] <<SYS>>\nGiven a contextual sentence and a expression, tell me if the expression is used figurative or literally. Either generate 'i' if figurative, or generate 'l' if literal. Generate a Python list containing the letter.\n<</SYS>>\n The expression is '{idiom}'. The sentence is '{sentence}'[/INST]"
        # prompt3 = f"<s>[INST] <<SYS>>\nDoes the expression hold a figurative or literal meaning in the following contextual sentence? Generate 'i' for figurative meaning, or 'l' for literal meaning. Generate a Python list containing the letter.\n<</SYS>>\n The expression is '{idiom}'. The sentence is '{sentence}'[/INST]"
        
        
        # prompt2 = f"""
        # # [INST] You are an assistant, who can only generate one letter. Given a contextual sentence and a expression, tell me if the expression is used figurative or literally. Either generate "i" if figurative, or generate "l" if literal.
        # # expression: '{idiom}'
        # # sentence; '{sentence}'
        # # Generate a Python list containing the letter.[/INST]"""

        # prompt3 = f"""
        # [INST] You are a native speaker of English, who can only generate one letter. Does the expression hold a figurative or literal meaning in the following contextual sentence? Generate "i" for figurative meaning, or "l" for literal meaning.
        # expression: '{idiom}'
        # sentence; '{sentence}'
        # Generate a Python list containing the letter.[/INST]"""

        answer1 = get_answer_from_model(prompt1)
        answer2 = get_answer_from_model(prompt2)
        answer3 = get_answer_from_model(prompt3)

        # print("Model said:")
        # print(answer1)

        # pred_letter1 = answer1.split("\n")[-1].lower()
        # pred_letter2 = answer2.split("\n")[-1].lower()
        # pred_letter3 = answer3.split("\n")[-1].lower()

        pred_letter1 = process_output(answer1)
        pred_letter2 = process_output(answer2)
        pred_letter3 = process_output(answer3)
        
        # answer2 = get_answer_from_model(pr3m3t2)
        # answer3 = get_answer_from_model(prompt3)

        # answer1_extracted = extract_answer(idiom, answer1)
        # answer2_extracted = extract_answer(idiom, answer2)
        # answer3_extracted = extract_answer(idiom, answer3)

        idioms_preds_p1.append((idiom, answer1))
        idioms_preds_p2.append((idiom, answer2))
        idioms_preds_p3.append((idiom, answer3))

        # prediction = int(prediction != 'i')
        # print(f"The model has predicted: \t{prediction}")

    # path = f"predictions/"
    path = f"."
    save_predictions(idioms_preds_p1, setting, path, model_abr, "p1")
    save_predictions(idioms_preds_p2, setting, path, model_abr, "p2")
    save_predictions(idioms_preds_p3, setting, path, model_abr, "p3")


    # return {"p1": idioms_preds_p1}
    return {"p1": idioms_preds_p1, "p2": idioms_preds_p2, "p3": idioms_preds_p3}


predictions_all_prompts = run_predictions(df.Idiom, df.Sentence, args.model_abr, args.setting)
