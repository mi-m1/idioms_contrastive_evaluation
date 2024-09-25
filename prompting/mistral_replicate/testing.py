import os
from getpass import getpass
from tqdm import tqdm
import argparse
import replicate
import pandas as pd
import csv

# output = replicate.run(
#   "mistralai/mistral-7b-instruct-v0.2",
#   input={"prompt": "an iguana on the beach, pointillism"}
# )
# print(output)
# print(" ".join(output))
 

 # The mistralai/mistral-7b-instruct-v0.2 model can stream output as it's running.
x = []


# raw_prompt1 = f"Is the expression '{idiom}' used figuratively or literally in the sentence: '{sentence}'. Answer 'i' for figurative, 'l' for literal. Put your answer after 'output: '"

for event in replicate.stream(
    "mistralai/mistral-7b-instruct-v0.2",
    input={"prompt": "is the expression 'spill the beans' used figuratively or literally in the sentence 'he spilled the beans over the floor'? Answer 'i' for figurative, 'l' for literal. Put your answer after 'output: '"},
):
    # print(str(event), end="")
    # print("trying new way of formating:\n")
    # print(str(event).replace(r'\n', '').replace(r"\r", ""))
    # print(str(event))
    # print("\n".join(event))

    # s = str(event).splitlines()

    s = ''.join(str(event).splitlines())
    print(s)
#     x.append(str(event))
# print(x)
    