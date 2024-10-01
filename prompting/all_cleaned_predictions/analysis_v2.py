import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
import os


predictions_dir = "."

models = [
    "gpt4o_",
    "gpt35turbo_",
    "flant5xxl_",
    "flant5xl_",
    "flant5large_",
    "flant5small_",
    "llama31405binstruct_",
    "llama370binstruct_",
    "llama38binstruct_",
    "llama270bchat_",
    "llama213bchat_",
    "llama27bchat_",
    # "mistral7binstructv0.3_",
    "gpt4_",
]

runs = [
    "p1",
    "p2",
    "p3"
]

settings = [
    "figurative",
    "literal",
]

# print(zip(models,settings, runs))

# combo = zip(models,settings, runs)
# print(list(combo))

# from itertools import cycle
# merged = zip(models, cycle(settings), cycle(runs))
# print(len(list(merged)))


from itertools import product
combinations = list(product(settings, models, runs))
print(len(combinations))
print(combinations)

combinations_without_settings = list(product(models, runs))
print(len(combinations_without_settings))
print(combinations_without_settings)

for filename in os.listdir(predictions_dir):

    if filename.endswith(".csv"):

        pass
