import pandas as pd
from tqdm import tqdm
import random


# df_fig = pd.read_csv("/mnt/parscratch/users/acq22zm/ae/prompting/dataset/figurative_1032.csv")
# df_lit = pd.read_csv("/mnt/parscratch/users/acq22zm/ae/prompting/dataset/literal_1032.csv")
# df.Idiom, df.Sentence

# idiom_sent_tpl = list(zip(df_fig.Idiom, df_fig.Sentence))



def get_random_item_from_list(lst):
    random_item = random.choice(lst)
    return random_item

def get_random_oneshot_example(idiom_in_question, seed, fig_data, lit_data):

    df_fig = pd.read_csv(fig_data)
    df_lit = pd.read_csv(lit_data)

    random.seed(seed)

    # make sure random_idiom isn't the idiom in question
    possible_choices = [v for v in df_fig.Idiom if v != idiom_in_question]

    random_idiom = random.choice(possible_choices)
    # print(random_idiom)

    # get figurative example for random idiom
    figurative_examples = list(df_fig.loc[df_fig['Idiom'] == random_idiom, 'Sentence'])
    oneshot_fig_example = get_random_item_from_list(figurative_examples)

    literal_examples = list(df_lit.loc[df_lit['Idiom'] == random_idiom, 'Sentence'])
    oneshot_lit_example = get_random_item_from_list(literal_examples)

    return random_idiom, oneshot_fig_example, oneshot_lit_example


fig_data = "/mnt/parscratch/users/acq22zm/ae/prompting/dataset/figurative_1032.csv"
lit_data = "/mnt/parscratch/users/acq22zm/ae/prompting/dataset/literal_1032.csv"
print(get_random_oneshot_example("against the grain", 2432, fig_data, lit_data))

# get_random_oneshot_example("nest egg", 2343242,)


# for idiom, sentence in tqdm(idiom_sent_tpl):

#     print(idiom)
    
#     random_idiom, oneshot_fig_example, oneshot_lit_example = get_random_oneshot_example(idiom, 2343242)

#     print(random_idiom)
#     print(oneshot_fig_example)
#     print(oneshot_lit_example)

#     break

