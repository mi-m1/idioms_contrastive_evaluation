import pandas as pd
from tqdm import tqdm

pairs =  {'gpt4o_p1': ['figurative_gpt4o_p1.csv', 'literal_gpt4o_p1.csv'], 'gpt4o_p2': ['figurative_gpt4o_p2.csv', 'literal_gpt4o_p2.csv'], 'gpt4o_p3': ['figurative_gpt4o_p3.csv', 'literal_gpt4o_p3.csv'], 'gpt35turbo_p1': ['figurative_gpt35turbo_p1.csv', 'literal_gpt35turbo_p1.csv'], 'gpt35turbo_p2': ['figurative_gpt35turbo_p2.csv', 'literal_gpt35turbo_p2.csv'], 'gpt35turbo_p3': ['figurative_gpt35turbo_p3.csv', 'literal_gpt35turbo_p3.csv'], 'flant5xxl_p1': ['figurative_flant5xxl_p1.csv', 'literal_flant5xxl_p1.csv'], 'flant5xxl_p2': ['figurative_flant5xxl_p2.csv', 'literal_flant5xxl_p2.csv'], 'flant5xxl_p3': ['figurative_flant5xxl_p3.csv', 'literal_flant5xxl_p3.csv'], 'flant5xl_p1': ['figurative_flant5xl_p1.csv', 'literal_flant5xl_p1.csv'], 'flant5xl_p2': ['figurative_flant5xl_p2.csv', 'literal_flant5xl_p2.csv'], 'flant5xl_p3': ['figurative_flant5xl_p3.csv', 'literal_flant5xl_p3.csv'], 'flant5large_p1': ['figurative_flant5large_p1.csv', 'literal_flant5large_p1.csv'], 'flant5large_p2': ['figurative_flant5large_p2.csv', 'literal_flant5large_p2.csv'], 'flant5large_p3': ['figurative_flant5large_p3.csv', 'literal_flant5large_p3.csv'], 'flant5small_p1': ['figurative_flant5small_p1.csv', 'literal_flant5small_p1.csv'], 'flant5small_p2': ['figurative_flant5small_p2.csv', 'literal_flant5small_p2.csv'], 'flant5small_p3': ['figurative_flant5small_p3.csv', 'literal_flant5small_p3.csv'], 'llama31405binstruct_p1': ['figurative_llama31405binstruct_p1.csv', 'literal_llama31405binstruct_p1.csv'], 'llama31405binstruct_p2': ['figurative_llama31405binstruct_p2.csv', 'literal_llama31405binstruct_p2.csv'], 'llama31405binstruct_p3': ['figurative_llama31405binstruct_p3.csv', 'literal_llama31405binstruct_p3.csv'], 'llama370binstruct_p1': ['figurative_llama370binstruct_p1.csv', 'literal_llama370binstruct_p1.csv'], 'llama370binstruct_p2': ['figurative_llama370binstruct_p2.csv', 'literal_llama370binstruct_p2.csv'], 'llama370binstruct_p3': ['figurative_llama370binstruct_p3.csv', 'literal_llama370binstruct_p3.csv'], 'llama38binstruct_p1': ['figurative_llama38binstruct_p1.csv', 'literal_llama38binstruct_p1.csv'], 'llama38binstruct_p2': ['figurative_llama38binstruct_p2.csv', 'literal_llama38binstruct_p2.csv'], 'llama38binstruct_p3': ['figurative_llama38binstruct_p3.csv', 'literal_llama38binstruct_p3.csv'], 'llama270bchat_p1': ['figurative_llama270bchat_p1.csv', 'literal_llama270bchat_p1.csv'], 'llama270bchat_p2': ['figurative_llama270bchat_p2.csv', 'literal_llama270bchat_p2.csv'], 'llama270bchat_p3': ['figurative_llama270bchat_p3.csv', 'literal_llama270bchat_p3.csv'], 'llama213bchat_p1': ['figurative_llama213bchat_p1.csv', 'literal_llama213bchat_p1.csv'], 'llama213bchat_p2': ['figurative_llama213bchat_p2.csv', 'literal_llama213bchat_p2.csv'], 'llama213bchat_p3': ['figurative_llama213bchat_p3.csv', 'literal_llama213bchat_p3.csv'], 'llama27bchat_p1': ['figurative_llama27bchat_p1.csv', 'literal_llama27bchat_p1.csv'], 'llama27bchat_p2': ['figurative_llama27bchat_p2.csv', 'literal_llama27bchat_p2.csv'], 'llama27bchat_p3': ['figurative_llama27bchat_p3.csv', 'literal_llama27bchat_p3.csv'], 'gpt4_p1': ['figurative_gpt4_p1.csv', 'literal_gpt4_p1.csv'], 'gpt4_p2': ['figurative_gpt4_p2.csv', 'literal_gpt4_p2.csv'], 'gpt4_p3': ['figurative_gpt4_p3.csv', 'literal_gpt4_p3.csv']}

predictions_file_df = pd.read_csv("figurative_gpt4o_p1.csv")

# if setting == "figurative":
#     correct_label = "i"

# elif setting == "literal":
#     correct_label = "l"

correct_label = "i"
predictions_file_df['is_correct'] = predictions_file_df['pred'] == correct_label
print(predictions_file_df)

idiom_groups = predictions_file_df.groupby('idiom')['is_correct'].all()
print(f'idiom_groups: {idiom_groups}')

correct_predictions = idiom_groups.sum()
print(f"correct_predictions: {correct_predictions}")

print(f"len(idiom_groups): {len(idiom_groups)}")

lc_accuracy = (correct_predictions / len(idiom_groups))*100

print(lc_accuracy)

# def loose_consistency(self, pairs):

#     def calculate_lc():
#         pass
        
#     for model_run, file_pair in pairs.items():

#         print(f"model_run:{model_run}, file_pair:{file_pair}")
#         df_fig = pd.read_csv(file_pair[0])
#         df_lit = pd.read_csv(file_pair[1])
#         df_merged = df_fig.append(df_lit, ignore_index=True)

#         print(df_merged.shape)
#         break
