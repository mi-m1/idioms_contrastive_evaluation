import pandas as pd
import os
from tqdm import tqdm



familarity_df = pd.read_csv("cleaned_gpt4_estimates/Values-Table 1.csv")

daevid_df = pd.read_csv("frequency_revised.csv")

daevid_idioms = daevid_df["idiom"]
print(daevid_idioms)

familarity_idioms = familarity_df["Word"]
print(familarity_idioms)

overlap = []

for x in list(daevid_idioms):
    if x in list(familarity_idioms):
        overlap.append(x)

print(len(overlap))

print(overlap)

# df[df['A'].isin([3, 6])]

overlapped_df = familarity_df[familarity_df["Word"].isin(list(daevid_idioms))]
print(overlapped_df)
# ,Word,GPT_Fam_dominant,GPT_Fam_probs,Multilex_Zipf,Type,Subtype_MWE

renamed = overlapped_df.rename(columns={
    ',': ',', 
    'Word': 'idiom', 
    'GPT_Fam_dominant': 'GPT_Fam_dominant',
    'GPT_Fam_probs': 'GPT_Fam_probs',
    'Multilex_Zipf': 'Multilex_Zipf',
    'Type':'Type',
    'Subtype_MWE':'Subtype_MWE'}) 

renamed.to_csv("fam_daevid_identical_form.csv")

odd_ones_out = set(list(daevid_idioms)) - set(list(overlapped_df["Word"]))
print(odd_ones_out)