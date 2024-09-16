import pandas as pd

groups_paths = ["group1.csv", "group2.csv", "group3.csv", "group4.csv"]
groups = []
for g in groups_paths:

    with open(g, "r") as f:

        content = f.readlines()
        content = [x.strip() for x in content]
        content = [x.replace(",", "") for x in content]

        groups.append(content)

print(groups)

path = "../prompting/collect_results/merged_preds/merged_flant5large_p1.csv"

df = pd.read_csv(path)

print(f"df:")
print(df.head)

print(groups[0])
# print(f"this is group: \t{group}")
# print(df.loc[df['idiom'].isin(group)])

df["idiom"] = df["idiom"].str.strip()

print(type(groups[0]))
df_for_group = df[df['idiom'].isin(groups[0])]

# df_for_group = df[df['idiom'].map(lambda x:x==groups[0])]

print(df_for_group)