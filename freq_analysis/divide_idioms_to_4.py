import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

df = pd.read_csv("frequency_results_396.csv")

expressions = df.expression
counts = df["number of hits"]
# plt.plot(expressions, counts)
# plt.show()

# plt.yscale("log")

# plt.savefig("log.png")


log_frequencies = np.log10(counts)
print(log_frequencies[0:10])
print(log_frequencies.tail)
print(log_frequencies.shape)
print(type(log_frequencies))
# print(log_frequencies.columns)

max_log = 4.263778

threshold = max_log / 4

filtered_df1 = log_frequencies[log_frequencies < threshold]
print(len(filtered_df1))

filtered_df2 = log_frequencies[log_frequencies < threshold*2]
print(len(filtered_df2))

filtered_df3 = log_frequencies[log_frequencies < threshold*3]
print(len(filtered_df3))

filtered_df4 = log_frequencies[log_frequencies <= max_log]
print(len(filtered_df4))

ls1 = [x for x in range(0, len(filtered_df1)+1)]
ls2 = [x for x in range(0, len(filtered_df2)+1)]
ls3 = [x for x in range(0, len(filtered_df3)+1)]
ls4 = [x for x in range(0, len(filtered_df4)+1)]

print(len(ls1),len(ls2),len(ls3),len(ls4),)


group1 = set(ls1)
print(f"this is group1: {len(group1)}")

group2 = set(ls2) - set(ls1)
print(f"this is group2: {len(group2)}")

group3 = set(ls3) - set(ls2)
print(f"this is group3: {len(group3)}")

group4 = set(ls4) - set(ls3)
print(f"this is group4 {len(group4)}")

mapping = [group1, group2, group3, group4]

# group1_expressions = []
# group2_expressions = []
# group3_expressions = []
# group4_expressions = []

expressions_division = []

for map in mapping:

    # for index in map:
            
    #     group1_expressions = [exp for exp in expressions]

    group_expressions = [expressions[x] for x in map]

    expressions_division.append(group_expressions)

# print(f"this is expressions_division: {expressions_division}")

for index, idiom_set in enumerate(expressions_division):

    filename = "group" + str(index+1) + ".csv"
    with open(filename, "w") as f:
        for idiom in idiom_set:
            f.write(idiom+",\n")



