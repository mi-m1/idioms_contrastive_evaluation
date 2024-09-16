import pandas as pd
from sklearn.cluster import KMeans


df = pd.read_csv("../prompting/dataset/literal_1032.csv")
# df = pd.read_csv("frequency_results.csv")

idioms_in_dataset = df.Idiom
print(set(df.Idiom))
print(len(set(df.Idiom)))

df_freq = pd.read_csv("frequency_results.csv")

df_in_dataset_only = df_freq[df_freq['expression'].isin(idioms_in_dataset)]
print(len(df_in_dataset_only))

df_in_dataset_only.to_csv("frequency_results_396.csv")
# wcss = []
# X = df[['expression', 'number of hits']].copy()

# for i in range(1, 4):
#     kmeans = KMeans(n_clusters=i, random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.intertia_)

# import matplotlib.pyplot as plt
# import seaborn as sns 

# sns.set()

# plt.plot(range(1, 4), wcss)
# plt.title('Selecting the Numbeer of Clusters using the Elbow Method')
# plt.xlabel('Clusters')
# plt.ylabel('WCSS')
# plt.show()


# import numpy as np

# def group_logarithmic(values, num_bins):
#     values = np.array(values)
#     min_val = np.min(values[values > 0])  # Avoid log(0)
#     max_val = np.max(values)
    
#     # Create logarithmic bins
#     bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)
#     groups = {i: [] for i in range(num_bins - 1)}
    
#     # Assign values to bins
#     bin_indices = np.digitize(values, bins) - 1
    
#     for value, bin_index in zip(values, bin_indices):
#         if bin_index < num_bins - 1:
#             groups[bin_index].append(value)
    
#     return groups

# # Example usage:
# frequencies = df["number of hits"]
# groups = group_logarithmic(frequencies, 5)
# print(groups)
