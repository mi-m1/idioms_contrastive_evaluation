import pandas as pd
import glob

# df = pd.read_csv("/mnt/parscratch/users/acq22zm/ae/freq_analysis/results_combined_3runs/g1_flant5large.csv")

# print(df)

# class_i = df.iloc[::2].reset_index(drop=True)  # Odd rows
# print(class_i)
# class_l = df.iloc[1::2].reset_index(drop=True)  # Even rows
# print(class_l)


# print(class_i.mean().to_frame().T)

def get_std_for_group(group, metric):

    ls = []

    for filename in glob.glob(f"results_std_per_class/{group}_*.csv"):

        split = filename.split("_")

        group_num = split[-3]
        setting = split[-2]
        model = split[-1]

        df = pd.read_csv(filename)
        print(df)

        f1_score = df[metric][0]

        ls.append((f"{model}_{setting}", f1_score))

    print(ls)

# get_std_for_group("g1", "f1")
# get_std_for_group("g2", "f1")
get_std_for_group("g3", "f1")
# get_std_for_group("g4", "f1")


    


