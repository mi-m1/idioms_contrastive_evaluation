import pandas as pd
import csv
import glob

model_abrs = ["flant5small",
              "flant5xxl",
              "flant5xl",
              "flant5large",
              "gpt4",
              "gpt4o",
              "gpt35turbo",
              "mistral7binstructv0.3",
              "llama38binstruct",
              "llama27bchathf"]


for filename in glob.glob("../predictions/*.csv"):

    print(filename)
    # break
    df = pd.read_csv(filename)

    print(df.head)

    if "figurative_" in filename:

        labels = ['i'] * len(df)
    elif "literal_" in filename:

        labels = ['l'] * len(df)

    df.insert(2, "label", labels, True)
    df.columns = ["idiom", "sentence", "label"]
    # df.rename(columns={"idiom": "idiom", "sentence": "prediction", "label":"label"})
    # df.rename(columns=)
    print(df)

    save_filename = filename.split("/")[-1]
    print(save_filename)
    df.to_csv(f"collected_preds/{save_filename}", index=False)



files_for_each_model = {}
for model in model_abrs:
    fig_files = sorted(glob.glob(f"collected_preds/figurative_{model}_p*.csv"))
    lit_files = sorted(glob.glob(f"collected_preds/literal_{model}_p*.csv"))

    files_for_each_model[model] = list(zip(fig_files, lit_files))
    # break

print(files_for_each_model)

# merge two files
for model, pair in files_for_each_model.items():

    for index in range(0, len(pair)):


        # print(pair[0][0])
        # break
        csv1 = pd.read_csv(pair[index][0])

        # Read the contents of the second CSV file
        csv2 = pd.read_csv(pair[index][1])

        # Concatenate the two dataframes vertically
        merged_csv = pd.concat([csv1, csv2], ignore_index=True)

        # run = pair[0][0][-6:-4]
        # print(run)
        newname = f"{pair[index][0].split('/')[-1].replace('figurative_', '').replace('.csv', '')}"


        save_filename = f"merged_preds/merged_{newname}.csv"

        # print(save_filename)

    
        # Write the merged dataframe to a new CSV file
        merged_csv.to_csv(save_filename, index=False)


