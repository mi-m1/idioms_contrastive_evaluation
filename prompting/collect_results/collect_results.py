import glob
import pandas as pd
import csv
# import 
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import io


for filename in glob.glob("merged_preds/*.csv"):
    split = filename.split("_")

    merged = split[0]
    model = split[1]
    run = split[2]

    df = pd.read_csv(filename)
    df['idiom'] = df['idiom'].apply(str)
    df['sentence'] = df['sentence'].apply(str)
    df['label'] = df['label'].apply(str)


    # print(df.head)

    true_labels = df.label
    preds = df.sentence


    prfs = precision_recall_fscore_support(true_labels, preds, labels=["i", "l"])

    print(prfs)
    results = {
    'precision': prfs[0],
    'recall': prfs[1],
    "fscore":prfs[2],
    "support":prfs[3],
    "f1": f1_score(true_labels, preds, labels=["i", "l"], average=None),
    "macro f1": f1_score(true_labels, preds, average="macro", labels=["i", "l"]),
    "accuracy": accuracy_score(true_labels, preds),
    # "per class accuracy": accuracy_score(true_labels, preds, method)
    }

    print(filename)
    print(results)
    output_filename = filename.replace("merged_preds/merged_", "")
    output_filename = output_filename.replace(".csv", "")

    print(output_filename)

    # break

    cr = classification_report(true_labels, preds, labels=["i", "l"], digits=6)
    # with open(f"results/{output_filename}", "w", encoding="utf-8") as f:
    #     f.write(cr)
        # f.write(results)

    df_out = pd.read_csv(io.StringIO(cr), sep="\t")
    print(df)

    df_out.to_csv("results/"+output_filename+"_cr.csv", index=False)

    df_out = pd.DataFrame.from_dict(results,)
    df_out.to_csv("results/"+output_filename+"_results.csv", index=False)   

    cr = classification_report(true_labels, preds, labels=["i", "l"], output_dict=True)
    print(f"this is cr:")
    print(cr)


    # df_out = pd.DataFrame.from_dict(cr).transpose()


    # df_out.to_csv("results/" + output_filename, index=False)
    
    # break

    # matrix = confusion_matrix(true_labels, preds)
    # print(matrix.diagonal()/matrix.sum(axis=0))
    # print(matrix.diagonal()/matrix.sum(axis=1))
     

    # break


