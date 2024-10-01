import os
import pandas as pd
from sklearn.metrics import f1_score


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

for model in models:

    count = 0

    for filename in os.listdir("."):

        if model in filename:
            count += 1
    
    print(f"model: {model}, count: {count}")


# for filename in os.listdir("."):
#     if "flant5small" in filename:
#         print(filename)

def calculate_f1_score(file_path, true_label, average='macro'):
    """
    Reads a CSV file, extracts true and predicted labels, and calculates the F1 score.

    Parameters:
    - file_path (str): Path to the CSV file.
    - true_label_col (str): Name of the column containing the true labels.
    - predicted_label_col (str): Name of the column containing the predicted labels.
    - average (str): Averaging method for F1 score ('micro', 'macro', 'weighted', etc.).

    Returns:
    - float: The F1 score.
    """
    # Step 1: Load the CSV file using pandas
    df = pd.read_csv(file_path)
    
    # Step 2: Extract the true and predicted labels
    true_labels = [true_label] * 1033
    predicted_labels = df["pred"]
    
    # Step 3: Calculate the F1 score
    f1 = f1_score(true_labels, predicted_labels, average=None)
    
    return f1

for filename in os.listdir("."):

    if filename.endswith(".csv"):

        true_label = filename.split("_")[0]

        if true_label == "figurative":
            # correct_labels_for_file = ["i"] * 1033
            correct_labels_for_file = "i"
        elif true_label == "literal":
            # correct_labels_for_file = ["l"] * 1033
            correct_labels_for_file = "l"

        print(f"filename: {filename}: {calculate_f1_score(filename, correct_labels_for_file, average='macro')}")


# taking gpt4o as a sample

filename = "figurative_gpt4o_p1.csv"

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_csv(filename)

y_true = ["i"] * 1033
y_pred = df["pred"]

print(precision_recall_fscore_support(y_true,y_pred,average="macro"))
# (0.3333333333333333, 0.2920296869958051, 0.3113175094599243, None)

print(precision_recall_fscore_support(y_true,y_pred,average=None,labels=["i","l","u"]))
# (array([1., 0., 0.]), 
# array([0.87608906, 0.        , 0.        ]), 
# array([0.93395253, 0.        , 0.        ]), 
# array([1033,    0,    0]))

print(precision_recall_fscore_support(y_true,y_pred,average=None,labels=["i","l",]))
# (array([1., 0.]), 
# array([0.87608906, 0.        ]), 
# array([0.93395253, 0.        ]), 
# array([1033,    0]))


lit_file = "literal_gpt4o_p1.csv"

z_true = ["l"] * 1033
z_pred = pd.read_csv(lit_file)["pred"]

print(precision_recall_fscore_support(z_true,z_pred,average=None,labels=["i","l",]))
# (array([0., 1.]), 
# array([0.        , 0.89545015]), 
# array([0.        , 0.94484168]), 
# array([   0, 1033]))

########## combined!

trues = y_true + z_true
preds = list(y_pred) + list(z_pred) 

print(precision_recall_fscore_support(trues,preds,average=None,labels=["i","l",]))
# (array([0.93395253, 0.91133005]), 
# array([0.87608906, 0.89545015]), 
# array([0.9040959 , 0.90332031]), 
# array([1033, 1033]))

print(precision_recall_fscore_support(trues,preds,average=None))
# (array([0.93395253, 0.91133005, 0.        ]), 
# array([0.87608906, 0.89545015, 0.        ]), 
# array([0.9040959 , 0.90332031, 0.        ]), 
# array([1033, 1033,    0]))

print(precision_recall_fscore_support(trues,preds,average="macro"))
# (0.6150941925469522, 0.5905130687318491, 0.6024720721986347, None)

print(f1_score(trues, preds, average="macro"))
# 0.6024720721986347

print(f1_score(y_true,y_pred,average="macro"))
# 0.3113175094599243
print(f1_score(z_true,z_pred,average="macro"))
# 0.3149472250595846

print(f1_score(y_true,y_pred,average=None))
print(f1_score(z_true,z_pred,average=None))
print(f1_score(trues,preds,average=None))

# [0.93395253 0.         0.        ]
# [0.         0.94484168 0.        ]
# [0.9040959  0.90332031 0.        ]



# print(f1_score(y_true,y_pred,average="macro"))

# from sklearn import metrics
import matplotlib.pyplot as plt

# confusion_matrix = metrics.confusion_matrix(trues, preds)

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels= ["i", "l", "u"])

# cm_display.plot()
# plt.show()

# plt.savefig("confusion.jpeg")


# print(precision_recall_fscore_support(trues,preds))

# from sklearn.metrics import precision_score,recall_score
# print(precision_score(trues,preds,average="micro"))
# print(precision_score(trues,preds,average="macro"))
# print(precision_score(trues,preds,average=None))
# # [0.93395253 0.91133005 0.        ]

# print(precision_score(trues,preds,average="samples"))
# # print(recall_score(trues,preds,average="micro"))

print(precision_recall_fscore_support(trues,preds,average=None,labels=["i","l","u"]))
# (array([0.93395253, 0.91133005, 0.        ]), 
# array([0.87608906, 0.89545015, 0.        ]), 
# array([0.9040959 , 0.90332031, 0.        ]), 
# array([1033, 1033,    0]))

print(precision_recall_fscore_support(y_true,y_pred,average="macro"))
# (0.3333333333333333, 0.2920296869958051, 0.3113175094599243, None)

# this fits the calculations done on paper
# this implementation = average="binary"
# true labels are either i or l
# figurative instances only
print(precision_recall_fscore_support(y_true,y_pred,))
# (array([1., 0., 0.]), 
#  array([0.87608906, 0.        , 0.        ]), 
#  array([0.93395253, 0.        , 0.        ]), 
#  array([1033,    0,    0])))

print(precision_recall_fscore_support(trues,preds))
# (array([0.93395253, 0.91133005, 0.        ]), 
#  array([0.87608906, 0.89545015, 0.        ]), 
#  array([0.9040959 , 0.90332031, 0.        ]), 
#  array([1033, 1033,    0]))

print(precision_recall_fscore_support(trues,preds,average=None))

u_added_trues = trues + ["u"]
u_added_preds = preds + ["l"]

print(precision_recall_fscore_support(u_added_trues,u_added_preds,average=None))
# (array([0.93395253, 0.91043307, 0.        ]), 
# array([0.87608906, 0.89545015, 0.        ]), 
# array([0.9040959 , 0.90287945, 0.        ]), 
# array([1033, 1033,    1]))


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

confusion_matrix = confusion_matrix(y_true, y_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels= ["i", "l", "u"])

cm_display.plot()
plt.show()

plt.savefig("confusion.jpeg")


from sklearn.metrics import classification_report

print(classification_report(trues,preds,target_names=["i","l","u"], digits=4))

print(classification_report(y_true,y_pred,target_names=["i","l","u"], digits=4))
print(classification_report(z_true,z_pred,target_names=["i","l","u"], digits=4))


tn, fp, fn, tp = confusion_matrix(trues, preds,).ravel()
print(tn, fp, fn, tp) 