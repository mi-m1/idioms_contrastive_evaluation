from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support




def get_scores(preds_tup, labels):

  preds = [item[1] for item in preds_tup]

  prfs = precision_recall_fscore_support(labels, preds, labels=["i", "l",])

  results = {
    'precision': prfs[0],
    'recall': prfs[1],
    "fscore":prfs[2],
    "support":prfs[3],
    "macro f1": f1_score(labels, preds, average="macro", labels=["i", "l",]),
    "accuracy": accuracy_score(labels, preds)
}
  return results

ip = [("eager beaver","i"), ("break a leg", "l"), ("spill the beans", "l"), ("eat the cake", "l")]



true_labels = ["l"] * 4
scores = get_scores(ip, true_labels)

print(scores)
