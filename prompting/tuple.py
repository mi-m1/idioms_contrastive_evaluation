import pandas as pd
import csv 

def save_predictions(idiom_preds, setting, path, model_abr, run):
  filename = f"{path}{setting}_{model_abr}_{run}.csv"
  data= pd.DataFrame.from_dict([idiom_preds])

  data.to_csv(filename, index=False)

ip = [(324,542),]

with open('ur file.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['name','num'])
    for row in ip:
        csv_out.writerow(row)

print(ip)
# save_predictions(ip, "testing", "", "no", 1)


def save_predictions(idioms_preds, setting, path, model_abr, run):
   filename = f"{path}{setting}_{model_abr}_{run}.csv"
   
   with open(filename,'w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['idiom','sentence'])
    for row in idioms_preds:
        csv_out.writerow(row)


ip = [("eager beaver","i"), ("break a leg", "l")]

outputs_modified = [(idiom, 'o' if category not in ['i', 'l'] else category) for idiom, category in ip]
print(outputs_modified)


list_test = []

exp ="hi there"
answer1 = "i"
list_test.append((exp, answer1))

print(list_test)

dict_test = {"p1": list_test}
print(dict_test)


for run, preds in dict_test.items():
   print(run)

   outputs_modified = [(idiom, 'o' if category not in ['i', 'l'] else category) for idiom, category in preds]

   print(outputs_modified)
