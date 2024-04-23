import sys, pdb
lines = open(sys.argv[1]).readlines()
preds = [line.strip().split("\t")[0] for line in lines]
gold = [line.strip().split("\t")[1] for line in lines]
#pdb.set_trace()
acc = sum(1 for a,b in zip(preds, gold) if a==b)/ len(preds)
print("Total Accuracy in", sys.argv[1], "is - ", str(acc*100), "%", "(Total ", str(len(preds)), ")")

LABELS = ["0_contradiction","1_entailment","2_neutral"]
neutral_ind = [i for i in range(len(gold)) if gold[i] == "neutral"]
ent_ind = [i for i in range(len(gold)) if gold[i] == "entailment"]
con_ind = [i for i in range(len(gold)) if gold[i] == "contradiction"]

neutral_pred = [preds[it] for it in neutral_ind]
neutral_acc = sum(1 for a in neutral_pred if a == "neutral")/len(neutral_pred)

ent_pred = [preds[it] for it in ent_ind]
#pdb.set_trace()
ent_acc = sum(1 for a in ent_pred if a == "entailment")/len(ent_pred)

con_pred = [preds[it] for it in con_ind]
con_acc = sum(1 for a in con_pred if a == "contradiction")/len(con_pred)


print("contradiction Accuracy in", sys.argv[1], "is - ", str(con_acc*100), "%")
print("entailment Accuracy in", sys.argv[1], "is - ", str(ent_acc*100), "%")
print("neutral Accuracy in", sys.argv[1], "is - ", str(neutral_acc*100), "%")

from sklearn.metrics import f1_score, precision_score, recall_score

print("Micro-F1:", f1_score(gold, preds, average='micro'))
print("Macro-F1:", f1_score(gold, preds, average='macro'))
print("Precision:", precision_score(gold, preds, average='macro'))
print("Recall:", recall_score(gold, preds, average='macro'))

from sklearn.metrics import classification_report

print(classification_report(gold, preds, target_names=LABELS))
