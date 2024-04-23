from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
import sys, pdb
f1=open(sys.argv[1])
f2=open(sys.argv[2])

#pdb.set_trace()
lines1=f2.read().strip().split("\n\n")
preds=[[it.split("\t")[1] for it in sent.split("\n")] for sent in lines1]
for it in preds:
	for j in range(len(it)):
		if it[j] == "":
			it[j] = 'O'

lines2=f1.read().strip().split("\n\n")
gold=[[it.split("\t")[1] for it in sent.split("\n")] for sent in lines2]
#pdb.set_trace()
try:
	f1_ = f1_score(gold, preds, average="micro")
	prec_ = precision_score(gold, preds)
	rec_ = recall_score(gold, preds)
	print(classification_report(gold, preds, mode='strict'))
except:
	#pdb.set_trace()
	try:
		f1_ = f1_score(gold, preds[:len(gold)], average="micro")
		prec_ = precision_score(gold, preds[:len(gold)])
		rec_ = recall_score(gold, preds[:len(gold)])
		print(classification_report(gold, preds[:len(gold)], mode='strict'))
	except:
		f1_ = f1_score(gold[:len(preds)], preds, average="micro")
		prec_ = precision_score(gold[:len(preds)], preds)
		rec_ = recall_score(gold[:len(preds)], preds)
		print(classification_report(gold[:len(preds)], preds, mode='strict'))
	#pdb.set_trace()
#f1_ = f1_score(gold, preds)
print("Precision - ", prec_)
print("Recall - ", rec_)
print("F1 - ", f1_)