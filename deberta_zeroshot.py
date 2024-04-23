from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, pdb, sys
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
from tqdm import tqdm
import json
gold = []
preds = []
probs = []
max_probs = []
targets = []
labels = {"entailment": 0, "neutral": 1, "contradiction":2}
with open("data/processed/test_" + sys.argv[1] + ".tsv") as f:
	#test_dict = json.load(f)
	l = f.readline()
	#pdb.set_trace()
	for item in tqdm(f):
		premise, hypothesis, label, _ = item.strip().split("\t")
		targets.append(labels[label])
		gold.append(label)
		#premise = item["premise"]
		#hypothesis = item["hypothesis"]

		input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
		output = model(input["input_ids"])  # device = "cuda:0" or "cpu"
		#pdb.set_trace()
		prediction = torch.softmax(output["logits"][0], -1)
		#pdb.set_trace()
		probs.append(prediction)
		max_probs.append(max(prediction).item())
		prediction = prediction.tolist()
		label_names = ["entailment", "neutral", "contradiction"]
		#prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
		pred_ind = prediction.index(max(prediction))
		preds.append(label_names[pred_ind])
		

		#break
	#break
#pdb.set_trace()
import os
output_predict_file = os.path.join("deberta_opensrc_outputs/", sys.argv[1]+"_predictions.json")
with open(output_predict_file, "w") as writer:
    write_preds = []
    for pred_, gold_ in zip(preds, gold):
        #pdb.set_trace()
        write_preds.append(pred_)
    json.dump(write_preds, writer)

f_w=open("deberta_opensrc_outputs/" + sys.argv[1] + "_max_probs.json", "w")
json.dump(max_probs, f_w)
f_w.close()
# acc = sum(1 for a,b in zip(preds, gold) if a==b)/ len(preds)
# print("Total Accuracy in", sys.argv[1], "is - ", str(acc*100), "%")

# neutral_ind = [i for i in range(len(gold)) if gold[i] == "neutral"]
# ent_ind = [i for i in range(len(gold)) if gold[i] == "entailment"]
# con_ind = [i for i in range(len(gold)) if gold[i] == "contradiction"]

# neutral_pred = [preds[it] for it in neutral_ind]
# neutral_acc = sum(1 for a in neutral_pred if a == "neutral")/len(neutral_pred)

# ent_pred = [preds[it] for it in ent_ind]
# ent_acc = sum(1 for a in ent_pred if a == "entailment")/len(ent_pred)

# con_pred = [preds[it] for it in con_ind]
# con_acc = sum(1 for a in con_pred if a == "contradiction")/len(con_pred)

# print("entailment Accuracy in", sys.argv[1], "is - ", str(ent_acc*100), "%")
# print("contradiction Accuracy in", sys.argv[1], "is - ", str(con_acc*100), "%")
# print("neutral Accuracy in", sys.argv[1], "is - ", str(neutral_acc*100), "%")

# probs = torch.stack(probs)
# targets = torch.tensor(targets)
# pdb.set_trace()
# from torchmetrics.classification import MulticlassCalibrationError
# ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='l1')
# ece_value = ece_metric(probs, targets)
# print(f"ECE L1: {ece_value.item()}")

# ece_metric = MulticlassCalibrationError(num_classes=3, n_bins=10, norm='max')
# ece_value = ece_metric(probs, targets)
# print(f"ECE max: {ece_value.item()}")

# tau=float(sys.argv[2])
# max_probs = [max(prob).item() for prob in probs]
# corr=[1 if p==g else 0 for p,g in zip(preds,gold)]
# corr_tau = [corr[j] for j,it in enumerate(max_probs) if it >= tau]

# print("Fraction of more than ", str(tau), "is: - ", sum(corr_tau)*100/len(corr_tau))
