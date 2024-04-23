import json
import torch
import pandas as pd
import math
MAX = 1000000000
import sys, pdb

LANG = sys.argv[1]

PERC = float(sys.argv[2])/100

#preds = json.load(open(LANG+"_100_zgul_plus_predictions.json", "r"))
preds = json.load(open(LANG+"_predictions.json", "r"))

#entr_list = json.load(open(LANG+"_100_zgul_plus_entropy.json", "r"))
#entr_list = json.load(open(LANG+"_100_zgul_entropy.json", "r"))

#all_probs = torch.load(LANG+"_100_max_probs_all_zplus.pth")
all_probs = json.load(open(LANG+"_max_probs.json"))

all_stats = []

#all_labels = ['PER', 'LOC', "ORG", "DATE", "O"]
#all_labels = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
all_labels = ["entailment", "neutral", "contradiction"]
idx = 0
for pred_, probs_ in zip(preds, all_probs):
	#pdb.set_trace()
	curr_dict = {}
	for label_ in all_labels:
		#pdb.set_trace()
		if label_ == pred_:
			cnt_label = 1
			curr_dict[label_+"_probs"] = probs_
			#curr_dict[label_+"_entr"] += ent_
		else:
			cnt_label = 0
			curr_dict[label_+"_probs"] =  0.0
		curr_dict[label_+"_cnt"] = cnt_label 

	all_stats.append(curr_dict)			

df = pd.DataFrame.from_dict(all_stats)
df.to_csv(LANG+"_" + sys.argv[2] + "_all_stats.csv")		
#pdb.set_trace()

sorted_lab = {}
write_dict = {}
for label_ in all_labels:
	for stat in all_stats:
		if label_ not in sorted_lab:
			sorted_lab[label_] = []
		if stat[label_+"_cnt"] != 0:
			sorted_lab[label_].append(stat[label_+"_probs"])

	curr_l = sorted(sorted_lab[label_])
	#pdb.set_trace()
	if curr_l == []:
		write_dict[label_+"_"+sys.argv[2]+"_PERC_PROB"] = 0.0
	else:
		percentile_index = math.floor(PERC * len(curr_l))
		write_dict[label_+"_"+sys.argv[2]+"_PERC_PROB"] = curr_l[percentile_index]
		print(label_, "PERCENTILE ", PERC, "Max probs", curr_l[percentile_index], "Total: ", len(curr_l)-percentile_index+1, "out of ", len(curr_l))

with open(LANG+"_" +sys.argv[2] +"_perc_stats.json", "w") as f:
	json.dump(write_dict, f)



