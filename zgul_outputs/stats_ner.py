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
all_probs = torch.load(LANG+"_max_probs_all.pth")

all_stats = []

all_labels = ['PER', 'LOC', "ORG", "O"]
#all_labels = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
idx = 0
for pred_, probs_ in zip(preds, all_probs):
	#pdb.set_trace()
	curr_dict = {}
	for label_ in all_labels:
		if label_ != 'O':
			cnt_ = pred_.count('B-'+label_) + pred_.count('I-'+label_) + pred_.count('O-'+label_)   #NER
			#cnt_ = pred_.count(label_)    #POS
		else:
			cnt_ = pred_.count(label_)

		curr_dict[label_+"_cnt"] = cnt_ 

		curr_dict[label_+"_probs"] =  0.0
		#curr_dict[label_+"_entr"] = 0.0
		#pdb.set_trace()
		cnt_label = 0
		try:
			assert probs_.tolist()[len(pred_)] == -1.0 and probs_.tolist()[len(pred_)-1] != -1.0
		except:
			pdb.set_trace()
		for pre_, pro_ in zip(pred_, probs_.tolist()[:len(pred_)]):
			assert pro_ != 0.0
			if label_ in pre_ or label_ == pre_: #NER
			#if label_ == pre_: #POS
				#pdb.set_trace()
				if label_ != "O":
					cnt_label += 1
					curr_dict[label_+"_probs"] += pro_
					#curr_dict[label_+"_entr"] += ent_
				elif label_ == pre_:
					cnt_label += 1
					curr_dict[label_+"_probs"] += pro_
		
		if cnt_ == 0:
			#pdb.set_trace()
			try:
			    assert curr_dict[label_+"_probs"] == 0.0
			except:
			    pdb.set_trace()
			#curr_dict[label_+"_entr"] = MAX
		else:
			try:
				curr_dict[label_+"_probs"] /= cnt_label
			except:
				pdb.set_trace()
			#curr_dict[label_+"_entr"] /= cnt_label

	all_stats.append(curr_dict)			

df = pd.DataFrame.from_dict(all_stats)
df.to_csv(LANG+"_all_stats.csv")		
#pdb.set_trace()

# 90 per stats
sorted_lab = {}
write_dict = {}
# for label_ in all_labels:
# 	for stat in all_stats:
# 		if label_ not in sorted_lab:
# 			sorted_lab[label_] = []
# 		if stat[label_+"_cnt"] != 0:
# 			sorted_lab[label_].append(stat[label_+"_entr"])

# 	curr_l = sorted(sorted_lab[label_])
# 	#pdb.set_trace()
# 	if curr_l == []:
# 		write_dict[label_+"_"+sys.argv[2]+"_PERC_ENTR"] = 1000000000
# 	else:
# 		percentile_index = math.ceil((1-PERC) * len(curr_l))
# 		if len(curr_l) == 1:
# 			percentile_index = 0
# 		try:
# 			write_dict[label_+"_"+sys.argv[2]+"_PERC_ENTR"] = curr_l[percentile_index]
# 		except:
# 			pdb.set_trace()
# 		print(label_, "PERCENTILE ", PERC, "Entropy", curr_l[percentile_index], "Total: ", percentile_index+1, "out of ", len(curr_l))


# sorted_lab = {}

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

with open(LANG+"_perc_stats.json", "w") as f:
	json.dump(write_dict, f)



