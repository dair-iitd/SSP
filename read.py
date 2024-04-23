from datasets import load_dataset
import pdb
import json
import random
import pandas as pd
train = dict()
TRAIN_TOT = 10000
label_dict = {"entailment":"entailment", "neutral": "neutral", "contradictory": "contradiction"}

def check(PATH):
	lines = open(PATH).readlines()
	return_dict = []
	for it,line in enumerate(lines):
		if it == 0:
			print(line)
			continue
		try:
			get_items = line.strip().split("\t")
			assert len(get_items) == 3
			return_dict.append({"premise": get_items[0], "hypothesis": get_items[1], "label": get_items[2]})
		except:
			#pdb.set_trace()
			print(it)
	return return_dict

for source in ["en", "es"]:
	PATH="multinli.train."+source+".tsv"
	#check(PATH)
	#pdb.set_trace()
	shuffled_dataset = check(PATH)
	random.shuffle(shuffled_dataset)
	#pdb.set_trace()
	train[source] = dict()
	for ex in shuffled_dataset:
		ex['label'] = label_dict[ex['label']]
		if ex['label'] not in train[source]:
			train[source][ex['label']] = []
		if len(train[source][ex['label']]) < int(TRAIN_TOT/3):
			train[source][ex['label']].append({"premise": ex['premise'], "hypothesis": ex['hypothesis']})
		try:
			if (0 in train[source]) & (len(train[source][0]) == int(TRAIN_TOT/3)) & (1 in train[source]) & (len(train[source][1]) == int(TRAIN_TOT/3)) & (2 in train[source]) & (len(train[source][2]) == int(TRAIN_TOT/3)):
				break

		except:
			continue
	
	with open("train_"+source+".json", "w") as f_w:
		json.dump(train[source], f_w)


# test = dict()
# TEST_TOT = 100
# for source in ["aym", "quy", "gn", "nah"]:
# 	PATH="americasnli/data/anli_final/test/"+source+".tsv"
# 	shuffled_dataset = pd.read_csv(PATH, sep="\t").to_dict('records')
# 	random.shuffle(shuffled_dataset)
# 	#pdb.set_trace()
# 	# shuffled_dataset = dataset_["test"].shuffle(seed=random.randint(1,99))
# 	test[source] = dict()
# 	for ex in shuffled_dataset:
# 		#pdb.set_trace()
# 		if ex['label'] not in test[source]:
# 			test[source][ex['label']] = []
# 		if len(test[source][ex['label']]) < int(TEST_TOT/3):
# 			test[source][ex['label']].append({"premise": ex['premise'], "hypothesis": ex['hypothesis']})
# 		try:
# 			if (0 in test[source]) & (len(test[source][0]) == int(TEST_TOT/3)) & (1 in test[source]) & (len(test[source][1]) == int(TEST_TOT/3)) & (2 in test[source]) & (len(test[source][2]) == int(TEST_TOT/3)):
# 				break
				
# 		except:
# 			continue
	
# 	with open("test_"+source+".json", "w") as f_w:
# 		json.dump(test[source], f_w)
