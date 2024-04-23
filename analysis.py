import numpy as np
import json
from prompt import PromptGenerator
from dataset import JSONLDataset, TabularDataset, PickleDataset
import sys, pdb
def construct_prompt(example, ind_tgt, src_ds, tgt_ds, prompt, gold_anno=False):
	pg = PromptGenerator('prompts')
    # retrieve demos
	label_dict = {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe", "NULL": "NULL"}
	demos = []
	assert len(ind_tgt) == 8
	tgt_demos = [src_ds[i].copy() for i in ind_tgt if i != -1]
	tgt_labels = [tgt_ds[i] for i in ind_tgt if i != -1]
    
	demos += tgt_demos
    #pdb.set_trace()
	if not(gold_anno):
		for it, d in enumerate(demos):
			if "p2" not in prompt:
			    d['output'] = label_dict[tgt_labels[it]]
			else:
			    d['output'] = tgt_labels[it]
	else:
	    pdb.set_trace()
	    for d in demos:
	        if "p2" not in prompt:
	            d['output'] = label_dict[d['output']]
	#pdb.set_trace()

	    
	prompt = pg.create_prompt(f'{prompt}', demos=demos, sentence=example)
	return prompt

LANG = sys.argv[1]
lines = open("new_results_p2/gpt-4-turbo/americas/8t_ilp_grb/"+LANG+"/responses.json").readlines()
ilp_pred = [line.strip().split("\t")[0] for line in lines]
gold = [line.strip().split("\t")[1] for line in lines]
corr_pred = [1 if p==g else 0 for p,g in zip(ilp_pred, gold)]
lines = open("new_results_p2/gpt-4-turbo/americas/8t_ilp_no_lab_grb/"+LANG+"/responses.json").readlines()
ilp_pred_no_lab = [line.strip().split("\t")[0] for line in lines]
corr_pred_no_lab = [1 if p==g else 0 for p,g in zip(ilp_pred_no_lab, gold)]


label = sys.argv[3]

idxs = []
idx = 0
for c1,c2,g in zip(corr_pred, corr_pred_no_lab, gold):
	if sys.argv[2] == "corr":
		if g == label and c1 == 1 and c2 == 0:
			idxs.append(idx)
	else:
		if g == label and c1 == 0 and c2 == 1:
			idxs.append(idx)
	idx += 1

tsim1 = np.load("data/processed/"+LANG+"/"+LANG+"_90_per_ILP.npy")
tsim2 = np.load("data/processed/"+LANG+"/"+LANG+"_90_per_ILP_no_lab_cons.npy")

sds = TabularDataset("data/processed/test_"+LANG+".tsv", delimiter="\t")
tds = json.load(open("data/processed/"+LANG+"/"+LANG+"_predictions.json"))
#sds = TabularDataset("", delimiter="\t")



with open(LANG+"_confusion_"+sys.argv[2]+"_"+label+".txt", "w") as f_w:
	for i in idxs:
		sel1 = tsim1[i].tolist()
		example = sds[i].copy()
		prom1 = construct_prompt(example, sel1, sds, tds, "nli_8s_tsv_adapt_p2")

		sel2 = tsim2[i].tolist()
		prom2 = construct_prompt(example, sel2, sds, tds, "nli_8s_tsv_adapt_p2")

		f_w.write("Prompt ILP: " + prom1 + ilp_pred[i] + "\n")
		f_w.write("Prompt ILP no label: " + prom2 + ilp_pred_no_lab[i] + "\n")
		f_w.write("###SEP###")

