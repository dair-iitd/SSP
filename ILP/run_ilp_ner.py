import argparse
import pandas as pd 
#from codex_run import few_shot_prompting, save_result_finqa, save_result_tatqa
#from generate_prompt import *

#from utils import load_file
import numpy as np
import sys, pdb
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='None')
    parser.add_argument('--taus', type=str, default='None')
    parser.add_argument('--it', type=int, default=1) 
    parser.add_argument('--test_path', type=str, default='datasets/finqa/test.json')
    parser.add_argument('--train_metadata_path',type=str,default='data_cache/finqa/metadata/finqa_train_df.csv')
    parser.add_argument('--test_metadata_path',type=str,default='data_cache/finqa/metadata/finqa_test_df.csv')
    parser.add_argument('--text_retriever',type=str,default='data_cache/finqa/text_retriever/retrieved_text_finqa_test.csv')
    parser.add_argument('--similarity_matrix',type=str,default='data_cache/finqa/similarity_matrices/finqa_test_sim.txt')
    parser.add_argument('--max_output_length',type=int,default=309,help='Maximum length allocated to the output in tokens')
    parser.add_argument('--max_model_length',type=int,default=4096,help='Maximum token capacity (input+output) opf the LLM')
    parser.add_argument('--remove_invalid_train',type=bool,default=True,help='if True, remove train instances that were not converted correctly to python code')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--output_path',type=str,default="output/finqa/predictions/run1/")
    #SEER parameters
    parser.add_argument('--alpha',type=float,default=0.75)
    parser.add_argument('--beta',type=float,default=0)
    parser.add_argument('--lamda',type=float,default=0.0)
    parser.add_argument('--modules',type=list,default=['modality'],choices=[['modality'],['modality','answer_type']])
    parser.add_argument('--n_exemplars', type=int, default=8, help='Number of n-shot training examples.') 
    parser.add_argument('--k',type=int,default=200)
    #CODEX parameters
    parser.add_argument('--key',type=str,default='OPENAI_API_KEY')
    parser.add_argument('--model', type=str, default='code-davinci-002')
    parser.add_argument('--data_dir', type=str, default='data_zgul') 
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--constr', action='store_true')
    parser.add_argument('--rem_lab_cons', action='store_true')
    parser.add_argument('--rem_conf_cons', action='store_true')
    parser.add_argument('--zplus', action='store_true')
    parser.add_argument("--pool_size", type=int, default=100)    
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    LANG = args.lang
    print(LANG)
    similarity = np.load("/home/vipul/llm/iterate/data/new/"+LANG+"/tgt_sim.npy")
    K = 100
    #TAU = 0.6
    M = args.n_exemplars
    if args.it == 8:
        pdb.set_trace()
        M=1
    BETAS = [0.1, 0.1, 0.1, 0.1]
    LABELS = ['PER', 'LOC', "ORG", "O"]
    #tau_90per = {'aym':{'entailment':0.8, 'contradiction':0.7, 'neutral':0.66}, 'gn': {'entailment':0.86, 'contradiction':0.76, 'neutral':0.66}, 'quy':{'entailment':0.7720, 'contradiction':0.7056, 'neutral':0.6357} , 'nah':{'entailment':0.7805 , 'contradiction':0.7276 , 'neutral':0.6460}}
    if args.zplus:
        pdb.set_trace()
        tau_80per = json.load(open(args.data_dir+"/"+LANG+"_perc_stats_zplus.json"))
        test_df = pd.read_csv(args.data_dir+"/"+LANG+"_all_stats_zplus.csv")    
    else:
        tau_80per = json.load(open(args.data_dir+"/"+LANG+"_perc_stats.json"))
        test_df = pd.read_csv(args.data_dir+"/"+LANG+"_all_stats.csv")
    #pdb.set_trace()
    # pdb.set_trace()
    # with open("deberta_ft_outputs/"+LANG+"_max_probs.json") as f:
    #     conf = json.load(f)
    # with open("deberta_ft_outputs/"+LANG+"_predictions.json") as f:
    #     preds = json.load(f)
    # taus_80per = [tau_80per[LANG][pred_] for pred_ in preds]
    # taus_90per = [tau_90per[LANG][pred_] for pred_ in preds]

    # test_df = {'conf': conf, 'predictions': preds, 'taus_80': taus_80per, 'taus_90': taus_90per}
    # for label_ in LABELS:
    #     cnt_label = [1 if pred_ == label_ else 0 for pred_ in preds]
    #     test_df[label_] = cnt_label
    # test_df = pd.DataFrame.from_dict(test_df)
    #pdb.set_trace()
    if args.constr:
        from ilp_constr import SEER_SEQ_GURB
    else:
        pdb.set_trace()
        from ilp import SEER 
    print(args.taus)
    #pdb.set_trace()
    seer = SEER_SEQ_GURB(k=K,M=M,taus=tau_80per,num_it=args.it,
                betas=BETAS,
                labels=LABELS)
    import time
    start = time.time()
    selections = []
    sel_lens = []
    for i in range(similarity.shape[0]):
        selection, sel_len_ = seer.get_few_shot_exemplars(i,similarity,test_df,args.rem_lab_cons, args.rem_conf_cons,num_it=args.it, cand_size=args.pool_size)
        selections.append(np.array(selection))
        sel_lens.append(sel_len_)
    end = time.time()
    print("Time: ", (end - start))
    selections = np.array(selections)
    #np.save("outputs_seq_80percile/"+LANG + "_80percile_8ex_ILP_seq_grb_zplus.npy", selections)
    if args.zplus:
        pdb.set_trace()
        if args.rem_lab_cons:
            np.save(LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_no_lab_cons_zplus.npy", selections)
        elif args.rem_conf_cons:
            np.save(LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_no_conf_cons_zplus.npy", selections)
        else:
            np.save(LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_zplus.npy", selections)
    else:
        if args.rem_lab_cons:
            if "llama" in args.data_dir:
                np.save(LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_no_lab_cons_llama.npy", selections)
            else:
                np.save(LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_no_lab_cons_zgul_lam_"+str(args.lamda)+".npy", selections)
        elif args.rem_conf_cons:
            if "llama" in args.data_dir:
                np.save(LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_no_conf_cons_llama.npy", selections)
            else:
                np.save(LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_no_conf_cons_zgul_lam_"+str(args.lamda)+".npy", selections)
        else:
            if "llama" in args.data_dir:
                np.save(LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_llama.npy", selections)
            else:
                np.save(args.data_dir+LANG + "_80percile_"+str(M)+"ex_ILP_seq_grb_zgul_lam_"+str(args.lamda)+"_"+str(args.pool_size)+"_v5.npy", selections)
    #selections = np.load(LANG+"_80percile_8ex_rem_lab_cons_ILP_seq_grb_zplus.npy")

if args.zplus:
    preds = json.load(open(args.data_dir+"/"+LANG+"_100_zgul_plus_predictions.json"))
else:
    preds = json.load(open(args.data_dir+"/"+LANG+"_predictions.json"))
from dataset import TabularDataset
import sys

if args.zplus:
    sds = TabularDataset(args.data_dir+"/"+LANG+"_100_zgul_plus_pred.tsv", delimiter='\t')
else:
    sds = TabularDataset(args.data_dir+"/"+LANG+"_pred.tsv", delimiter='\t')

all_samples = [sds[i].copy() for i in range(len(sds))]

import pdb, os
#pdb.set_trace()
write_samples = [sample['input'].split(" ") for sample in all_samples]

per_ex = {'PER':[0.0]*len(selections), 'LOC':[0.0]*len(selections), 'ORG':[0.0]*len(selections), 'DATE':[0.0]*len(selections), 'O':[0.0]*len(selections)}
for j,it in enumerate(selections):
    for idx in it:
        #print(write_samples[idx], preds[idx])
        #pdb.set_trace()
        #print(preds[idx])
        for label in LABELS:
            if label != 'O':
                per_ex[label][j] += preds[idx].count('B-'+label)+preds[idx].count('I-'+label)+preds[idx].count('O-'+label)
            else:
                per_ex[label][j] += preds[idx].count(label)

    #print("END of Examples")

#pdb.set_trace()
print("Selection stats: ")
for label in LABELS:
    print(label, "Avg per prompt")
    print(sum(per_ex[label])/len(per_ex[label]))

#print("Avg. selected prompts", sum(sel_lens)/len(sel_lens))


tot = 0
sat = 0
for i in range(len(selections)):
    for label in LABELS:
        for sel_ in selections[i]:
            tot += 1
            if test_df.loc[sel_, label+'_probs'] >= tau_80per[label+'_80_PERC_PROB']:
            #if tau_80per[label+'_80_PERC_ENTR'] >= test_df.loc[i,label+'_entr']:
                sat += 1
        

print("Sat. conf constr.", sat/tot, "tot:- ", tot, "sat:- ", sat)


tot = 0
sat = 0
for i in range(len(selections)):
    for j,label in enumerate(LABELS):
        tot += 1
        if sum([test_df.loc[k, label+"_cnt"] for k in selections[i]]) >= BETAS[j] :
            sat += 1
        

print("Sat. label constr.", sat/tot, "tot:- ", tot, "sat:- ", sat)
