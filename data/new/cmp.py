from dataset import JSONLDataset, TabularDataset, PickleDataset
#import models.openai as openai
#from util import parse_example, parse_tsv_example, score_sets
import numpy as np

# from dotenv import load_dotenv

# from prompt import PromptGenerator

import argparse
from tqdm.auto import tqdm
import os
import json
import shutil
#import logging
from datetime import datetime
import signal
import sys, pdb


# LANG = sys.argv[1]

def construct_prompt(idx, tgt_ds, src_ds):

    #ind_tgt = tgt_sim_mat[idx]
    #pdb.set_trace()
    try:
        preds_ = [word.split("_")[1] for word in tgt_ds[idx]['output'].strip().split(" ")]
        golds_ = [word.split("_")[1] for word in src_ds[idx]['output'].strip().split(" ")]
    except:
        pdb.set_trace()
    try:
        assert(len(preds_) == len(golds_))
    except:
        pdb.set_trace()
    #pdb.set_trace()
    #preds_golds = [pred_+"\t"+gold_ for pred_, gold_ in zip(tgt_ds[idx].copy()['pred_labels'], src_ds[idx].copy()['pred_labels'])]
    
    return preds_, golds_


all_intersec = []
all_exact = []
all_prec = []
all_rec = []
all_f1 = []
#for LANG in ['fo', 'got', 'gsw']:
#for LANG in ['fo']:
for LANG in sys.argv[2].split(","):
#for LANG in ['hau','ibo','kin','lug','luo']:
    args_source_dataset = LANG+"/test.tsv"

    if args_source_dataset.endswith('.json'):
        #pdb.set_trace()
        sds = JSONLDataset(args_source_dataset)
    elif args_source_dataset.endswith('.tsv'):
        sds = TabularDataset(args_source_dataset, delimiter='\t')
    else:
        pdb.set_trace()

    tds = None
    args_target_dataset = LANG+"/"+LANG+"_pred.tsv"
    if args_target_dataset.endswith('.json'):
        pdb.set_trace()
        tds = JSONLDataset(args_target_dataset)
    elif args_target_dataset.endswith('.tsv'):
        tds = TabularDataset(args_target_dataset, delimiter='\t')
    else:
        pdb.set_trace()

    x1=np.load(LANG+"/"+LANG+"_80percile_8ex_ILP_seq_grb_zgul.npy")
    #x1=np.load(LANG+"_80percile_8ex_ILP_seq_grb_zgul_lam_"+str(sys.argv[1])+"_"+sys.argv[3]+".npy")
    #x1=np.load(LANG+"_80percile_8ex_ILP_seq_grb_zgul_lam_0.0.npy")
    #x2 = np.load("outputs_seq_80percile/"+LANG+"_80percile_8ex_ILP_seq_grb_zgul.npy")
    x2=np.load(LANG+"/"+LANG+"_80percile_8ex_ILP_seq_grb_zgul_lam_"+str(sys.argv[1])+"_"+sys.argv[4]+".npy")
    
    l1 = []
    l2 = []

    for it in x1:
        l1 += it.tolist()

    for it in x2:
        l2 += it.tolist()

    # print("old", set(l1), len(set(l1)))
    # print("new", set(l2), len(set(l2)))
    #pdb.set_trace()
    all_intersec.append((sum([len(set(it1).intersection(set(it2))) for it1, it2 in zip(x1,x2)])))
    all_exact.append((sum([sum(it1==it2) for it1, it2 in zip(x1,x2)])))
    # preds = []
    # gold = []
    # #print(selections)
    # #pdb.set_trace()
    # for idx in l1:
    #     pred_, gold_ = construct_prompt(idx, tds, sds)
    #     preds.append(pred_)
    #     gold.append(gold_)

    # from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
    # try:
    #     f1_ = f1_score(gold, preds, average="micro")
    #     prec_ = precision_score(gold, preds)
    #     rec_ = recall_score(gold, preds)
    #     print(classification_report(gold, preds))
    # except:
    #     pdb.set_trace()
    #     try:
    #         f1_ = f1_score(gold, preds[:len(gold)], average="micro")
    #         prec_ = precision_score(gold, preds[:len(gold)])
    #         rec_ = recall_score(gold, preds[:len(gold)])
    #         print(classification_report(gold, preds[:len(gold)], mode='strict'))
    #     except:
    #         f1_ = f1_score(gold[:len(preds)], preds, average="micro")
    #         prec_ = precision_score(gold[:len(preds)], preds)
    #         rec_ = recall_score(gold[:len(preds)], preds)
    #         print(classification_report(gold[:len(preds)], preds, mode='strict'))
    #     #pdb.set_trace()
    # #f1_ = f1_score(gold, preds)
    # print("old Precision - ", prec_)
    # print("old Recall - ", rec_)
    # print("old F1 - ", f1_)


    preds = []
    gold = []
    #print(selections)
    #pdb.set_trace()
    for idx in l2:
        pred_, gold_ = construct_prompt(idx, tds, sds)
        preds.append(pred_)
        gold.append(gold_)

    from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
    #pdb.set_trace()
    try:
        f1_ = f1_score(gold, preds, average="micro")
        prec_ = precision_score(gold, preds)
        #pdb.set_trace()
        rec_ = recall_score(gold, preds)
        print(classification_report(gold, preds))
    except:
        pdb.set_trace()
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
    #print("new Precision - ", prec_)
    ##print("new Recall - ", rec_)
    #print("new F1 - ", f1_)
    all_prec.append(prec_)
    all_rec.append(rec_)
    all_f1.append(f1_)


print("Set intersec: ", sum(all_intersec)/len(all_intersec))
print("Exact intersec: ", sum(all_exact)/len(all_exact))
print("Avg precision: ", sum(all_prec)/len(all_prec))
print("Avg recall: ", sum(all_rec)/len(all_rec))
print("Avg f1: ", sum(all_f1)/len(all_f1))
print(all_intersec)
