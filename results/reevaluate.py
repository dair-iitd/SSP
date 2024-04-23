from seqeval.metrics import f1_score, precision_score, recall_score
import traceback
import json
import sys
import os

def load_jsonl(filepath):
    lines = open(filepath, 'r').readlines()
    data = [json.loads(line.strip()) for line in lines]
    return data

def score_sets(data):

    try:
        gold_labels = [f['gold_labels'] if isinstance(f['gold_labels'][0], str) else [g[1] for g in f['gold_labels']] for f in data]
        pred_labels = [f['pred_labels'] if isinstance(f['pred_labels'][0], str) else [g[1] for g in f['pred_labels']] for f in data]
        precision = precision_score(gold_labels, pred_labels)
        recall = recall_score(gold_labels, pred_labels)
        f1 = f1_score(gold_labels, pred_labels)
    except Exception as e:
        # print(f"Could not compute scores after example {i}, please compute them manually later")
        # print(e)
        traceback.exc_info()
        precision = 0
        recall = 0
        f1 = 0

    scores = {}
    scores['precision'] = precision
    scores['recall'] = recall
    scores['f1'] = f1
    scores['total'] = len(data)
    return scores

def save_scores(save_dir, scores):
    with open(os.path.join(save_dir, f'accuracies.csv'), 'w') as accfile:
        accfile.write(f"precision,recall,f1,total\n")
        accfile.write(f"{scores['precision']},{scores['recall']},{scores['f1']},{scores['total']}\n")

def score_dir(dirpath):
    data = load_jsonl(os.path.join(dirpath, 'responses.json'))
    score = score_sets(data)
    save_scores(dirpath, score)

def score_family(directory):

    # Traverse the family subdirectories
    for expt_dir in os.listdir(directory):
        expt_path = os.path.join(directory, expt_dir)
        if not os.path.isdir(expt_path):
            continue
        for lang_dir in os.listdir(expt_path):
            lang_path = os.path.join(expt_path, lang_dir)
            if os.path.isdir(lang_path):
                score_dir(lang_path)

score_dir(sys.argv[1])
