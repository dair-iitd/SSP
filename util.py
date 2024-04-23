import re
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
import pdb

def parse_tsv_example(task, example, response):
    
    lines = response.strip().split('\n')
    pred = [a.split('\t') for a in lines if '\t' in a]

    gold_parsed : list[tuple[str, str]] = [a.rsplit('_', 1) for a in example['output'].strip().split(' ')]
    pred_parsed : list[tuple[str, str]] = align_dp(task, [a[0] for a in gold_parsed], pred)

    assert(len(pred_parsed) == len(gold_parsed))

    gold_labels = [g[1] for g in gold_parsed]
    pred_labels = [p[1] for p in pred_parsed]

    return {
        'pred_labels': pred_labels,
        'gold_labels': gold_labels
    }

def parse_example(task, example, response):

    gold_parsed : list[tuple[str, str]] = [a.rsplit('_', 1) for a in example['output'].strip().split(' ')]
    pred_parsed : list[tuple[str, str]] = align_dp(task, [a[0] for a in gold_parsed], parse_pred(response))

    assert(len(pred_parsed) == len(gold_parsed))

    gold_labels = [g[1] for g in gold_parsed]
    pred_labels = [p[1] for p in pred_parsed]

    return {
        'pred_labels': pred_labels,
        'gold_labels': gold_labels
    }

def score_sets(data):
    #pdb.set_trace()
    try:
        pred_labels = data['responses']
        gold_labels = data["gold"]
        #precision = precision_score(gold_labels, pred_labels)
        #recall = recall_score(gold_labels, pred_labels)
        #f1 = f1_score(gold_labels, pred_labels)
        accuracy = sum(1 for x,y in zip(gold_labels,pred_labels) if x == y)/(len(pred_labels) - pred_labels.count("NULL"))
        non_null = len(pred_labels) - pred_labels.count("NULL")
    except e:
        print("Could not compute scores, please compute them manually later")
        precision = 0
        recall = 0
        f1 = 0
        acc = 0
        non_null = 0

    #data['precision'] = precision
    #data['recall'] = recall
    #data['f1'] = f1
    data["acc"] = accuracy
    data["non_null"] = non_null

def align_dp(task, gold_seq: list[str], parsed_seq: list[tuple[str,str]], default_lbl='O') -> list[tuple[str, str]]:
    """
    Aligns the two sequences using dynamic programming
    """

    n = len(gold_seq)
    m = len(parsed_seq)
    G = gold_seq
    P = parsed_seq
    S = np.zeros((n+1, m+1))

    # DP
    for i in range(n):
        for j in range(m):
            s = 0
            if G[i] == P[j][0]:
                s = 1 + S[i][j]
            else:
                s = max(S[i][j+1], S[i+1][j])
            S[i+1][j+1] = s

    # get pairs 
    pairs = []
    (i,j) = (n,m)
    while i > 0 and j > 0:
        if S[i,j-1] != S[i,j] and S[i-1,j] != S[i,j]:
            pairs.append((i,j))
            i -= 1 
            j -= 1 
        elif S[i,j-1] == S[i,j]:
            j -= 1 
        elif S[i-1,j] == S[i,j]:
            i -= 1 

    pairs.sort()
    pairs = {i-1:j-1 for i,j in pairs}

    final_seq = []
    valid_tags = []
    if task.startswith('ner'):
        valid_tags = ['PER', 'LOC', 'ORG', 'DATE']
        valid_tags = [f'B-{t}' for t in valid_tags] + [f'I-{t}' for t in valid_tags] 
        valid_tags.append('O')
    elif task.startswith('pos'):
        valid_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    else:
        raise ValueError(f'Unknown task type {task}')
    valid_tags = set(valid_tags)

    for i, tok in enumerate(gold_seq):
        if i in pairs:
            tag = parsed_seq[pairs[i]][1]
            if tag in valid_tags:
                final_seq.append((tok, tag))
            else:
                final_seq.append((tok, default_lbl))
        else:
            final_seq.append((tok, default_lbl))

    return final_seq 

def parse_gold(gold_str):
    gold_tuples = gold_str.strip()[2:-2].split('), (')
    gold_parsed = []
    for g in gold_tuples:
        m = re.match(r'``(.*)", ``([A-Z_-]*)"', g)
        if not m:
            print(g)
            raise Exception("Gold string should follow format")
        gold_parsed.append(m.group(1))

    return gold_parsed

def parse_pred(pred_str):
    pred_str = pred_str.split('\n')[0].strip()

    pred_str += ')]'
    pred_match = re.match(r'\[(.*)\].*', pred_str)

    if not pred_match:
        #print(f'Error: could not obtain match for prediction {pred_str}')
        return []

    pred_tuples = re.split(r'[)\]], [(\[]', pred_match.group(1)[1:-1])

    pred_parsed = []
    for p in pred_tuples:
        pred_match = re.match(r'(``|"|“)(.*)("|”), (``|"|“)([A-Z_-]*)("|”)?', p)
        if pred_match:
            pred_parsed.append((pred_match.group(2), pred_match.group(5)))
        else:
            pass
            #print(f'Error: could not match prediction {p}')

    return pred_parsed
