import re
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
import pdb

from itertools import groupby

def split_tokens_newline(all_tokens_probs):
    element = "\n"
    all_tup_list =  [list(group) for key, group in groupby(all_tokens_probs, key=lambda x: x[0] == element) if not key]
    #pdb.set_trace()
    all_list = []
    for it in all_tup_list:
        curr_len = len(it)
        curr_s_list = [t[0] for t in it]
        if "\t" not in curr_s_list:
            continue
        #pdb.set_trace()
        curr_prob_list = [t[1] for t in it]
        idx = curr_s_list.index("\t")
        curr_prob_list_valid = curr_prob_list[idx+1:curr_len]
        try:
            curr_prob = sum(curr_prob_list_valid)/len(curr_prob_list_valid)
        except:
            #pdb.set_trace()
            continue
        all_list.append(("".join(curr_s_list), curr_prob))
    return all_list


def get_token_probs(log_probs, output_tokens):
    ret_ind = [i+1 for i,it in enumerate(output_tokens) if it == "\t"]
    try:
        ret_probs = np.exp([log_probs[j] for j in ret_ind]).tolist()
    except:
        #pdb.set_trace()
        ret_probs = np.exp([log_probs[j] for j in ret_ind[:-1]]).tolist()+[0.0]
    #pdb.set_trace()
    return ret_probs



def parse_tsv_example(task, example, response, all_tokens_probs=None):
    #pdb.set_trace()
    lines = response.strip().split('\n')
    pred = [a.split('\t') for a in lines if '\t' in a]
    all_tokens_probs = False
    if all_tokens_probs:
        # pdb.set_trace()
        pred_probs = split_tokens_newline(all_tokens_probs)
        try:
            assert len(pred) == len(pred_probs)
            l1 = [it[0] for it in pred]
            l2 = [it[0].split("\t")[0] for it in pred_probs]
            assert l1 == l2
        except:
            pdb.set_trace()
        for pre, pro in zip(pred, pred_probs):
            pre.append(pro[1])
    else:
        for pre in pred:
            pre.append(-1.0)
    #pdb.set_trace()

    gold_parsed : list[tuple[str, str]] = [a.rsplit('_', 1) for a in example['output'].strip().split(' ')]
    pred_parsed, word_probs = align_dp(task, [a[0] for a in gold_parsed], pred)
    #pdb.set_trace()
    assert(len(pred_parsed) == len(gold_parsed))

    gold_labels = [g[1] for g in gold_parsed]
    pred_labels = [p[1] for p in pred_parsed]

    if all_tokens_probs:
        #label_probs = get_token_probs(log_probs, output_tokens)
        try:
            assert len(word_probs) == len(pred_labels)
        except:
            #pdb.set_trace()
            word_probs = word_probs[:len(pred_labels)]
        return {
            'pred_labels': pred_labels,
            'gold_labels': gold_labels
        }, word_probs
    else:
        word_probs = [-1.0]*len(pred_labels)
        return {
            'pred_labels': pred_labels,
            'gold_labels': gold_labels,
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

def parse_qaner_example(task, example, responses):
    
    gold_parsed : list[tuple[str, str]] = [a.rsplit('_', 1) for a in example['output'].strip().split(' ')]
    pred_parsed = [(a[0], 'O') for a in gold_parsed]
    toks = ['PER', 'ORG', 'LOC', 'DATE']
    for (label, response_list) in zip(toks, responses):
        if response_list.strip() == 'None':
            continue
        response_toks = [a.strip() for a in response_list.strip().split(';')]
        for response in response_toks:
            words = response.split(' ')
            i = 0
            while i < len(pred_parsed):
                j = 0
                while j < len(words) and i+j < len(pred_parsed) and pred_parsed[i+j][0] == words[j]:
                    if j == 0:
                        pred_parsed[i+j] = (pred_parsed[i+j][0], 'B-'+label)
                    else:
                        pred_parsed[i+j] = (pred_parsed[i+j][0], 'I-'+label)
                    j += 1
                i += (j+1)

    gold_labels = [g[1] for g in gold_parsed]
    pred_labels = [p[1] for p in pred_parsed]

    return {
        'pred_labels': pred_labels,
        'gold_labels': gold_labels
    }

def score_sets(data):

    try:
        gold_labels = [f['gold_labels'] for f in data['responses']]
        pred_labels = [f['pred_labels'] for f in data['responses']]
        precision = precision_score(gold_labels, pred_labels)
        recall = recall_score(gold_labels, pred_labels)
        f1 = f1_score(gold_labels, pred_labels)
    except Exception as e:
        print("Could not compute scores, please compute them manually later")
        precision = 0
        recall = 0
        f1 = 0

    data['precision'] = precision
    data['recall'] = recall
    data['f1'] = f1

def align_dp(task, gold_seq: list[str], parsed_seq: list[tuple[str,str,float]]) -> list[tuple[str, str]]:
    """
    Aligns the two sequences using dynamic programming
    """

    n = len(gold_seq)
    m = len(parsed_seq)
    G = gold_seq
    P = parsed_seq
    S = np.zeros((n+1, m+1))
    default_lbl = 'O'
    default_prob = 0.0

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
    final_probs = []
    valid_tags = []
    if task.startswith('ner'):
        valid_tags = ['PER', 'LOC', 'ORG', 'DATE']
        valid_tags = [f'B-{t}' for t in valid_tags] + [f'I-{t}' for t in valid_tags] 
        valid_tags.append('O')
    elif task.startswith('pos'):
        valid_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        default_lbl = 'X'
    else:
        raise ValueError(f'Unknown task type {task}')
    valid_tags = set(valid_tags)
    FLAG1 = 0
    FLAG2 = 0
    FLAG3 = 0
    for i, tok in enumerate(gold_seq):
        if i in pairs:
            tag = parsed_seq[pairs[i]][1]
            prob_ = parsed_seq[pairs[i]][2]
            if tag in valid_tags:
                FLAG3 = 1
                final_seq.append((tok, tag))
                final_probs.append(prob_)
            else:
                #print("Invalid tag")
                FLAG2 = 1
                final_seq.append((tok, default_lbl))
                final_probs.append(default_prob)
        else:
            #print("Not in pairs")
            FLAG1 = 1
            final_seq.append((tok, default_lbl))
            final_probs.append(default_prob)

    # if FLAG1 == 1:
    #     print("Not in pairs")
    # if FLAG2 == 1:
    #     print("Invalid tag")
    # if FLAG3 == 1:
    #     print("Valid tag")
    return final_seq, final_probs

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
