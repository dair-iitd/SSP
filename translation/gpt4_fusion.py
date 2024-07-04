from dataset import JSONLDataset, TabularDataset, PickleDataset
import models.openai as openai
from util import parse_example, parse_tsv_example, score_sets
import numpy as np

from dotenv import load_dotenv

from prompt import PromptGenerator

import argparse
from tqdm.auto import tqdm
import os
import json
import shutil
import logging
from datetime import datetime
import signal
import sys, pdb
import time
logger = logging.getLogger('main')

running = True

# prompt = """
# You are working as a named entity recognition expert and your task is to label a given text with named entity labels. Your task is to identify and label any named entities present in the text. Specifically, you will be given an English sentence that has already been tagged, and you will predict on a translation of that sentence in {}. The named entity labels that you will be using are PER (person), LOC (location), and ORG (organization). You may encounter multi-word entities, so make sure to label 
# each word of the entity with the appropriate prefix (“B” for the first word of the entity, “I” for any non-initial word of the entity). For words which are not part of any named entity, you should return “O”. Note: Your output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding named entity label.

# English input:
# Manchester  B-ORG
# City    I-ORG
# should  O
# have    O
# saved   O
# one O
# point   O
# to  O
# be  O
# among   O
# the O
# winners.    O

# Sentence:
# Manchester City waroon naa denc benn poñ ngir bokk ci ñi raw .

# Output:
# ```
# Manchester  B-ORG
# City    I-ORG
# waroon  O
# naa O
# denc    O
# benn    O
# poñ O
# ngir    O
# bokk    O
# ci  O
# ñi  O
# raw O
# .   O
# ```

# English input:
# {}

# Sentence:
# {}

# Output:
# ```
# """

prompt = """
You are working as a Part-of-Speech (POS) tagging expert and your task is to label a given text with POS labels. Your task is to label all words in a sentence with their part of speech tags. Specifically, you will be given an English sentence that has already been tagged, and you will predict on a translation of that sentence in {}. The POS labels that you will be using are ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X. Note: Your output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding POS tag.

English input:
Pope NOUN
is VERB
the highest ADJ
authority NOUN
for ADP
the Roman NOUN
Church NOUN
. PUNCT

Sentence:
Pávin er hægsti myndugleiki fyri róma- kirkjuni .

Output:
```
Pávin NOUN                                                                      
er VERB                                                                         
hægsti ADJ                                                                      
myndugleiki NOUN                                                                
fyri ADP                                                                        
róma- NOUN                                                                      
kirkjuni NOUN                                                                   
. PUNCT
```

English input:
{}

Sentence:
{}

Output:
```
"""

def signal_handler(sig, frame):
    print('Exiting...')
    global running
    running = False

def parse_args():

    parser = argparse.ArgumentParser(
            prog='promptbench',
            description='Prompt benchmarking utility'
        )

    parser.add_argument('-l', '--lang', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-p', '--prompt', type=str, default='ner')
    parser.add_argument('-td', '--target-dataset', type=str)
    parser.add_argument('-sd', '--source-dataset', type=str)
    #parser.add_argument('-l', '--llama-url', type=str, help="LLaMa API URL")
    parser.add_argument('-m', '--model', type=str, help="model", default='gpt-3.5-turbo')
    parser.add_argument('-tr', '--target-retrieve', type=int, help="no. examples to retrieve from target", default=0)
    parser.add_argument('-sr', '--source-retrieve', type=int, help="no. examples to retrieve from source", default=8)
    parser.add_argument('-y', '--yes', action="store_true", help="Say yes to any conditionals")
    parser.add_argument('-r', '--result-dir', type=str, default=f"results/run_{datetime.now().strftime('%Y%m%dT%H%M%S')}")

    parser.add_argument('-ssim', '--source-sim', type=str, help="Source similarity matrix")
    parser.add_argument('-tsim', '--target-sim', type=str, help="Target similarity matrix")
    

    parser.add_argument('-s', '--split-start',   type=int, default=0)
    parser.add_argument('-e', '--split-end',     type=int, default=100000)
    parser.add_argument('-i', '--interm',        type=int, default=10)
    parser.add_argument('-t', '--temperature',   type=float, default=0.0)
    parser.add_argument('--slow', action="store_true", help="slow down API calls")
    
    return parser.parse_args()

def create_save_dir(save_dir, overwrite):
    if os.path.exists(save_dir):
        if overwrite:
            print('Output folder already exists, overwriting')
            shutil.rmtree(save_dir)
        else:
            print('Overwrite preexisting output folder? (y/N): ', end='')
            ch = input()
            if (ch == 'y'):
                shutil.rmtree(save_dir)
            else:
                save_dir += '_1'

    os.makedirs(save_dir)
    return save_dir

def setup_logger(save_dir):

    logging.basicConfig(
            filename=os.path.join(save_dir, 'logfile.log'),
            filemode='a',
            format='[%(asctime)s.%(msecs)d](%(name)s:%(levelname)s) %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )

def gold_tags_to_tsv_output(sentence):
    temp = sentence.strip().split(' ')
    temp = [w.rsplit('_', 1) for w in temp]
    return '\n'.join([f'{x[0]}\t{x[1]}' for x in temp])

def sentence_to_input(sentence):
    temp = sentence.split(' ')
    return "[" + ", ".join([f'"{a}"' for a in temp]) + "]"

def gold_tags_to_output(sentence):
    temp = [a.rsplit('_', 1) for a in sentence.strip().split(' ')]
    return "[" + ", ".join([f'(``{a[0]}", ``{a[1]}")' for a in temp]) + "]"

def construct_prompts(translated_conll, source, lang):
    source_egs = [l.strip() for l in open(source, 'r').readlines()]
    translated_lines = [l.strip() for l in open(translated_conll, 'r').readlines()]
    translated_egs = []
    accum = []
    for line in translated_lines:
        if line == '':
            translated_egs.append('\n'.join(accum))
            accum = []
        else:
            accum.append(line.replace(' ', '\t'))
    translated_egs.append('\n'.join(accum))
    accum = []

    return [prompt.format(lang, eg, src) for (eg, src) in zip(translated_egs, source_egs)]

def run_prompt(prompt, model):
    completion = model.complete(prompt)
    model.cleanup()
    return completion

def construct_prompt(idx, example, tgt_ds, src_ds, tgt_sim_mat, src_sim_mat, pg, 
                     prompt, n_from_tgt=0, n_from_src=8):

    # retrieve demos
    demos = []
    if n_from_src > 0:
        pdb.set_trace()
        ind = np.argpartition(src_sim_mat[idx], -n_from_src)[-n_from_src:]
        demos += [src_ds[i].copy() for i in ind]
        #pdb.set_trace()

    # will include itself, we don't want that
    #pdb.set_trace()
    if n_from_tgt > 0:
        ind_tgt = tgt_sim_mat[idx].tolist()
        if len(ind_tgt) > n_from_tgt and len(ind_tgt) == 100:
            ind_tgt = np.argsort(tgt_sim_mat[idx])[::-1][1:1+n_from_tgt].tolist()
        assert len(ind_tgt) == n_from_tgt
        #pdb.set_trace()
        tgt_demos = [tgt_ds[i].copy() for i in ind_tgt]
        #assert len(tgt_demos) == n_from_tgt  and idx not in ind_tgt
        
        if 'output' not in tgt_demos[0]:
            pdb.set_trace()
            # convert silver tags to gold tag format
            for d in tgt_demos:
                d['output'] = ' '.join([f'{a}_{b}' for a,b in zip(d['input'].strip().split(' '), d['pred_labels'])])

        demos += tgt_demos

    examples = [d['output'] for d in demos]
    for d in demos:
        d['output'] = gold_tags_to_tsv_output(d['output'])

    prompt = pg.create_prompt(f'{prompt}', demos=demos, sentence=example['input'])
    #pdb.set_trace()
    return (prompt, examples)

def get_response_from_gpt(example, task, prompt, model):
    # confidence scores via sampling multiple times...
    # not now.
    completion = model.complete(prompt)
    #pdb.set_trace()
    if completion is None or completion == "":
        logger.error(f"Did not obtain response for input {example['input']}, setting everything to O")
        model.cleanup()
        return {
            'gold_labels': [a.split('_') for a in example['output'].strip().split(' ')],
            'pred_labels': [(a, 'O') for a in example['input'].strip().split(' ')]
        }, completion

    logger.info(f'Obtained completion: {completion}')
    #pdb.set_trace()
    response = parse_tsv_example(task, example, completion)

    model.cleanup()
    return response, completion

def save_data(data, save_dir):
    with open(os.path.join(save_dir, f'responses.json'), 'w+') as outfile:
        for response in data['responses']:
            outfile.write(f"{json.dumps(response)}\n")

    with open(os.path.join(save_dir, f'accuracies.csv'), 'w+') as accfile:
        accfile.write(f"precision,recall,f1,total\n")
        accfile.write(f"{data['precision']},{data['recall']},{data['f1']},{data['total']}\n")

def main():

    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    openai.setup_api_key(os.environ.get('OPENAI_API_KEY'))
    args.yes=True
    

    

    pg = PromptGenerator('prompts')
    model_args = openai.ChatGPT.DEFAULT_ARGS
    model_args['model'] = 'gpt-4-turbo' # gpt-35-turbo for ChatGPT
    model_args['timeout'] = 100
    model_args['temperature'] = 0.0
    model = openai.ChatGPT(model_args)

    # ssim = np.load(args.source_sim)
    # #tsim = None
    # #if args.target_sim:
    # tsim = np.load(args.target_sim)

    #model.default_args['temperature'] = 0.0

    #model = create_model()
    sys_1 = args.lang
    sys_2 = "../Codec/tagged/{}.txt".format(sys_1)
    sys_3 = "data/new/{}/test_trans.txt".format(sys_1)
    sys_4 = "data/new/{}/test.tsv".format(sys_1)
    sys_5 = "new_results/codec/{}".format(sys_1)
    save_dir = create_save_dir(sys_5, True)
    setup_logger(save_dir)

    prompts = construct_prompts(sys_2, sys_3, sys_1)
    sds = TabularDataset(sys_4, delimiter='\t')
    results = []

    # if args.dataset.endswith('.pkl'):
    #     ds = PickleDataset(args.dataset)[args.split_start:args.split_end]
    # elif args.dataset.endswith('.tsv'):
    #     ds = TabularDataset(args.dataset, delimiter='\t')[args.split_start:args.split_end]
    # else:
    #     logger.error('Dataset type not recognized. Continuing.')
    #     exit()

    # sds=None
    # #sds = JSONLDataset(args.source_dataset)
    # tds = None
    # if args.target_dataset.endswith('.json'):
    #     pdb.set_trace()
    #     tds = JSONLDataset(args.target_dataset)
    # elif args.target_dataset.endswith('.tsv'):
    #     tds = TabularDataset(args.target_dataset, delimiter='\t')
    # else:
    #     logger.error('Dataset type not recognized. Continuing.')
    #     exit()

    interm = args.interm
    data = {
        'total': 0,
        'responses': []
    }

    data_kv_store = {}

    #bar = tqdm(ds)
    skip_ind = []
    for i, prompt in enumerate(prompts):
        # if i == 10:
        #     break
        example = sds.examples[i]
        if not running:
            break
        if interm==0:
            score_sets(data)
            save_data(data, save_dir)
            interm=args.interm
            # bar.set_postfix(prec=f"{data['precision']*100:.2f}", 
            #                 recall=f"{data['recall']*100:.2f}", 
            #                 f1=f"{data['f1']*100:.2f}")
        
        result = run_prompt(prompt, model)
        if i < 5:
            print(prompt)
        if result == "":
            continue

        response = parse_tsv_example('ner', example, result)
        preds = [it.split()[-1] for it in result.strip().split("\n")][1:-1]
        try:
            assert len(preds) == len(response['pred_labels'])
            response['pred_labels'] = preds
            print(preds)
        except:
            print("mismatch")
        #pdb.set_trace()
        #if args.slow:
        time.sleep(5)
        if response != ""  and response is not None:
            data['responses'].append({
                **example,
                **response
            })
            data['total'] += 1
            data_kv_store[example['input']] = [response]
        else:
            skip_ind.append(i)

        #data['total'] += 1
        # put response in K-V store
        

        interm-=1

    score_sets(data)
    save_data(data, save_dir)
    # bar.set_postfix(prec=f"{data['precision']*100:.2f}", 
    #                 recall=f"{data['recall']*100:.2f}", 
    #                 f1=f"{data['f1']*100:.2f}")
    print(f"{data['total']} examples run")
    with open(save_dir+"/"+args.lang+"_skip_ind.json", "w") as f_w:
        json.dump(skip_ind, f_w)


if __name__ == "__main__":
    main()
