from dataset import JSONLDataset, TabularDataset, PickleDataset
import json
import models.openai as openai
from util import parse_example, parse_tsv_example, parse_qaner_example, score_sets
import numpy as np
import time

from dotenv import load_dotenv

import argparse
import random
from tqdm.auto import tqdm
import os, pdb
import json
import shutil
import logging
from datetime import datetime
import signal
import sys, pdb

prompt = """
You are working as a named entity recognition expert and your task is to label a given text with named entity labels. Your task is to identify and label any named entities present in the text. Specifically, you will be given an English sentence that has already been tagged, and you will predict on a translation of that sentence in {}. The named entity labels that you will be using are PER (person), LOC (location), and ORG (organization). You may encounter multi-word entities, so make sure to label 
each word of the entity with the appropriate prefix (“B” for the first word of the entity, “I” for any non-initial word of the entity). For words which are not part of any named entity, you should return “O”. Note: Your output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding named entity label.

English input:
Manchester	B-ORG
City	I-ORG
should	O
have	O
saved	O
one	O
point	O
to	O
be	O
among	O
the	O
winners.	O

Sentence:
Manchester City waroon naa denc benn poñ ngir bokk ci ñi raw .

Output:
```
Manchester	B-ORG
City	I-ORG
waroon	O
naa	O
denc	O
benn	O
poñ	O
ngir	O
bokk	O
ci	O
ñi	O
raw	O
.	O
```

English input:
{}

Sentence:
{}

Output:
```
"""

def run_prompt(prompt, model):
    completion = model.complete(prompt)
    model.cleanup()
    return completion

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

def get_prompt_eg_acc(srcfile, eg_idxs):
    ds = JSONLDataset(srcfile)
    egs = []
    for idx in eg_idxs:
        egs.append(ds[idx])
    
    for eg in egs:
        for (ptok, gtok) in zip(eg['pred_labels'], eg['gold_labels']):
            if ptok == gtok:
                print(ptok, end=' ')
            else:
                print(f'[{ptok}]', end=' ')
        print()
    
def create_model():
    load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    openai.setup_api_key(os.environ.get('OPENAI_API_KEY'))

    model_args = openai.ChatGPT.DEFAULT_ARGS
    model_args['engine'] = 'gpt-4-turbo'
    model_args['request_timeout'] = 200
    model_args['temperature'] = 0.0
    return openai.ChatGPT(model_args)

# def parse_tsv_example(response):
#     lines = response.strip().split('\n')
#     pred = [a.split('\t') for a in lines if '\t' in a]
#     return list(zip(*pred))
    
def main():
    model = create_model()
    prompts = construct_prompts(sys.argv[2], sys.argv[3], sys.argv[1])
    sds = TabularDataset(sys.argv[4], delimiter='\t')
    results = []
    for i in tqdm(range(len(prompts))):
        prompt = prompts[i]
        example = sds.examples[i]
        #pdb.set_trace()
        result = run_prompt(prompt, model)
        if result == "":
            continue
        final_output = parse_tsv_example('ner', example, result)
        results.append(final_output)

    with open(sys.argv[5], 'w') as outfile:
        for result in results:
            outfile.write(json.dumps(result)+"\n")
    
if __name__ == '__main__':
    main()
