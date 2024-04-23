from dataset import JSONLDataset, TabularDataset, PickleDataset
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
import sys

def run_prompt(filename, model):
    with open(filename, 'r') as prompt_file:
        prompt = ''.join(prompt_file.readlines())
        completion = model.complete(prompt)
        if completion is None or completion == '':
            return ""
        return completion

def construct_prompt(srcfile, promptfile, egs):
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
    model_args['engine'] = 'gpt-4'
    model_args['request_timeout'] = 200
    return openai.ChatGPT(model_args)

def parse_tsv_example(response):
    lines = response.strip().split('\n')
    pred = [a.split('\t') for a in lines if '\t' in a]
    return list(zip(*pred))
    
def main():
    model = create_model()
    tags = parse_tsv_example(run_prompt(sys.argv[1], model))
    print(' '.join(tags[0]))
    print(' '.join(tags[1]))
    
if __name__ == '__main__':
    main()
