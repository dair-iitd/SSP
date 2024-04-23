from dataset import JSONLDataset, TabularDataset, PickleDataset
import models.openai as openai
#from util import parse_example, parse_tsv_example, parse_qaner_example, score_sets
import numpy as np
import time
import re

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
import time

def run_prompt(prompt, model):
    completion = model.complete(prompt)
    time.sleep(10)
    if completion is None or completion == '':
        return "Could not complete prompt"
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
                print(f'[{ptok}->{gtok}]', end=' ')
        print()
    
def create_model():
    load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    openai.setup_api_key(os.environ.get('OPENAI_API_KEY'))

    model_args = openai.ChatGPT.DEFAULT_ARGS
    model_args['engine'] = "gpt-4-turbo"
    model_args['request_timeout'] = 200
    # change model to llama to use llama
    model = openai.ChatGPT(model_args)
    model.default_args['temperature'] = 0
    return openai.ChatGPT(model_args)

def parse_response(response):
    lines = response.strip().split('\n')
    pred = [a.split('\t') for a in lines if '\t' in a]
    return list(zip(*pred))


model = create_model()

def print_results(prompt, model):
    print(run_prompt(prompt, model))

inp_prompt = open("single_prompt.txt").read()
print_results(inp_prompt, model)

