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
        assert len(tgt_demos) == n_from_tgt  and idx not in ind_tgt
        
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

    if completion is None or completion == "":
        logger.error(f"Did not obtain response for input {example['input']}, setting everything to O")
        model.cleanup()
        return {
            'gold_labels': [a.split('_') for a in example['output'].strip().split(' ')],
            'pred_labels': [(a, 'O') for a in example['input'].strip().split(' ')]
        }, completion

    logger.info(f'Obtained completion: {completion}')
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

    load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    openai.setup_api_key(os.environ.get('OPENAI_API_KEY'))

    save_dir = create_save_dir(args.result_dir, args.yes)

    setup_logger(save_dir)

    pg = PromptGenerator('prompts')
    model_args = openai.ChatGPT.DEFAULT_ARGS
    model_args['engine'] = args.model
    model_args['request_timeout'] = 100
    model = openai.ChatGPT(model_args)

    ssim = np.load(args.source_sim)
    #tsim = None
    #if args.target_sim:
    tsim = np.load(args.target_sim)

    model.default_args['temperature'] = args.temperature

    if args.dataset.endswith('.pkl'):
        ds = PickleDataset(args.dataset)[args.split_start:args.split_end]
    elif args.dataset.endswith('.tsv'):
        ds = TabularDataset(args.dataset, delimiter='\t')[args.split_start:args.split_end]
    else:
        logger.error('Dataset type not recognized. Continuing.')
        exit()

    sds = JSONLDataset(args.source_dataset)
    tds = None
    if args.target_dataset.endswith('.json'):
        pdb.set_trace()
        tds = JSONLDataset(args.target_dataset)
    elif args.target_dataset.endswith('.tsv'):
        tds = TabularDataset(args.target_dataset, delimiter='\t')
    else:
        logger.error('Dataset type not recognized. Continuing.')
        exit()

    interm = args.interm
    data = {
        'total': 0,
        'responses': []
    }

    data_kv_store = {}

    bar = tqdm(ds)
    skip_ind = []
    for i, example in enumerate(bar):
        if not running:
            break
        if interm==0:
            score_sets(data)
            save_data(data, save_dir)
            interm=args.interm
            bar.set_postfix(prec=f"{data['precision']*100:.2f}", 
                            recall=f"{data['recall']*100:.2f}", 
                            f1=f"{data['f1']*100:.2f}")

        (prompt, examples) = construct_prompt(i, example, tds, sds, tsim, ssim, pg, args.prompt,
                                  n_from_tgt=args.target_retrieve, n_from_src=args.source_retrieve)
        response, completion = get_response_from_gpt(example, args.prompt, prompt, model)
        #if args.slow:
        time.sleep(20)
        if completion != ""  and completion is not None:
            data['responses'].append({
                **example,
                **response,
                'examples': examples
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
    bar.set_postfix(prec=f"{data['precision']*100:.2f}", 
                    recall=f"{data['recall']*100:.2f}", 
                    f1=f"{data['f1']*100:.2f}")
    print(f"{data['total']} examples run")
    with open(save_dir+"/"+args.lang+"_skip_ind.json", "w") as f_w:
        json.dump(skip_ind, f_w)


if __name__ == "__main__":
    main()
