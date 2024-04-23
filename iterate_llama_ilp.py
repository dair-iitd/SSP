from dataset import JSONLDataset, TabularDataset, PickleDataset
import models.together as together
from util import parse_example, parse_tsv_example, parse_qaner_example, score_sets
import numpy as np
import time

from dotenv import load_dotenv

from prompt import PromptGenerator

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

logger = logging.getLogger('main')

running = True
randomize_labels = False

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
    parser.add_argument('-m', '--model', type=str, help="model", default='meta-llama/llama-2-70b-hf')
    parser.add_argument('-tr', '--target-retrieve', type=int, help="no. examples to retrieve from target", default=0)
    parser.add_argument('-sr', '--source-retrieve', type=int, help="no. examples to retrieve from source", default=8)
    parser.add_argument('-y', '--yes', action="store_true", help="Say yes to any conditionals")
    parser.add_argument('-r', '--result-dir', type=str, default=f"results/run_{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    parser.add_argument('-rl', '--randomize-labels', action="store_true", help="randomize labels (for ablation)")
    parser.add_argument('--slow', action="store_true", help="slow down API calls")
    parser.add_argument('-cf', '--content-filter', action="store_true", help="ignore content filter (save all egs)")

    parser.add_argument('-ssim', '--source-sim', type=str, help="Source similarity matrix")
    parser.add_argument('-tsim', '--target-sim', type=str, help="Target similarity matrix")
    

    parser.add_argument('-s', '--split-start',   type=int, default=0)
    parser.add_argument('-e', '--split-end',     type=int, default=100000)
    parser.add_argument('-i', '--interm',        type=int, default=10)
    parser.add_argument('-t', '--temperature',   type=float, default=0) # was 0.5 earlier!
    
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

def random_ner_label():
    # labels = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-DATE', 'I-DATE']
    labels = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    return random.choice(labels)

def gold_tags_to_tsv_output(sentence):
    temp = sentence.strip().split(' ')
    temp = [w.rsplit('_', 1) for w in temp]
    # randomize labels here for ablation
    if randomize_labels:
        temp = [(w[0], random_ner_label()) for w in temp]
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

def get_response_from_llama(example, task, prompt, model):
    completion = model.complete(prompt)

    if completion is None or completion == "":
        logger.error(f"Did not obtain response for input {example['input']}, setting everything to O")
        model.cleanup()
        default_lbl = 'O'
        if task.startswith('pos'):
            default_lbl = 'X'
        return {
            'gold_labels': [a.split('_')[1] for a in example['output'].strip().split(' ')],
            'pred_labels': [default_lbl for a in example['input'].strip().split(' ')]
        }, completion

    logger.info(f'Obtained completion: {completion}')
    response = parse_tsv_example(task, example, completion)

    model.cleanup()
    return response, completion

def save_data(data, skip_ind, save_dir):
    with open(os.path.join(save_dir, f'responses.json'), 'w+') as outfile:
        for response in data['responses']:
            outfile.write(f"{json.dumps(response, ensure_ascii=False)}\n")

    with open(os.path.join(save_dir, f'accuracies.csv'), 'w+') as accfile:
        accfile.write(f"precision,recall,f1,total\n")
        accfile.write(f"{data['precision']},{data['recall']},{data['f1']},{data['total']}\n")

    json.dump(skip_ind, open(os.path.join(save_dir, f'skip_idxs.json'), 'w'), ensure_ascii=False)

def main():

    args = parse_args()

    global randomize_labels
    randomize_labels = args.randomize_labels

    load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    together.setup_api_key()

    save_dir = create_save_dir(args.result_dir, args.yes)

    setup_logger(save_dir)

    logger.info("Running with args:")
    logger.info(args)

    pg = PromptGenerator('prompts')
    model_args = together.CompletionModel.DEFAULT_ARGS
    model_args['model'] = args.model
    # model_args['request_timeout'] = 200
    model = together.CompletionModel(model_args)

    ssim = np.load(args.source_sim)
    tsim = None
    if args.target_sim:
        tsim = np.load(args.target_sim)

    model.default_args['temperature'] = args.temperature

    if args.dataset.endswith('.pkl'):
        ds = PickleDataset(args.dataset)[args.split_start:args.split_end]
    elif args.dataset.endswith('.tsv'):
        ds = TabularDataset(args.dataset, delimiter='\t')[args.split_start:args.split_end]
    else:
        logger.error('Dataset type not recognized. Continuing.')
        exit()

    sds = TabularDataset(args.source_dataset, delimiter='\t')
    print(len(sds))
    tds = None
    if args.target_dataset:
        if args.target_dataset.endswith('.json'):
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
    # pdb.set_trace()
    bar = tqdm(ds)

    skip_ind = []

    for i, example in enumerate(bar):
        if not running:
            break
        if interm==0:
            score_sets(data)
            save_data(data, skip_ind, save_dir)
            interm=args.interm
            bar.set_postfix(prec=f"{data['precision']*100:.2f}", 
                            recall=f"{data['recall']*100:.2f}", 
                            f1=f"{data['f1']*100:.2f}")

        # pdb.set_trace()
        (prompt, examples) = construct_prompt(i, example, tds, sds, tsim, ssim, pg, args.prompt,
                                  n_from_tgt=args.target_retrieve, n_from_src=args.source_retrieve)
        # pdb.set_trace()
        response, completion = get_response_from_llama(example, args.prompt, prompt, model)
        # sleep for 30s because rate limit at api
        if args.slow:
            time.sleep(15)

        if args.content_filter:
            if completion != "":
                data['responses'].append({
                    **example,
                    **response,
                    'examples': examples
                })
                data['total'] += 1
            else:
                skip_ind.append(i)
        else:
            data['responses'].append({
                **example,
                **response,
                'examples': examples
            })
            data['total'] += 1

        # put response in K-V store
        data_kv_store[example['input']] = [response]

        interm-=1

    score_sets(data)
    save_data(data, skip_ind, save_dir)
    bar.set_postfix(prec=f"{data['precision']*100:.2f}", 
                    recall=f"{data['recall']*100:.2f}", 
                    f1=f"{data['f1']*100:.2f}")
    print(f"{data['total']} examples run")
    # with open(save_dir+"/"+args.lang+"_skip_ind.json", "w") as f_w:
    #     json.dump(skip_ind, f_w)

if __name__ == "__main__":
    main()
