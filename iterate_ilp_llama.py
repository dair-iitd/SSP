from dataset import JSONLDataset, TabularDataset, PickleDataset
#import models.openai as openai
import models.together as together
from util import parse_example, parse_tsv_example, score_sets
import numpy as np
import time

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
import sys
import pdb
logger = logging.getLogger('main')

def parse_args():

    parser = argparse.ArgumentParser(
            prog='promptbench',
            description='Prompt benchmarking utility'
        )

    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-p', '--prompt', type=str, default='ner')
    parser.add_argument('-td', '--target-dataset', type=str)
    parser.add_argument('-sd', '--source-dataset', type=str)
    parser.add_argument('-l', '--llama-url', type=str, help="LLaMa API URL")
    parser.add_argument('-m', '--model', type=str, help="model", default='gpt-3.5-turbo')
    parser.add_argument('-tr', '--target-retrieve', type=int, help="no. examples to retrieve from target", default=0)
    parser.add_argument('-sr', '--source-retrieve', type=int, help="no. examples to retrieve from source", default=8)
    parser.add_argument('-y', '--yes', action="store_true", help="Say yes to any conditionals")
    parser.add_argument('-r', '--result-dir', type=str, default=f"results/run_{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    parser.add_argument('-rv', '--reverse', action="store_true", help="Reverse a prompt or Not?")
    parser.add_argument('-ssim', '--source-sim', type=str, help="Source similarity matrix")
    parser.add_argument('-tsim', '--target-sim', type=str, help="Target similarity matrix")
    parser.add_argument('-glabel', '--gold_anno', default=False, action="store_true")

    parser.add_argument('-s', '--split-start',   type=int, default=0)
    parser.add_argument('-e', '--split-end',     type=int, default=100000)
    parser.add_argument('-i', '--interm',        type=int, default=10)
    parser.add_argument('-t', '--temperature',   type=float, default=0.0)
    
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
                     prompt, n_from_tgt=0, n_from_src=8, gold_anno=False, rev=False):

    # retrieve demos
    label_dict = {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe", "NULL": "NULL"}
    demos = []
    if n_from_src > 0:
        pdb.set_trace()
        ind = src_sim_mat[idx].argsort()[-n_from_src:]  #Aniruddha
        #ind = np.flip(ind) #Reveser order (Vipul)
        #pdb.set_trace()
        demos += [src_ds[i].copy() for i in ind]

        for d in demos:
            if "p2" not in prompt:
                d['output'] = label_dict[d['output']]

    # will include itself, we don't want that
    if n_from_tgt > 0:
        #pdb.set_trace()
        #ind_tgt = np.argpartition(tgt_sim_mat[idx], -n_from_tgt-1)[-n_from_tgt-1:-1]  #Aniruddha
        ind_tgt = tgt_sim_mat[idx].tolist()  #Decreasing order
        if len(ind_tgt) > n_from_tgt and len(ind_tgt) == 99:
            ind_tgt = np.argsort(tgt_sim_mat[idx])[::-1][1:1+n_from_tgt].tolist()
        assert len(ind_tgt) == n_from_tgt
        #pdb.set_trace()
        # if rev:
        #     pdb.set_trace()
        #     ind_tgt = ind_tgt[::-1]
        tgt_demos = [src_ds[i].copy() for i in ind_tgt if i != -1]
        tgt_labels = [tgt_ds[i] for i in ind_tgt if i != -1]
        #pdb.set_trace()
        # if 'output' not in tgt_demos[0]:
        #     # convert silver tags to gold tag format
        #     for d in tgt_demos:
        #         d['output'] = d['pred_labels']

        demos += tgt_demos
        #pdb.set_trace()
        if not(gold_anno):
            for it, d in enumerate(demos):
                if "p2" not in prompt:
                    d['output'] = label_dict[tgt_labels[it]]
                else:
                    d['output'] = tgt_labels[it]
        else:
            pdb.set_trace()
            for d in demos:
                if "p2" not in prompt:
                    d['output'] = label_dict[d['output']]
    #pdb.set_trace()
    
    #pdb.set_trace()
    if ind_tgt.count(-1) == 1:
        pdb.set_trace()
        prompt = "nli_8s_tsv_adapt_p2_7ex"
    elif ind_tgt.count(-1) == 2:
        pdb.set_trace()
        prompt = "nli_8s_tsv_adapt_p2_6ex"
    elif ind_tgt.count(-1) == 3:
        pdb.set_trace()
        prompt = "nli_8s_tsv_adapt_p2_5ex"
    elif ind_tgt.count(-1) == 4:
        pdb.set_trace()
        prompt = "nli_8s_tsv_adapt_p2_4ex"
    
    #pdb.set_trace()  
    prompt = pg.create_prompt(f'{prompt}', demos=demos, sentence=example)
    return prompt

def get_response_from_gpt(example, task, prompt, model):
    # confidence scores via sampling multiple times...
    # not now.
    if "p2" in task:
        label_dict_rev = {"entailment": "entailment", "contradiction": "contradiction", "neutral": "neutral"}
    else:
        label_dict_rev = {"Yes": "entailment", "No": "contradiction", "Maybe": "neutral"}
    completion = model.complete(prompt)

    if completion is None or completion == "":
        logger.error(f"Did not obtain response for input {example['premise']}, setting everything to O")
        model.cleanup()
        return "NULL", completion

    logger.info(f'Obtained completion: {completion}')
    #pdb.set_trace()
    completion = completion.strip().split("\n")[0]
    if completion in label_dict_rev:
        response = label_dict_rev[completion]
    else:
        response = completion
    model.cleanup()
    return response.split("Answer:")[-1].strip(), completion

def save_data(data, save_dir):
    with open(os.path.join(save_dir, f'responses.json'), 'w+') as outfile:
        for response, gold in zip(data['responses'], data["gold"]):
            outfile.write(response+"\t"+gold+"\n")

    with open(os.path.join(save_dir, f'accuracies.csv'), 'w+') as accfile:
        accfile.write(f"acc,total\n")
        accfile.write(f"{data['acc']},{data['non_null']}\n")

def main():

    args = parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    together.setup_api_key()

    save_dir = create_save_dir(args.result_dir, args.yes)

    setup_logger(save_dir)

    pg = PromptGenerator('prompts')
    model_args = together.CompletionModel.DEFAULT_ARGS
    model_args['model'] = args.model
    model_args['max_tokens'] = 20
    # change model to llama to use llama
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
            #pdb.set_trace()
            tds = json.load(open(args.target_dataset))
        elif args.target_dataset.endswith('.tsv'):
            tds = TabularDataset(args.target_dataset, delimiter='\t')
        else:
            logger.error('Dataset type not recognized. Continuing.')
            exit()

    interm = args.interm
    data = {
        'total': 0,
        "input": [],
        'gold': [],
        'responses': []
    }

    data_kv_store = {}
    skip_ind = []
    bar = tqdm(ds)

    for i, example in enumerate(bar):
        if interm==0:
            score_sets(data)
            save_data(data, save_dir)
            interm=args.interm
            bar.set_postfix(#prec=f"{data['precision']*100:.2f}", 
                            #recall=f"{data['recall']*100:.2f}", 
                            #f1=f"{data['f1']*100:.2f}",
                            acc=f"{data['acc']*100:.2f}")

        prompt = construct_prompt(i, example, tds, sds, tsim, ssim, pg, args.prompt,
                                  n_from_tgt=args.target_retrieve, n_from_src=args.source_retrieve, gold_anno=args.gold_anno, rev=args.reverse)
        #pdb.set_trace()
        if i < 5:
            print(prompt)
        response, completion = get_response_from_gpt(example, args.prompt, prompt, model)
        # sleep for 20s because or takes a lot of tokens
        #time.sleep(20)
        #pdb.set_trace()
        if completion != ""  and completion is not None:
            data['responses'].append(response)
            data["input"].append(prompt)
            data['gold'].append(example["output"])
            data['total'] += 1
            # put response in K-V store
            data_kv_store[example['premise']+"####"+example["hypothesis"]] = [response]
        else:
            skip_ind.append(i)

        interm-=1

        # if i == 5:
        #     break

    #pdb.set_trace()
    score_sets(data)
    save_data(data, save_dir)
    bar.set_postfix(acc=f"{data['acc']*100:.2f}")
    print(f"{data['total']} examples run")
    with open(save_dir+"/skip_ind.json", "w") as f_w:
        json.dump(skip_ind, f_w)

if __name__ == "__main__":
    main()
