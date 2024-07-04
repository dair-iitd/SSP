from dataset import JSONLDataset, TabularDataset, PickleDataset
#import models.openai as openai
#from util import parse_example, parse_tsv_example, score_sets
import numpy as np

# from dotenv import load_dotenv

# from prompt import PromptGenerator

import argparse
from tqdm.auto import tqdm
import os
import json
import shutil
#import logging
from datetime import datetime
import signal
import sys, pdb

#logger = logging.getLogger('main')

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
    parser.add_argument('-r', '--result-file', type=str, default=f"{datetime.now().strftime('%Y%m%dT%H%M%S')}")

    parser.add_argument('-ssim', '--source-sim', type=str, help="Source similarity matrix")
    parser.add_argument('-tsim', '--target-sim', type=str, help="Target similarity matrix")
    

    parser.add_argument('-s', '--split-start',   type=int, default=0)
    parser.add_argument('-e', '--split-end',     type=int, default=100000)
    parser.add_argument('-i', '--interm',        type=int, default=10)
    parser.add_argument('-t', '--temperature',   type=float, default=0.5)
    
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

def construct_prompt(idx, tgt_ds, src_ds, tgt_sim_mat):

    #ind_tgt = tgt_sim_mat[idx]
    #pdb.set_trace()
    try:
        preds_ = [word.split("_")[1] for word in tgt_ds[idx]['output'].strip().split(" ")]
        golds_ = [word.split("_")[1] for word in src_ds[idx]['output'].strip().split(" ")]
    except:
        pdb.set_trace()
    try:
        assert(len(preds_) == len(golds_))
    except:
        pdb.set_trace()
    #pdb.set_trace()
    #preds_golds = [pred_+"\t"+gold_ for pred_, gold_ in zip(tgt_ds[idx].copy()['pred_labels'], src_ds[idx].copy()['pred_labels'])]
    
    return preds_, golds_

def get_response_from_gpt(example, task, prompt, model):
    # confidence scores via sampling multiple times...
    # not now.
    completion = model.complete(prompt)

    if completion is None:
        logger.error(f"Did not obtain response for input {example['input']}, setting everything to O")
        model.cleanup()
        return {
            'gold_labels': [a.split('_') for a in example['output'].strip().split(' ')],
            'pred_labels': [(a, 'O') for a in example['input'].strip().split(' ')]
        }

    logger.info(f'Obtained completion: {completion}')
    response = parse_tsv_example(task, example, completion)

    model.cleanup()
    return response

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

    #load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    #openai.setup_api_key(os.environ.get('OPENAI_API_KEY'))

    #save_dir = create_save_dir(args.result_dir, args.yes)

    #setup_logger(save_dir)

    # pg = PromptGenerator('prompts')
    # model_args = openai.ChatGPT.DEFAULT_ARGS
    # model_args['engine'] = args.model
    # model_args['request_timeout'] = 100
    # model = openai.ChatGPT(model_args)

    # #ssim = np.load(args.source_sim)
    # #tsim = None
    # #if args.target_sim:
    tsim = np.load(args.target_sim)

    #model.default_args['temperature'] = args.temperature

    # if args.dataset.endswith('.pkl'):
    #     ds = PickleDataset(args.dataset)[args.split_start:args.split_end]
    # elif args.dataset.endswith('.tsv'):
    #     ds = TabularDataset(args.dataset, delimiter='\t')[args.split_start:args.split_end]
    # else:
    #     pdb.set_trace()

    if args.source_dataset.endswith('.json'):
        #pdb.set_trace()
        sds = JSONLDataset(args.source_dataset)
    elif args.source_dataset.endswith('.tsv'):
        sds = TabularDataset(args.source_dataset, delimiter='\t')
    else:
        pdb.set_trace()

    tds = None
    if args.target_dataset.endswith('.json'):
        pdb.set_trace()
        tds = JSONLDataset(args.target_dataset)
    elif args.target_dataset.endswith('.tsv'):
        tds = TabularDataset(args.target_dataset, delimiter='\t')
    else:
        pdb.set_trace()

    #pdb.set_trace()
    if tsim.shape == (100,100):
        #pdb.set_trace()
        selections = []
        for i in range(tsim.shape[0]):
            selections += np.argsort(tsim[i])[::-1][1:][:8].tolist()
            assert i not in np.argsort(tsim[i])[::-1][1:][:8].tolist()
            #pdb.set_trace()

    else:
        selections = [j for it in tsim.tolist() for j in it]
    preds = []
    gold = []
    #print(selections)
    #pdb.set_trace()
    for idx in selections:
        pred_, gold_ = construct_prompt(idx, tds, sds, tsim)
        preds.append(pred_)
        gold.append(gold_)

    from seqeval.metrics import f1_score, classification_report, precision_score, recall_score
    try:
        f1_ = f1_score(gold, preds, average="micro")
        prec_ = precision_score(gold, preds)
        rec_ = recall_score(gold, preds)
        print(classification_report(gold, preds))
    except:
        pdb.set_trace()
        try:
            f1_ = f1_score(gold, preds[:len(gold)], average="micro")
            prec_ = precision_score(gold, preds[:len(gold)])
            rec_ = recall_score(gold, preds[:len(gold)])
            print(classification_report(gold, preds[:len(gold)], mode='strict'))
        except:
            f1_ = f1_score(gold[:len(preds)], preds, average="micro")
            prec_ = precision_score(gold[:len(preds)], preds)
            rec_ = recall_score(gold[:len(preds)], preds)
            print(classification_report(gold[:len(preds)], preds, mode='strict'))
        #pdb.set_trace()
    #f1_ = f1_score(gold, preds)
    print("Precision - ", prec_)
    print("Recall - ", rec_)
    print("F1 - ", f1_)


    


if __name__ == "__main__":
    main()
