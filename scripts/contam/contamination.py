import os
import argparse
from typing import Dict, Any, Optional, Union, List, Tuple
import openai
import sys
import time
import random
import json
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import TabularDataset
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
import pdb
# TODO implement these for complete and generate_few_shot methods
# from mega.data.load_datasets import (
#     load_dataset_mega,
#     load_json_datasets,
#     get_dataset_splits,
# )

from dotenv import load_dotenv

from mega_parser import parse_args
import backoff

CHAT_MODELS = ["gpt-35-turbo", "gpt-4", "gpt-4-turbo"]

load_dotenv('../.env')

def load_openai_env_variables():
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15'

def get_dataset_splits(dataset):
    return ["test"]

def load_xnli_dataset(
    lang: str, split: str, dataset_frac: float = 1.0
) -> Union[Dataset, DatasetDict]:
    """
    Args:
        lang (str): Language for which xnli dataset is to be loaded
        split (str): Train test of validation split of the model to load
        dataset_frac (float): Fraction of examples to load. Defaults to 1.0

    Returns:
        Union[Dataset, DatasetDict]: huggingface dataset object
    """
    if lang in set(
        ["as", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "bn"]
    ):  ##PJ:To add except hindi
        dataset = load_dataset("Divyanshu/indicxnli", lang)[split]
    else:
        #dataset = load_dataset("nala-cub/americas_nli", lang)[split]
        dataset = load_dataset("xnli", lang)[split]
    N = len(dataset)
    selector = np.arange(int(N * dataset_frac))
    return dataset.select(selector)

def load_dataset_mega(dataset, lang, split='test'):
    assert dataset in ['masakhaner', 'udpos', 'americasnli', 'xnli']
    if dataset == 'masakhaner' or dataset == 'udpos':
        assert lang in ['fo', 'got', 'gsw', 'hau', 'ibo', 'kin', 'lug', 'luo']
        ret_dat = TabularDataset(f'../iterate/data/{lang}_100/test_{lang}_100.tsv', delimiter='\t')
    elif 'americas' in dataset:
        assert lang in ['aym', 'quy', 'nah', 'gn']
        ret_dat = TabularDataset(f'../AmericasNLI/data/processed/test_{lang}.tsv', delimiter='\t')
        #ret_dat = load_xnli_dataset(lang, split, dataset_frac = 1.0)
    else:
        ret_dat = load_xnli_dataset(lang, split, dataset_frac = 1.0)

    return ret_dat

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@backoff.on_exception(backoff.expo, openai.error.APIError)
def gpt3x_completion(
    prompt: Union[str, List[Dict[str, str]]],
    model: str,
    run_details: Any = {},
    num_evals_per_sec: int = 2,
    **model_params,
) -> str:
    output = None
    if isinstance(prompt, str):
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=model_params.get("max_tokens", 20),
            temperature=model_params.get("temperature", 1),
            top_p=model_params.get("top_p", 1),
        )
        if "num_calls" in run_details:
            run_details["num_calls"] += 1
        output = response["choices"][0]["text"].strip().split("\n")[0]
        time.sleep(1 / num_evals_per_sec)
    else:
        response = openai.ChatCompletion.create(
            engine=model,
            messages=prompt,
            max_tokens=model_params.get("max_tokens", 20),
            temperature=model_params.get("temperature", 1),
            top_p=model_params.get("top_p", 1),
        )
        if "num_calls" in run_details:
            run_details["num_calls"] += 1
        if response["choices"][0]["finish_reason"] == "content_filter":
            output = ""
        else:
            output = response["choices"][0]["message"][
                "content"
            ].strip()  # .split("\n")[0]
        time.sleep(1 / num_evals_per_sec)

    return output

langcodes2lang = {
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin",
    "it": "Italian",
    "sw": "Swahili",
    "id": "Indonesian",
    "et": "Estonian",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "qu": "Quechua",
    "ht": "Haitian Creole",
    "el": "Greek",
    "ru": "Russian",
    "ro": "Romanian",
    "fi": "Finnish",
    "ko": "Korean",
    "te": "Telugu",
    "bn": "Bengali",
    # modifications
    "hau" : "Hausa",
    "kin" : "Kinyarwanda",
    "ibo" : "Igbo",
    "lug" : "Luganda",
    "luo" : "Luo",
    "fo"  : "Faroese",
    "got" : "Gothic",
    "gsw" : "Swiss-German",
    "aym" : "Aymara",
    "cni" : "Asháninka",
    "bzd" : "Bribri",
    "gn"  : "Guaraní",
    "nah" : "Nahuatl",
    "oto" : "Otomí",
    "quy" : "Quechua",
    "tar" : "Rarámuri",
    "shp" : "Shipibo-Konibo",
    "hch" : "Wixarika",
}

lang2langcodes = {v: k for k, v in langcodes2lang.items()}

DATASET2PROPER_NAMES = {
    "xcopa": "XCOPA",
    "copa": "COPA",
    "xnli": "XNLI",
    "mnli": "MNLI",
    "indic_xnli": "Indic-XNLI",
    "xstory_cloze": "XStory Cloze",
    "xquad": "XQuAD",
    "squad": "SQuAD",
    "tydiqa": "TyDiQA-GoldP",
    "indicqa": "IndicQA",
    "mlqa": "MLQA",
    "pawsx": "PAWS-X",
    "xlsum": "XLSum",
    "udpos": "UDPOS",
    "panx": "PAN-X",
    "wikiann": "WikiANN",
    "gluecos": "GLUECoS",
    "en-es-cs": "EN-ES-CS",
    "jigsaw": "Jigsaw Multilingual Toxic Comment Classification",
    # Modifications
    "masakhaner": "MasakhaNER",
    "americasnli": "AmericasNLI",
}
cls_datasets = ["udpos", "masakhaner", "americasnli", "xnli", "mnli", "xcopa", "copa", "pawsx", "xstory_cloze"]
qa_datasets = ["xquad", "squad", "tydiqa", "indicqa", "mlqa"]
NLI_FORMAT = """id | premise | hypothesis | label
---|---------|------------|------
1 | {premise1} | {hypothesis1} | {label1}
2 | {premise2} | {hypothesis2} | {label2}
...
"""

COPA_FORMAT = """id | premise | choice1 | choice2 | question_type | label
---|--------|---------|---------|----------|------
1 | {premise1} | {choice1_1} | {choice1_2} | {question_type_1} | {label1}
1 | {premise2} | {choice2_1} | {choice2_2} | {question_type_2} | {label2}
...
"""

QA_FORMAT = """id | context | question | answers
---|---------|----------|--------
{id1} | {context1} | {question1} | {answers1}
{id2} | {context2} | {question2} | {answers2}
...
"""

COPA_JSON_FORMAT = {
    "premise": "{premise}",
    "choice1": "{choice1}",
    "choice2": "{choice2}",
    "question": "{question}",
    "label": "{label}",
    "idx": "{idx}",
    "changed": "{changed}",
}

QA_JSON_FORMAT = {
    "context": "{context}",
    "qas": [
        {
            "answers": [
                {
                    "answer_start": "{answer_start}",
                    "text": "{text}",
                }
            ],
            "id": "{id}",
            "question": "{question}",
        }
    ],
}

NLI_VERBALIZER = "0 for entailement, 1 for neutral, 2 for contradiction"
COPA_VERBALIZER = "0 for choice 1, 1 for choice 2"
FORMATS = {
    "americasnli": NLI_FORMAT,
    "xnli": NLI_FORMAT,
    "mnli": NLI_FORMAT,
    "xcopa": COPA_FORMAT,
    "copa": COPA_FORMAT,
    "squad": QA_FORMAT,
    "xquad": QA_FORMAT,
    "tydiqa": QA_FORMAT,
    "mlqa": QA_FORMAT,
    "indicqa": QA_FORMAT,
}

JSON_FORMATS = {
    "xcopa": COPA_JSON_FORMAT,
    "copa": COPA_JSON_FORMAT,
    "xquad": QA_JSON_FORMAT,
}

VERBALIZERS = {
    "americasnli": NLI_VERBALIZER,
    "xnli": NLI_VERBALIZER,
    "mnli": NLI_VERBALIZER,
    "xcopa": COPA_VERBALIZER,
    "copa": COPA_VERBALIZER,
}

DATASET_CARD_SCHEMA = """languages: 
- "List of ISO 639-1 code for languages covered in the dataset"
- lang1
- lang2
...
- langM
pretty_name: "Pretty Name of the Dataset"
tags:
- tag1
- tag2
dataset_info:
    features:
        - name: "name of the feature1"
          dtype: "dtype of the feature1"
        ...
        - name: "name of the featureN"
          dtype: "dtype of the featureN"
    splits:
        train:
            num_examples: "number of examples in the train split"
        validation:
            num_examples: "number of examples in the validation split"
        test:
            num_examples: "number of examples in the test split"
dataset_summary: "Summary of the dataset's description"
"""


def check_contamination(
    model: str,
    dataset: str,
) -> str:
    """
    Checks if a dataset is present in the LM's pre-training data
    """

    pass


def remove_tokens(text: str, tokenizer, ratio: float = 0.5) -> str:
    tokens = tokenizer.tokenize(text, add_special_tokens=False)
    tokens_removed = tokens[: int(len(tokens) * ratio)]
    text_removed = tokenizer.convert_tokens_to_string(tokens_removed)
    text_removed = text_removed + "[MASK]"
    return text_removed


def get_fewshot_prompt(dataset, lang, split, k=10):
    """
    Generates few-shot examples from the dataset

    Inputs:
        - dataset: str, name of the dataset
        - lang: str
        - split: str
        - k: int, number of few-shot examples to provide
    """

    hf_dataset = load_dataset_mega(dataset, lang=lang, split=split)
    #pdb.set_trace()
    if dataset in JSON_FORMATS:
        pdb.set_trace()
        fewshot_prompt = ""
        json_prompt = True

        json_dataset = load_json_datasets(dataset, lang=lang, split=split)
        # if dataset not in qa_datasets:
        fewshot_prompt += json.dumps(json_dataset[:k])
        # else:
        #     fewshot_prompt = json.dumps(
        #         {
        #             "context": json_dataset[0]["context"],
        #             "qas": json_dataset[0]["qas"][:k],
        #         }
        #     )
        #     fewshot_prompt = f"{fewshot_prompt[:-2]},"

    else:
        format_header = FORMATS[dataset].split("\n")[0]
        format_sep = FORMATS[dataset].split("\n")[1]
        fewshot_prompt = f"{format_header}\n{format_sep}\n"
        json_prompt = False

        for i in range(k):
            example = hf_dataset[i]

            if json_prompt:
                format = JSON_FORMATS[dataset]
                format_filled = copy.deepcopy(format)
                for attr in format.keys():
                    if attr in example.keys():
                        format_filled[attr] = example[attr]
                    else:
                        raise ValueError(f"Attribute {attr} not found in the dataset")
                fewshot_prompt += json.dumps(format_filled) + "\n"
            else:
                example_fs_prompt = ""
                attrs = format_header.split("|")
                for attr in attrs:
                    if attr.strip() == "question_type" and dataset in ["xcopa", "copa"]:
                        attr = "question"  # Hack for COPA datasets

                    if attr.strip() == "id":
                        if "id" not in example.keys():
                            example_fs_prompt += f"{i+1} | "
                        else:
                            example_fs_prompt += f"{example['id']} | "
                    elif attr.strip() in example.keys():
                        example_fs_prompt += f"{example[attr.strip()]} | "
                    elif attr.strip() == 'label':
                        example_fs_prompt += f"{example['output']} | "
                    else:
                        raise ValueError(
                            f"Attribute {attr.strip()} not found in the dataset"
                        )

                fewshot_prompt += example_fs_prompt + "\n"

    return fewshot_prompt


def construct_dataset_completion_prompt(
    dataset, lang, split, tokenizer, num_instances=10, completion_ratio=0.5
) -> Tuple[str, str]:
    """
    Takes first `num_instances` examples of the dataset, remove `compeltion_ratio` of the tokens, and construct a prompt

    Inputs:
        - dataset: str
        - lang: str
        - split: str
        - tokenizer: HuggingFace tokenizer
        - num_instances: int, number of instances to use
        - completion_ratio: float, ratio of tokens to remove
    """
    hf_dataset = load_dataset_mega(dataset, lang=lang, split=split)
    format_header = FORMATS[dataset].split("\n")[0]
    format_sep = FORMATS[dataset].split("\n")[1]
    dataset_completion_prompt = f"{format_header}\n{format_sep}\n"
    dataset_full_prompt = f"{format_header}\n{format_sep}\n"
    for i in range(num_instances):
        example = hf_dataset[i]
        example_completion_prompt = ""
        example_full_prompt = ""
        attrs = format_header.split("|")
        for attr in attrs:
            if attr.strip() == "id":
                if "id" not in example.keys():
                    example_completion_prompt += f"{i+1} | "
                    example_full_prompt += f"{i+1} | "
                else:
                    example_completion_prompt += f"{example['id']} | "
                    example_full_prompt += f"{example[attr.strip()]} | "
            elif attr.strip() in ["label", "answers"]:
                example_completion_prompt += f"[MASK]\n"
                if attr.strip() == 'label':
                    example_full_prompt += f"{example['output']}\n"
                else:
                    example_full_prompt += f"{example[attr.strip()]}\n"
            elif attr.strip() == "question_type" and dataset in ["xcopa", "copa"]:
                example_completion_prompt += f"[MASK] | "
                example_full_prompt += f"{example['question']} | "
            elif attr.strip() in example.keys():
                example_completion_prompt += f"{remove_tokens(example[attr.strip()], tokenizer, ratio=completion_ratio)} | "
                example_full_prompt += f"{example[attr.strip()]} | "
            else:
                raise ValueError(f"Attribute {attr} not found in example")

        dataset_completion_prompt += example_completion_prompt
        dataset_full_prompt += example_full_prompt

    return dataset_completion_prompt, dataset_full_prompt


def main(sys_args):
    args = parse_args(sys_args)
    load_openai_env_variables()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_proper_name = DATASET2PROPER_NAMES[args.dataset]

    if args.dataset == "xcopa" and args.contam_lang == "en":
        dataset_proper_name = "COPA"

    # elif args.dataset == "xquad" and args.contam_lang == "en":
    #     dataset_proper_name = "SQuAD"

    # used in the "complete" method
    mbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    if args.contam_method in ["generate", "generate_few_shot", "complete"]:
        save_dir = os.path.join(
            args.save_dir, args.contam_method, args.dataset, args.model
        )
        save_file = os.path.join(save_dir, f"{args.dataset}.{args.contam_lang}.txt")
        save_file_json = os.path.join(
            save_dir, f"{args.dataset}.{args.contam_lang}.json"
        )
    elif args.contam_method in ["fill_dataset_card", "fill_dataset_card_w_example"]:
        save_dir = os.path.join(args.save_dir, args.contam_method, args.model)
        save_file = os.path.join(save_dir, f"{args.dataset}.txt")
        save_file_json = os.path.join(save_dir, f"{args.dataset}.json")
    else:
        raise ValueError(f"Contamination method {args.contam_method} not supported")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results = []
    splits = get_dataset_splits(args.dataset)
    for sp_idx, split in enumerate(splits):
        if args.contam_method == "generate":
            if args.dataset in cls_datasets:
                prompt = [
                    {
                        "role": "user",
                        "content": f"Please generate the first instances of {dataset_proper_name} {langcodes2lang[args.contam_lang]} dataset {split} split.",
                    },
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": f"Please generate the first instances of {dataset_proper_name} {langcodes2lang[args.contam_lang]} dataset {split} split in squad format.",
                    },
                ]
            print(prompt)
            model_response = gpt3x_completion(
                prompt=prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(model_response)
            read_mode = "a" if sp_idx > 0 else "w"
            with open(save_file, read_mode) as f:
                f.write("Split: " + split + "\n")
                f.write("Prompt:\n")
                f.write("\n".join([p["content"] for p in prompt]))
                f.write("\n")
                f.write("Response:\n")
                f.write(model_response)
                f.write("\n")
                f.write("=" * 50 + "\n")
                f.write("\n")

            results.append(
                {"split": split, "prompt": prompt, "response": model_response}
            )

        elif args.contam_method == "generate_few_shot":
            fs_exs = get_fewshot_prompt(
                args.dataset, args.contam_lang, split, k=args.few_shot_k
            )

            if args.model in CHAT_MODELS:
                prompt = [
                    {
                        "role": "user",
                        "content": f"Please generate the first instances of {dataset_proper_name} {langcodes2lang[args.contam_lang]} dataset {split} split. Here are the first few examples, please continue them to help me complete the full dataset:\n{fs_exs}",
                    },
                ]
            else:
                prompt = f"Please generate instances of {dataset_proper_name} {langcodes2lang[args.contam_lang]} dataset {split} split.\n\n{fs_exs}"

            print(prompt)
            model_response = gpt3x_completion(
                prompt=prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(model_response)
            read_mode = "a" if sp_idx > 0 else "w"
            with open(save_file, read_mode) as f:
                f.write("Split: " + split + "\n")
                f.write("Prompt:\n")
                if args.model in CHAT_MODELS:
                    f.write("\n".join([p["content"] for p in prompt]))
                else:
                    f.write(prompt)
                f.write("\n")
                f.write("Response:\n")
                f.write(model_response)
                f.write("\n")
                f.write("=" * 50 + "\n")
                f.write("\n")

            results.append(
                {"split": split, "prompt": prompt, "response": model_response}
            )

        elif args.contam_method == "complete":
            (
                dataset_completion_prompt,
                dataset_full_prompt,
            ) = construct_dataset_completion_prompt(
                args.dataset,
                args.contam_lang,
                split,
                mbert_tokenizer,
                num_instances=10,
                completion_ratio=0.5,
            )
            if args.dataset in cls_datasets:
                prompt = [
                    {
                        "role": "user",
                        "content": f"I am trying to recover {split} split of {dataset_proper_name} {langcodes2lang[args.contam_lang]} dataset. I have lost 50% of the tokens for each piece of text and lost the labels altogether. I have the following prompt:\n{dataset_completion_prompt}\nPlease fill in the missing tokens (indicated by [MASK]) in the prompt and return the recovered dataset in the same format. You should map the labels based on the following rules:\n{VERBALIZERS[args.dataset]}",
                    }
                ]
            else:
                prompt = [
                    {
                        "role": "user",
                        "content": f"I am trying to recover {split} split of {dataset_proper_name} {langcodes2lang[args.contam_lang]} dataset. I have lost 50% of the tokens for each piece of text and lost the labels altogether. I have the following prompt:\n{dataset_completion_prompt}\nPlease fill in the missing tokens (indicated by [MASK]) in the prompt and return the recovered dataset in the same format.",
                    }
                ]
            print(prompt)
            model_response = gpt3x_completion(
                prompt=prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(model_response)
            read_mode = "a" if sp_idx > 0 else "w"
            with open(save_file, read_mode) as f:
                f.write("Split: " + split + "\n")
                f.write("Completion Prompt:\n")
                f.write(dataset_completion_prompt)
                f.write("\n")
                f.write("Model's Response:\n")
                f.write(model_response)
                f.write("\n")
                f.write("Ground Truth:\n")
                f.write(dataset_full_prompt)
                f.write("\n")
                f.write("=" * 50 + "\n")
                f.write("\n")
            results.append(
                {
                    "split": split,
                    "completion_prompt": dataset_completion_prompt,
                    "model_response": model_response,
                    "ground_truth": dataset_full_prompt,
                }
            )

        elif args.contam_method == "fill_dataset_card":
            prompt = [
                {
                    "role": "user",
                    "content": f"Please fill the dataset card for the {dataset_proper_name} dataset. You can use the following schema:\n{DATASET_CARD_SCHEMA}",
                },
            ]
            print(prompt)
            model_response = gpt3x_completion(
                prompt=prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(model_response)
            with open(save_file, "w") as f:
                f.write(model_response)
                f.write("\n")
            results = {
                "prompt": prompt,
                "response": model_response,
            }
            break  # Not needed to generate response for each split in this case

        elif args.contam_method == "fill_dataset_card_w_example":
            if "nli" not in args.dataset:
                with open("xnli.yaml", "r") as f:
                    example_dataset_card = f.read()
            else:
                with open("xcopa.yaml", "r") as f:
                    example_dataset_card = f.read()

            prompt = [
                {
                    "role": "user",
                    "content": f"Please fill the dataset card for the {dataset_proper_name} dataset. You can use the following schema:\n{DATASET_CARD_SCHEMA} and the following example:\n{example_dataset_card}",
                },
            ]
            print(prompt)
            model_response = gpt3x_completion(
                prompt=prompt,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(model_response)
            with open(save_file, "w") as f:
                f.write(model_response)
                f.write("\n")
            results = {
                "prompt": prompt,
                "response": model_response,
            }
            break  # Not needed to generate response for each split in this case

    with open(save_file_json, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1:])
