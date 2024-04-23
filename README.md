# SSP-CLT

```
usage: python iterate.py [-h] [-l LANG] [-d DATASET] [-p PROMPT] [-td TARGET_DATASET] [-sd SOURCE_DATASET] [-m MODEL]
                   [-tr TARGET_RETRIEVE] [-sr SOURCE_RETRIEVE] [-y] [-r RESULT_DIR] [-rl] [--slow] [-cf] [-ssim SOURCE_SIM]
                   [-tsim TARGET_SIM] [-s SPLIT_START] [-e SPLIT_END] [-i INTERM] [-t TEMPERATURE]

Prompt benchmarking utility

options:
  -h, --help            show this help message and exit
  -l LANG, --lang LANG
  -d DATASET, --dataset DATASET
  -p PROMPT, --prompt PROMPT
  -td TARGET_DATASET, --target-dataset TARGET_DATASET
  -sd SOURCE_DATASET, --source-dataset SOURCE_DATASET
  -m MODEL, --model MODEL
                        model
  -tr TARGET_RETRIEVE, --target-retrieve TARGET_RETRIEVE
                        no. examples to retrieve from target
  -sr SOURCE_RETRIEVE, --source-retrieve SOURCE_RETRIEVE
                        no. examples to retrieve from source
  -y, --yes             Say yes to any conditionals
  -r RESULT_DIR, --result-dir RESULT_DIR
  -rl, --randomize-labels
                        randomize labels (for ablation)
  --slow                slow down API calls
  -cf, --content-filter
                        ignore content filter (save all egs)
  -ssim SOURCE_SIM, --source-sim SOURCE_SIM
                        Source similarity matrix
  -tsim TARGET_SIM, --target-sim TARGET_SIM
                        Target similarity matrix
  -s SPLIT_START, --split-start SPLIT_START
  -e SPLIT_END, --split-end SPLIT_END
  -i INTERM, --interm INTERM
  -t TEMPERATURE, --temperature TEMPERATURE
```

## Quickstart

1. Create a conda environment

```
conda create -n sspclt python=3.10
conda activate sspclt
pip install -r requirements.txt
```

2. Add keys in .env. You will require an `AZURE_OPENAI_KEY` and an 
`AZURE_OPENAI_ENDPOINT` for running GPT models, and a `TOGETHER_API_KEY` for 
running LLaMa. Place the .env file in the parent directory.

3. Run iterate.py using the commands shown above. You would need to pass a path 
   to the source dataset, and the source retrievals in tsv/jsonl format. The 
   columns of the TSV should be input, output and language, and JSONL dataset
   records should have these three keys as well. Refer to the shell script (run_all.sh)
   for more details.

4. For running LLaMa, use `iterate_llama.py`. For running SSP-CLT-ILP, use
   `iterate_ilp.py`.
