python iterate.py \
    -d ../llama/data/processed/test_hau.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/hau_100/test_hau_allsrc_sim.npy \
    -tr 0 -sr 8 -m gpt-4 \
    -p ner -e 100 \
    -r new_results/gpt4/hau_8s

python iterate.py \
    -d ../llama/data/processed/test_ibo.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/ibo_100/test_ibo_allsrc_sim.npy \
    -tr 0 -sr 8 -m gpt-4 \
    -p ner -e 100 \
    -r new_results/gpt4/ibo_8s

python iterate.py \
    -d ../llama/data/processed/test_luo.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/luo_100/test_luo_allsrc_sim.npy \
    -tr 0 -sr 8 -m gpt-4 \
    -p ner -e 100 \
    -r new_results/gpt4/luo_8s

python iterate.py \
    -d ../llama/data/processed/test_kin.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/kin_100/test_kin_allsrc_sim.npy \
    -tr 0 -sr 8 -m gpt-4 \
    -p ner -e 100 \
    -r new_results/gpt4/kin_8s

python iterate.py \
    -d ../llama/data/processed/test_lug.tsv \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -ssim ../llama/data/lug_100/test_lug_allsrc_sim.npy \
    -tr 0 -sr 8 -m gpt-4 \
    -p ner -e 100 \
    -r new_results/gpt4/lug_8s
