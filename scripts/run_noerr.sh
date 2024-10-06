python iterate.py \
    -d ../llama/data/processed/test_hau.tsv \
    -td new_results/gpt35/hau_8s/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/hau_100/test_hau_sim.npy \
    -ssim ../llama/data/hau_100/test_hau_allsrc_sim.npy \
    -tr 4 -sr 4 -m gpt-3.5-turbo \
    -p ner -e 100 \
    -r new_results/gpt35/hau_nomistake_4s4t

python iterate.py \
    -d ../llama/data/processed/test_ibo.tsv \
    -td new_results/gpt35/ibo_8s/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/ibo_100/test_ibo_sim.npy \
    -ssim ../llama/data/ibo_100/test_ibo_allsrc_sim.npy \
    -tr 4 -sr 4 -m gpt-3.5-turbo \
    -p ner -e 100 \
    -r new_results/gpt35/ibo_nomistake_4s4t

python iterate.py \
    -d ../llama/data/processed/test_kin.tsv \
    -td new_results/gpt35/kin_8s/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/kin_100/test_kin_sim.npy \
    -ssim ../llama/data/kin_100/test_kin_allsrc_sim.npy \
    -tr 4 -sr 4 -m gpt-3.5-turbo \
    -p ner -e 100 \
    -r new_results/gpt35/kin_nomistake_4s4t

python iterate.py \
    -d ../llama/data/processed/test_luo.tsv \
    -td new_results/gpt35/luo_8s/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/luo_100/test_luo_sim.npy \
    -ssim ../llama/data/luo_100/test_luo_allsrc_sim.npy \
    -tr 4 -sr 4 -m gpt-3.5-turbo \
    -p ner -e 100 \
    -r new_results/gpt35/luo_nomistake_4s4t

python iterate.py \
    -d ../llama/data/processed/test_lug.tsv \
    -td new_results/gpt35/lug_8s/responses.json \
    -sd ../llama/data/processed/train_en_conll-amh-swa-wol.tsv \
    -tsim ../llama/data/lug_100/test_lug_sim.npy \
    -ssim ../llama/data/lug_100/test_lug_allsrc_sim.npy \
    -tr 4 -sr 4 -m gpt-3.5-turbo \
    -p ner -e 100 \
    -r new_results/gpt35/lug_nomistake_4s4t
