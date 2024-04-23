# udpos
dataset=udpos
for lang in fo got gsw
do
    for model in $1
    do
        echo $dataset $model $lang
        #python3 contamination.py -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_lang ${lang} --contam_method "generate" --save_dir contamination_results
    done
done

# masakhaNER
dataset=masakhaner
for lang in hau ibo kin lug luo
do
    for model in $1
    do
        echo $dataset $model $lang
        #python3 contamination.py -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_lang ${lang} --contam_method "generate" --save_dir contamination_results
    done
done

# americasNLI
dataset=americasnli
for lang in aym quy nah gn
do
    for model in $1
    do
        echo $dataset $model $lang
        python3 contamination.py -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_lang ${lang} --contam_method "generate" --save_dir contamination_results
    done
done
