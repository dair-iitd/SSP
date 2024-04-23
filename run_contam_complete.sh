# americasNLI
dataset=americasnli
for lang in aym quy nah gn
do
    for model in "gpt-35-turbo" "gpt-4"
    do
        echo $dataset $model $lang
        python3 contamination.py -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_lang ${lang} --contam_method "complete" --save_dir contamination_results
    done
done
