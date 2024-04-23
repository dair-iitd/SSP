# americasNLI
dataset=xnli
for lang in $2
do
    #for model in "gpt-35-turbo" "gpt-4"
    for model in $1
    do
        echo $dataset $model $lang
        python3 contamination.py -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_lang ${lang} --contam_method "generate_few_shot" --save_dir contamination_results
    done
done
