# for dataset in udpos americasnli masakhaner
# do
#     for model in "gpt-35-turbo" "gpt-4"
#     do
#         echo $dataset $model
#         python3 contamination.py -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_method "fill_dataset_card" --save_dir contamination_results
#     done
# done

for dataset in udpos americasnli masakhaner
do
    for model in "gpt-35-turbo" "gpt-4"
    do
        echo $dataset $model
        python3 contamination.py -e gpt4v2 --model ${model} -d ${dataset} --max_tokens 500 --contam_method "fill_dataset_card_w_example" --save_dir contamination_results
    done
done
