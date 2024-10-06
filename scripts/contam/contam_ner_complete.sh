for lang in got_100 got_200 got_500
do
    for model in "gpt-35-turbo" "gpt-4"
    do
        python contam_ner_complete.py ${lang} ${model}
    done
done
