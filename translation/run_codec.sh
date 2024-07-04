for lang in $1; do
    python pos_prompt.py ${lang} \
        ../Codec/tagged/${lang}.txt \
        ../iterate/data/new/${lang}/test_trans.txt \
        ../iterate/data/new/${lang}/test.tsv \
        new_results/codec/${lang}/responses.json;
done
