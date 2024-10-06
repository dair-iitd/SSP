for lang in $1; do
    python ner_prompt.py ${lang} \
        ../Codec/tagged/${lang}.txt \
        data/new/${lang}/test_trans.txt \
        data/new/${lang}/test.tsv \
        new_results/codec/${lang}/responses.json;
done
