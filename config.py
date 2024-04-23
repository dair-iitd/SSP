from promptbench.strategies import *
from promptbench.models.openai import GPT, ChatGPT
from promptbench.models.google import PaLM
from promptbench.models.llama import Llama2

strategies = {
    'ner_4s': ner_4s,
    'qaner_4s': qaner_4s,
    'ner_4s_tsv': ner_4s_tsv,
    'qaner_zs': qaner_zs,
    'qaner_zs_ml': qaner_zs_ml,
    'qaner_9s_ml': qaner_9s_ml,
    'ner_2s_conf': ner_2s_conf,
    'ner_2s_tsv': ner_2s_tsv,
    'ner_4s_adapt_conf': ner_4s_adapt_conf
}

datasets = {
    # your dataset paths go here
}

models = {
    'text-davinci-003': GPT,
    'text-davinci-002': GPT,
    'gpt-3.5-turbo': ChatGPT,
    'gpt-4': ChatGPT,
    'text-bison-001' : PaLM,
    'llama-2-70b' : Llama2
}
