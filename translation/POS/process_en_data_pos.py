import os
import logging
import copy
import pickle
import argparse
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, langs=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          words: list. The words of the sequence.
          labels: (Optional) list. The labels for each word of the sequence. This should be
          specified for train and dev examples, but not for test examples.
        """
        tag2mark_xml = {
            'ADJ': {'s': '<ORG>', 'e': '</ORG>'},
            'ADP': {'s': '<ADP>', 'e': '</ADP>'},
            'ADV': {'s': '<ADV>', 'e': '</ADV>'},
            'AUX': {'s': '<AUX>', 'e': '</AUX>'},
            'CCONJ': {'s': '<CCONJ>', 'e': '</CCONJ>'},
            'DET': {'s': '<DET>', 'e': '</DET>'},
            'INTJ': {'s': '<INTJ>', 'e': '</INTJ>'},
            'NOUN': {'s': '<NOUN>', 'e': '</NOUN>'},
            'NUM': {'s': '<NUM>', 'e': '</NUM>'},
            'PART': {'s': '<PART>', 'e': '</PART>'},
            'PRON': {'s': '<PRON>', 'e': '</PRON>'},
            'PROPN': {'s': '<PROPN>', 'e': '</PROPN>'},
            'PUNCT': {'s': '<PUNCT>', 'e': '</PUNCT>'},
            'SCONJ': {'s': '<SCONJ>', 'e': '</SCONJ>'},
            'SYM': {'s': '<SYM>', 'e': '</SYM>'},
            'VERB': {'s': '<VERB>', 'e': '</VERB>'},
            'X': {'s': '<X>', 'e': '</X>'}            
        }

        tag2mark_marker = {
            'ADJ': {'s': '[', 'e': ']'},
            'ADP': {'s': '[', 'e': ']'},
            'ADV': {'s': '[', 'e': ']'},
            'AUX': {'s': '[', 'e': ']'},
            'CCONJ': {'s': '[', 'e': ']'},
            'DET': {'s': '[', 'e': ']'},
            'INTJ': {'s': '[', 'e': ']'},
            'NOUN': {'s': '[', 'e': ']'},
            'NUM': {'s': '[', 'e': ']'},
            'PART': {'s': '[', 'e': ']'},
            'PRON': {'s': '[', 'e': ']'},
            'PROPN': {'s': '[', 'e': ']'},
            'PUNCT': {'s': '[', 'e': ']'},
            'SCONJ': {'s': '[', 'e': ']'},
            'SYM': {'s': '[', 'e': ']'},
            'VERB': {'s': '[', 'e': ']'},
            'X': {'s': '[', 'e': ']'}
        }


        self.guid = guid
        self.words = words
        self.labels = labels
        self.langs = langs
        self.entity_list, self.tag_list, self.span_labels = self.extract_entity(self.words, self.labels)
        self.xml_encode_sent = self.encode(self.words, self.labels, tag2mark_xml)
        self.marker_encode_sent = self.encode(self.words, self.labels, tag2mark_marker)
        self.org_trans = {}
        self.xml_trans = {}
        self.marker_trans = {}
        self.ent_trans = {}


    def decode_label_span(self, label):
        label_tags = label
        span_labels = []
        last = 'O'
        start = -1
        # for i, tag in enumerate(label_tags):
        #     pos, _ = (None, 'O') if tag == 'O' else tag.split('-')
        #     if (pos == 'B' or tag == 'O') and last != 'O':  # end of span
        #         span_labels.append((start, i, last.split('-')[1]))
        #     if pos == 'B' or last == 'O':  # start of span or move on
        #         start = i
        #     last = tag
        # if label_tags[-1] != 'O':
        #     span_labels.append((start, len(label_tags), label_tags[-1].split('-')[1]))

        for i, tag in enumerate(label_tags):
            start = i
            span_labels.append((start, start + 1, tag))

        return span_labels

    def extract_entity(self, word, label):

        span_labels = self.decode_label_span(label)
        entity_list = []
        tag_list = []
        for span in span_labels:
            s, e, tag = span
            entity = word[s: e]
            entity_list.append(entity)
            tag_list.append(tag)

        return entity_list, tag_list, span_labels

    def encode(self, tokens, label, tag2mark):
        copy_tokens = copy.deepcopy(tokens)
        for idx, lab in enumerate(label):
            # print("idx: ", idx, "lab: ",lab)
            S = tag2mark[lab]['s'] + " " + tokens[idx] + " " + tag2mark[lab]['e']
            copy_tokens[idx] = S
                
        encoded_sentence = ' '.join(copy_tokens)
        return encoded_sentence

    def add_org_translation(self, lang, sent):
        self.org_trans[lang] = sent
    def add_xml_translation(self, lang, sent):
        self.xml_trans[lang] = sent
    def add_marker_translation(self, lang, sent):
        self.marker_trans[lang] = sent
    def add_ent_translation(self, lang, ent_list):
        self.ent_trans[lang] = ent_list



    def __str__(self):
        str = "words:{}\nlabels:{}\nentity_list:{}\ntag_list:{}\nspan:{}\nxml:{}\nmarker:{}".format(
            self.words, self.labels, self.entity_list, self.tag_list, self.span_labels,
            self.xml_encode_sent, self.marker_encode_sent
        )
        return str


# 1. load English data
def read_examples_from_file(file_path, lang, lang2id=None):
    if not os.path.exists(file_path):
        logger.info("[Warming] file {} not exists".format(file_path))
        return []
    guid_index = 1
    examples = []
    subword_len_counter = 0
    if lang2id:
        lang_id = lang2id.get(lang, lang2id['en'])
    else:
        lang_id = 0
    logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        langs = []
        cnt = 0
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if word:
                    examples.append(InputExample(guid="{}-{}".format(lang, guid_index),
                                                 words=words,
                                                 labels=labels,
                                                 langs=langs))
                    guid_index += 1
                    # print("guid_index: ", guid_index)
                    words = []
                    labels = []
                    langs = []
                    subword_len_counter = 0
                else:
                    print(f'guid_index', guid_index, words, langs, labels, subword_len_counter)
            else:
                splits = line.strip().split(" ")
                word = splits[0]
                # print("Length of splits: ", len(splits))
                # print("word: ", word)

                
                langs.append(lang_id)
                if len(splits) > 1:
                    words.append(splits[0])
                    labels.append(splits[-1].replace("\n", ""))
                # else:
                #     # Examples could have no label for mode = "test"
                #     labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(lang, guid_index),
                                         words=words,
                                         labels=labels,
                                         langs=langs))
    return examples





def save_pickle(file_name, data):
    # save results from model
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    examples = read_examples_from_file(args.input_path, 'en')

    print(examples[0])
    print(examples[58])

    org_sentences, entities = [], []
    for example in examples:
        org_sentences.append(' '.join(example.words))
        entities.extend(example.entity_list)

    pkl_path = os.path.join(args.output_dir, "conll_en_train_examples.pkl")
    org_sentence_path = os.path.join(args.output_dir, "conll_en_train_org.txt")
    entity_path = os.path.join(args.output_dir, "conll_en_train_entity.txt")
    save_pickle(pkl_path, examples)
    with open(org_sentence_path, 'w') as f:
        f.write('\n'.join(org_sentences))
    with open(entity_path, 'w') as f:
        f.write("\n".join([" ".join(it) for it in entities]))
        # f.write('\n'.join(entities))
