from dataset import JSONLDataset
import sys

ner_tags = ["ORG", "PER", "LOC", "DATE", "O"]

if __name__ == "__main__":
    ds = JSONLDataset(sys.argv[1])
    idxs = []
    with open(sys.argv[2], 'r') as idxfile:
        for l in idxfile:
            idxs.append([int(a) for a in l.split(' ')])

    for idxlist in idxs:
        tag_cnt = {tag: 0 for tag in ner_tags}
        for idx in idxlist:
            for lbl in ds[idx]['pred_labels']:
                tag_cnt[lbl.split('-')[-1]] += 1
        for tag, cnt in tag_cnt.items():
            print(f"{tag}: {cnt}", end="\t")
        print()
