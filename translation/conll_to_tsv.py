import sys
all_sents = open("data/new/"+sys.argv[1]+"/train.txt").read().strip().split("\n\n")

def write_to_tsv(data, filename):
	with open(filename, 'w') as tsv_file:
		for row in data:
			tsv_file.write('\t'.join(map(str, row)) + '\n')

data = [["input", "output", "language"]]

for sent in all_sents:
	words = [it.split(" ")[0] for it in sent.split("\n")]
	labels = [it.split(" ")[1] for it in sent.split("\n")]
	inp_ = " ".join(words)
	out_ = " ".join(["_".join([w,o]) for w,o in zip(words, labels)])
	data.append([inp_, out_, sys.argv[1]])

filename = "data/new/"+sys.argv[1]+"/train.tsv"

write_to_tsv(data, filename)


