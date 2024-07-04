import sys, pdb
import csv
f_w = open("data/new/"+sys.argv[1]+"/test.txt", "w")
import sys
with open("data/new/"+sys.argv[1]+"/test.tsv") as file:
	tsv_file = csv.reader(file, delimiter="\t")
	next(tsv_file)
    # printing data line by line
	for line in tsv_file:
		sent = line[1]
		words = ["".join(it.split("_")[:-1]) for it in sent.split(" ")]
		labels = [it.split("_")[-1] for it in sent.split(" ")]
		assert len(words) == len(labels)
		f_w.write("\n".join([" ".join([w,p]) for w,p in zip(words,labels)]))
		f_w.write("\n\n")
		print(line)

f_w.close()


