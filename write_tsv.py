import csv, sys, os
import json

lang = sys.argv[1]
mode = sys.argv[2]
with open(mode + "_" + lang + ".json") as f:
	d = json.load(f)

label_dict = {"0": "entailment", "1": "neutral", "2": "contradiction"}
data = []
for label in d:
	for item in d[label]:
		data.append({"premise": item["premise"], "hypothesis": item["hypothesis"], "output": label, "language": lang })

# TSV file path
tsv_file_path = mode + "_" + lang + ".tsv"

# Open the TSV file in write mode
with open(tsv_file_path, 'w', newline='') as tsvfile:
    # Create a TSV writer
    writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=data[0].keys())

    # Write the header
    writer.writeheader()

    # Write the data
    writer.writerows(data)

