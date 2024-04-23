import sys
import os
import csv
from statistics import mean

def aggregate_accuracies(directory):
    # Create the top-level directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Traverse the directory tree
    for family_dir in os.listdir(directory):
        family_path = os.path.join(directory, family_dir)

        if not os.path.isdir(family_path):
            continue

        # Traverse the family subdirectories
        for expt_dir in os.listdir(family_path):
            expt_path = os.path.join(family_path, expt_dir)

            if not os.path.isdir(expt_path):
                continue

            results_path = os.path.join(expt_path, "results.csv")
            expt_results = []
            with open(results_path, "w", newline='') as expt_results_file:
                expt_results_writer = csv.writer(expt_results_file)
                expt_results_writer.writerow(["lang", "precision", "recall", "f1", "total"])

                for lang_dir in os.listdir(expt_path):
                    lang_path = os.path.join(expt_path, lang_dir)

                    if os.path.isdir(lang_path):
                        accuracies_path = os.path.join(lang_path, "accuracies.csv")
                        if not os.path.exists(accuracies_path):
                            print(f"WARN(DNE)  : {accuracies_path}")
                            continue

                        with open(accuracies_path, "r") as accuracies_file:
                            reader = csv.DictReader(accuracies_file)
                            for row in reader:
                                for k in row:
                                    if k in ['precision', 'recall', 'f1']:
                                        row[k] = f"{100*float(row[k]):05.2f}"
                                if row['total'] != '100':
                                    print(f'WARN(LT100): {accuracies_path}')
                                expt_results_writer.writerow([lang_dir] + [row["precision"], row["recall"], row["f1"], row["total"]])
                                row['lang'] = lang_dir
                                expt_results.append(row)

                # Calculate and write the average row to expt_results.csv
                if len(expt_results) > 0:
                    avg_row = ["avg"] + [f'{mean([float(row["precision"]) for row in expt_results]):05.2f}',
                                         f'{mean([float(row["recall"]) for row in expt_results]):05.2f}',
                                         f'{mean([float(row["f1"]) for row in expt_results]):05.2f}',
                                         f'{int(mean([int(row["total"]) for row in expt_results]))}']
                    expt_results_writer.writerow(avg_row)

        aggr_results_path = os.path.join(family_path, "aggr_results.csv")
        with open(aggr_results_path, 'w') as aggr_results:
            for expt_dir in os.listdir(family_path):
                expt_path = os.path.join(family_path, expt_dir)
                if not os.path.isdir(expt_path):
                    continue
                results_path = os.path.join(expt_path, 'results.csv')
                aggr_results.write(expt_dir+'\n')
                with open(results_path, 'r') as expt_results_file:
                    for l in expt_results_file:
                        aggr_results.write(l)
                aggr_results.write('\n')

if __name__ == "__main__":
    gpt4_directory = sys.argv[1]
    aggregate_accuracies(gpt4_directory)
