LANG=$1
PER=$2
echo "All constr"
python show_count_stats.py -sd data/processed/test_${LANG}.tsv -td data/processed/${LANG}/${LANG}_predictions.json -tsim data/processed/${LANG}/${LANG}_${PER}_per_ILP.npy
echo "No label"
python show_count_stats.py -sd data/processed/test_${LANG}.tsv -td data/processed/${LANG}/${LANG}_predictions.json -tsim data/processed/${LANG}/${LANG}_${PER}_per_ILP_no_lab_cons.npy
echo "No conf"
python show_count_stats.py -sd data/processed/test_${LANG}.tsv -td data/processed/${LANG}/${LANG}_predictions.json -tsim data/processed/${LANG}/${LANG}_${PER}_per_ILP_no_conf_cons.npy
echo "Sim-based"
python show_count_stats.py -sd data/processed/test_${LANG}.tsv -td data/processed/${LANG}/${LANG}_predictions.json -tsim data/processed/${LANG}_tgt_tgt_sim.npy
# echo "Sim-based ZGUL"
# python save_exemplar_predictions.py -sd data/processed/test_${LANG}.tsv -td data/processed/${LANG}/${LANG}_predictions.json -tsim data/new/${LANG}/tgt_sim.npy
# echo "Random ZGUL"
# python save_exemplar_predictions.py -sd data/processed/test_${LANG}.tsv -td data/processed/${LANG}/${LANG}_predictions.json -tsim data/new/${LANG}/rand.npy
