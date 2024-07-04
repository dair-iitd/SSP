LANG=$1
#echo "All constr"
#python save_exemplar_predictions.py -sd data/new/${LANG}/test.tsv -td ../zgul_plus/outputs/${LANG}_100_zgul_pred.tsv -tsim data/new/${LANG}/${LANG}_80percile_8ex_ILP_seq_grb_zgul.npy
echo "All constr new"
python save_exemplar_predictions.py -sd data/new/${LANG}/test.tsv -td ../zgul_plus/outputs/${LANG}_100_zgul_pred.tsv -tsim data/new/${LANG}/${LANG}_80percile_8ex_ILP_seq_grb_zgul_lam_1.0.npy
echo "No label"
python save_exemplar_predictions.py -sd data/new/${LANG}/test.tsv -td ../zgul_plus/outputs/${LANG}_100_zgul_pred.tsv -tsim data/new/${LANG}/${LANG}_80percile_8ex_ILP_seq_grb_no_lab_cons_zgul.npy
echo "No conf"
python save_exemplar_predictions.py -sd data/new/${LANG}/test.tsv -td ../zgul_plus/outputs/${LANG}_100_zgul_pred.tsv -tsim data/new/${LANG}/${LANG}_80percile_8ex_ILP_seq_grb_no_conf_cons_zgul.npy
echo "Sim-based ZGUL"
python save_exemplar_predictions.py -sd data/new/${LANG}/test.tsv -td ../zgul_plus/outputs/${LANG}_100_zgul_pred.tsv -tsim data/new/${LANG}/${LANG}_sim.npy
echo "Random ZGUL"
python save_exemplar_predictions.py -sd data/new/${LANG}/test.tsv -td ../zgul_plus/outputs/${LANG}_100_zgul_pred.tsv -tsim data/new/${LANG}/rand.npy
