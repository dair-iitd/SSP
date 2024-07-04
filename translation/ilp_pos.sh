OUTPUT_DIR=$1
cp stats_pos.py $OUTPUT_DIR
cd $OUTPUT_DIR
source activate <env_path> 
for LANG in fo got gsw
    do
    python stats_pos.py $LANG 80
    done
cd ~/llm/emnlp/
cd <pwd>
conda deactivate
source activate <env_path>
bash run_ilp_pos.sh 100.0 $OUTPUT_DIR
