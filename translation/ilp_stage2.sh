OUTPUT_DIR=$1
cp stats_ner.py $OUTPUT_DIR
cd $OUTPUT_DIR
source activate <env_path> 
for LANG in hau ibo kin lug luo
    do
    python stats_ner.py $LANG 80
    done
cd ~/llm/emnlp/
cd <pwd>
conda deactivate
source activate <env_path>
bash run_ilp_ner.sh 1.0 $OUTPUT_DIR
cd ~/llm/emnlp/
conda deactivate
source activate <env_path>
bash run_gpt4.sh $OUTPUT_DIR
