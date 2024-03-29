epochs="10840"
save_every="1084"
train_size="1084"
header_num="3"
cur_time=`date  +"%Y%m%d%H%M"`
model_name="345M"


sp_name="gpt2_run1"
run_name="${sp_name}_${model_name}_${save_every}_${epochs}"
echo ${run_name}
tmp_val_name="TD_val_${sp_name}_${model_name}_${save_every}_${epochs}"
tmp_test_name="TD_test_${sp_name}_${model_name}_${save_every}_${epochs}"
tmp_train_name="TD_rewrite_train_${sp_name}_${model_name}_${save_every}_${epochs}"

PYTHONPATH=src ./train_table.py --dataset ./afs/data/TD_train --model_name ${model_name} --run_name ${run_name} --save_every ${save_every} --epochs ${epochs}
python ./src/table_generate_conditional_samples.py --ckpt_dir ${run_name} --save_every ${save_every} --epochs ${epochs} --model_name ${model_name} \
            --data_dir TD_val_input --out_dir ${tmp_val_name} --header_num ${header_num}
PYTHONPATH=src ./beam_evaluate_rewrite.py --ckpt_dir ${run_name} --save_every ${save_every} --epochs ${epochs} --model_name ${model_name} \
            --data_train_dir TD_train_rewrite_input --data_train_gold_dir TD_train_rewrite_gold --out_train_dir ${tmp_train_name} --ori_path_val TD_val_gold \
            --gen_path ${tmp_val_name} --rewrite_train_dir ${rewrite_train} --train_size ${train_size} --data_val_dir ${tmp_val_name}\
            --data_test_dir TD_test_input --out_test_dir ${tmp_test_name} --stage first --data_rewrite_test_dir ${tmp_test_name} --ori_path_test TD_test_gold --table True --header_num ${header_num}
