set -o errexit

dataset="numericNLG"

epochs=30
every=3
learning_rate=1e-6
model_type=gpt2-en

export CUDA_VISIBLE_DEVICES=2

# #train
# python gpt2-finetune.py \
#     --epochs $epochs \
#     --every $every \
#     --learning_rate $learning_rate \
#     --model_type afs/$dataset/models/$model_type \
#     --checkpoint_path afs/$dataset/checkpoint/${model_type}_${epochs}_${every}_${learning_rate} \
#     --pickle_file_path afs/$dataset/data/table_data/tokens_train.pkl \
#     --input_data_path afs/$dataset/data/origin/TD_train_input \
#     --gold_data_path afs/$dataset/data/origin/TD_train_gold

#evaluate
for mode in "val" ; do
    start_epoch=$every-1
    end_epoch=30
    for (( i=$start_epoch; i<$end_epoch; i+=$every ))
    do
        python gpt2-generate.py \
            --checkpoint_path afs/$dataset/checkpoint/${model_type}_${epochs}_${every}_${learning_rate}/$i \
            --pickle_file_path afs/$dataset/data/table_data/tokens_${mode}.pkl \
            --data_path afs/$dataset/data/origin/TD_${mode}_input \
            --generate_path afs/$dataset/generated_result/${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.out
        
        python beam_evaluate.py \
            --groundtruth_path afs/$dataset/data/origin/TD_${mode}_gold \
            --generate_path afs/$dataset/generated_result/${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.out \
            --evaluate_path afs/$dataset/evaluated_result/${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.metric
    done
done

#rewrite
for mode in "val" "test" "train" ; do
python rewrite.py \
    --checkpoint_path afs/$dataset/checkpoint/${model_type}_${epochs}_${every}_${learning_rate} \
    --pickle_file_path afs/$dataset/data/table_data/tokens_${mode}.pkl \
    --data_path afs/$dataset/data/origin/TD_${mode}_input \
    --generated_path afs/$dataset/data/${model_type}_${epochs}_${every}_${learning_rate}/TD_${mode}_input \
    --evaluated_path afs/$dataset/evaluated_result/${model_type}_${epochs}_${every}_${learning_rate}

done

rewrite_data_path=${model_type}_${epochs}_${every}_${learning_rate}

epoch=10
#retrain
python gpt2-finetune.py \
    --epochs $epoch \
    --every $every \
    --learning_rate $learning_rate \
    --model_type $model_type \
    --checkpoint_path afs/$dataset/checkpoint/rewrite_${model_type}_${epochs}_${every}_${learning_rate} \
    --pickle_file_path afs/$dataset/tokens_table_data.pkl \
    --input_data_path afs/$dataset/data/${rewrite_data_path}/TD_train_input \
    --gold_data_path afs/$dataset/data/origin/TD_train_gold

mode="test"
start_epoch=0
end_epoch=30
for (( i=$start_epoch; i<$end_epoch; i+=$every ))
do
    python gpt2-generate.py \
        --checkpoint_path afs/$dataset/checkpoint/rewrite_${model_type}_${epochs}_${every}_${learning_rate}/$i \
        --pickle_file_path afs/$dataset/tokens_table_data.pkl \
        --data_path afs/$dataset/data/${rewrite_data_path}/TD_${mode}_input \
        --generate_path afs/$dataset/generated_result/rewrite_${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.out
    
    python beam_evaluate.py \
        --groundtruth_path afs/$dataset/data/${rewrite_data_path}/TD_${mode}_gold \
        --generate_path afs/$dataset/generated_result/rewrite_${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.out \
        --evaluate_path afs/$dataset/evaluated_result/rewrite_${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.metric
done