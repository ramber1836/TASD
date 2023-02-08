set -o errexit

dataset=$1
epochs=$2
every=$3
learning_rate=$4
model_type=$5
start_epoch=$6
end_epoch=$7
export CUDA_VISIBLE_DEVICES=$8

#train
python gpt2-finetune.py \
    --epochs $epochs \
    --every $every \
    --learning_rate $learning_rate \
    --model_type ../../models/paddle/$dataset/$model_type \
    --checkpoint_path afs/$dataset/checkpoint/${model_type}_${epochs}_${every}_${learning_rate} \
    --table_data_path ../../data/$dataset/tokens_train.pkl \
    --input_data_path ../../data/$dataset/TD_train_input \
    --gold_data_path ../../data/$dataset/TD_train_gold

#evaluate
for mode in "val" "test" ; do
    for (( i=$start_epoch; i<$end_epoch; i+=$every ))
    do
        python gpt2-generate.py \
            --checkpoint_path afs/$dataset/checkpoint/${model_type}_${epochs}_${every}_${learning_rate}/$i \
            --table_data_path ../../data/$dataset/tokens_${mode}.pkl \
            --data_path ../../data/$dataset/TD_${mode}_input \
            --generate_path afs/$dataset/generated_result/${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.out
        
        python beam_evaluate.py \
            --groundtruth_path ../../data/$dataset/TD_${mode}_gold \
            --generate_path afs/$dataset/generated_result/${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.out \
            --evaluate_path afs/$dataset/evaluated_result/${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.metric
    done
done

#rewrite
for mode in "val" "test" "train" ; do
python rewrite.py \
    --checkpoint_path afs/$dataset/checkpoint/${model_type}_${epochs}_${every}_${learning_rate} \
    --table_data_path ../../data/$dataset/tokens_${mode}.pkl \
    --data_path ../../data/$dataset/TD_${mode}_input \
    --generated_path afs/$dataset/data/${model_type}_${epochs}_${every}_${learning_rate}/TD_${mode}_input \
    --evaluated_path afs/$dataset/evaluated_result/${model_type}_${epochs}_${every}_${learning_rate}
done

rewrite_data_path=${model_type}_${epochs}_${every}_${learning_rate}

#retrain
python gpt2-finetune.py \
    --epochs $epochs \
    --every $every \
    --learning_rate $learning_rate \
    --model_type ../../models/paddle/$dataset/$model_type \
    --checkpoint_path afs/$dataset/checkpoint/rewrite_${model_type}_${epochs}_${every}_${learning_rate} \
    --table_data_path ../../data/$dataset/tokens_train.pkl \
    --input_data_path afs/$dataset/data/${rewrite_data_path}/TD_train_input \
    --gold_data_path ../../data/$dataset/TD_train_gold

mode="test"

for (( i=$start_epoch; i<$end_epoch; i+=$every ))
do
    python gpt2-generate.py \
        --checkpoint_path afs/$dataset/checkpoint/rewrite_${model_type}_${epochs}_${every}_${learning_rate}/$i \
        --table_data_path ../../data/$dataset/tokens_${mode}.pkl \
        --data_path afs/$dataset/data/${rewrite_data_path}/TD_${mode}_input \
        --generate_path afs/$dataset/generated_result/rewrite_${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.out
    
    python beam_evaluate.py \
        --groundtruth_path ../../data/$dataset/TD_${mode}_gold \
        --generate_path afs/$dataset/generated_result/rewrite_${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.out \
        --evaluate_path afs/$dataset/evaluated_result/rewrite_${model_type}_${epochs}_${every}_${learning_rate}/$i/${mode}.metric
done