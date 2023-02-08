set -o errexit

epochs=1
save_every=1
beam_num=1
generate_length=128
model_size="small"
lr="1e-5"
length=-1
cuda_num=4
cudas="0,1,2,3"
batch_size=1
data_name="Totto"
table="T"
tuning_mode=finetune
turn="first"

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=1,2,3,5

# 对数据进行预处理
python data_process.py --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn $turn --lr $lr --data_name $data_name --table $table

# 第一论微调
python train.py --epochs $epochs --save_every $save_every --model_size $model_size --turn $turn --lr $lr --length $length --beam_num $beam_num --generate_length $generate_length --cudas $cudas --batch_size ${batch_size} --data_name ${data_name} --table ${table} --tuning_mode ${tuning_mode}

mode="test"
python generate.py --mode $mode --model_size $model_size --beam_num $beam_num --epochs $epochs --save_every $save_every --cuda ${cuda_num} --generate_length $generate_length --turn $turn --lr $lr --length $length --start_epoch 0 --end_epoch $epochs --data_name $data_name --table ${table}

python evaluate.py --mode $mode --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn $turn --lr $lr --data_name $data_name --table ${table} --end_epoch $epochs --start_epoch 0

# 根据验证集找到最优的模型
mode="val"
python generate.py --mode $mode --model_size $model_size --beam_num $beam_num --epochs $epochs --save_every $save_every --cuda ${cuda_num} --generate_length $generate_length --turn $turn --lr $lr --length $length --start_epoch 0 --end_epoch $epochs --data_name $data_name --table ${table}

python evaluate.py --mode $mode --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn $turn --lr $lr --mode $mode --data_name $data_name --table ${table} --end_epoch $epochs --start_epoch 0

# 根据最优模型生成训练集
mode="train"
python generate.py --mode $mode --model_size $model_size --beam_num $beam_num --epochs $epochs --save_every $save_every --cuda ${cuda_num} --generate_length $generate_length --turn $turn --lr $lr --length $length --start_epoch 0 --end_epoch $epochs --data_name $data_name --table ${table}

python evaluate.py --mode $mode --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn $turn --lr $lr --mode $mode --data_name $data_name --table ${table} --end_epoch $epochs --start_epoch 0

# 根据最优模型进行所有数据集的重写

python rewrite.py --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --model_size $model_size --turn first --lr $lr --cuda_num $cuda_num --length $length --mode train --data_name $data_name --table ${table} 

python rewrite.py --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --model_size $model_size --turn first --lr $lr --cuda_num $cuda_num --length $length --mode val --data_name $data_name --table ${table} 

python rewrite.py --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --model_size $model_size --turn first --lr $lr --cuda_num $cuda_num --length $length --mode test --data_name $data_name --table ${table} 

# 重新对重写数据进行预处理
python data_process.py --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn rewrite --lr $lr --data_name $data_name --table ${table}

# 第二论微调
python train.py --epochs $epochs --save_every $save_every --model_size $model_size --turn rewrite --lr $lr --length $length --beam_num $beam_num --generate_length $generate_length --cudas $cudas --batch_size ${batch_size} --data_name ${data_name} --table ${table} --tuning_mode ${tuning_mode}

python generate.py --mode test --model_size $model_size --beam_num $beam_num --epochs $epochs --save_every $save_every --cuda ${cuda_num} --generate_length $generate_length --turn rewrite --lr $lr --length $length --start_epoch 0 --end_epoch $epochs --data_name $data_name --table ${table}

python evaluate.py --mode test --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn rewrite --lr $lr --mode $mode --data_name $data_name --table ${table} --end_epoch $epochs --start_epoch 0

echo "the pipleline is over"