set -o errexit

epochs=$1
save_every=$2
beam_num=$3
generate_length=$4
model_size=$5
lr=$6
length=$7
cuda_num=$8
cudas=$9
batch_size=${10}
data_name=${11}
turn=${12}

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=4

# Preprocess the data
python data_process.py --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn $turn --lr $lr --data_name $data_name

# Finetune the first turn
python train.py --epochs $epochs --save_every $save_every --model_size $model_size --turn $turn --lr $lr --length $length --beam_num $beam_num --generate_length $generate_length --cudas $cudas --batch_size ${batch_size} --data_name ${data_name}

mode="test"
python generate.py --mode $mode --model_size $model_size --beam_num $beam_num --epochs $epochs --save_every $save_every --cuda ${cuda_num} --generate_length $generate_length --turn $turn --lr $lr --length $length --start_epoch 0 --end_epoch $epochs --data_name $data_name

python evaluate.py --mode $mode --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn $turn --lr $lr --data_name $data_name  --end_epoch $epochs --start_epoch 0

# Find the best model according to the valuation
mode="val"
python generate.py --mode $mode --model_size $model_size --beam_num $beam_num --epochs $epochs --save_every $save_every --cuda ${cuda_num} --generate_length $generate_length --turn $turn --lr $lr --length $length --start_epoch 0 --end_epoch $epochs --data_name $data_name

python evaluate.py --mode $mode --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn $turn --lr $lr --mode $mode --data_name $data_name --end_epoch $epochs --start_epoch 0

# Generate the train set based on the best model
mode="train"
python generate.py --mode $mode --model_size $model_size --beam_num $beam_num --epochs $epochs --save_every $save_every --cuda ${cuda_num} --generate_length $generate_length --turn $turn --lr $lr --length $length --start_epoch 0 --end_epoch $epochs --data_name $data_name

python evaluate.py --mode $mode --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn $turn --lr $lr --mode $mode --data_name $data_name --end_epoch $epochs --start_epoch 0

# Rewrite the dataset based on the best model

python rewrite.py --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --model_size $model_size --turn first --lr $lr --cuda_num $cuda_num --length $length --mode train --data_name $data_name 

python rewrite.py --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --model_size $model_size --turn first --lr $lr --cuda_num $cuda_num --length $length --mode val --data_name $data_name 

python rewrite.py --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --model_size $model_size --turn first --lr $lr --cuda_num $cuda_num --length $length --mode test --data_name $data_name 

# Preprocess the rewriting dataset
python data_process.py --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn rewrite --lr $lr --data_name $data_name

# Finetune again
python train.py --epochs $epochs --save_every $save_every --model_size $model_size --turn rewrite --lr $lr --length $length --beam_num $beam_num --generate_length $generate_length --cudas $cudas --batch_size ${batch_size} --data_name ${data_name}

python generate.py --mode test --model_size $model_size --beam_num $beam_num --epochs $epochs --save_every $save_every --cuda ${cuda_num} --generate_length $generate_length --turn rewrite --lr $lr --length $length --start_epoch 0 --end_epoch $epochs --data_name $data_name

python evaluate.py --mode test --model_size $model_size --epochs $epochs --save_every $save_every --beam_num $beam_num --generate_length $generate_length --turn rewrite --lr $lr --data_name $data_name --end_epoch $epochs --start_epoch 0

# The pipleline is over