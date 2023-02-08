set -o errexit

epochs=31
save_every=1
beam_num=3
generate_length=128
model_size="large"
lr="1e-5"
length=-1
cuda_num=4
cudas="0"
batch_size=4
turn=first
data_name="Totto"
table="NT"
model_type=gpt2tasd
tuning_mode=finetune

# # 对数据进行预处理
# sh data_process/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr $data_name T

# 第一论微调
# sh train/run.sh $epochs $save_every $turn $model_size $lr $length $beam_num $generate_length $cudas ${batch_size} $data_name $table $model_type $tuning_mode
sh generate/run.sh $epochs $save_every $beam_num $generate_length $model_size $turn $lr test $length ${cuda_num} $data_name $table $model_type
sh bleu/run.sh $epochs $save_every $beam_num $generate_length $model_size $turn $lr test $data_name $table $model_type