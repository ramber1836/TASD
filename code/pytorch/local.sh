set -o errexit

# epochs=2
# save_every=1
# beam_num=1
# generate_length=10
# model_size="medium"
# lr="1e-5"
# length=4
# cuda_num=4
# cudas="0,1,2,3"
# batch_size=4
# data_name="Totto"
# table="NT"

epochs=31
save_every=1
beam_num=2
generate_length=256
model_size="medium"
lr="1e-5"
length=-1
cuda_num=4
cudas="0,1,2,3"
batch_size=4
data_name="Totto"
table="NT"


# # 对数据进行预处理
# sh data_process/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr $data_name $table

# # 第一论微调
# sh train/run.sh $epochs $save_every first $model_size $lr $length $beam_num $generate_length $cudas ${batch_size} $data_name $table
# sh generate/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr test $length ${cuda_num} $data_name $table
# sh bleu/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr test $data_name $table

# # 根据验证集找到最优的模型
# sh generate/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr val $length ${cuda_num} $data_name $table
# sh bleu/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr val $data_name $table


# # # 根据最优模型生成训练集
# sh generate/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr train $length ${cuda_num} $data_name $table
# sh bleu/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr train $data_name $table

# # # 根据最优模型进行所有数据集的重写
# sh rewrite/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr $length train ${cuda_num} $data_name $table
# sh rewrite/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr $length val ${cuda_num} $data_name $table
# sh rewrite/run.sh $epochs $save_every $beam_num $generate_length $model_size first $lr $length test ${cuda_num} $data_name $table

# # # 重新对重写数据进行预处理
# sh data_process/run.sh $epochs $save_every $beam_num $generate_length $model_size rewrite $lr $data_name $table

# # 第二论微调
# sh train/run.sh $epochs $save_every rewrite $model_size $lr $length $beam_num $generate_length $cudas ${batch_size} $data_name $table
sh generate/run.sh $epochs $save_every $beam_num $generate_length $model_size rewrite $lr test $length ${cuda_num} $data_name $table
sh bleu/run.sh $epochs $save_every $beam_num $generate_length $model_size rewrite $lr test $data_name $table
echo "the pipleline is over"