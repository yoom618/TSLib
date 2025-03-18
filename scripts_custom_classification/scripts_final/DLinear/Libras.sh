data_dir="/data/yoom618/TSLib/dataset"
checkpoint_dir="/data/yoom618/TSLib/checkpoints_best/DLinear"
gpu_id=0

# below all have the same performance

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Libras" \
  --seq_len 45 \
  --enc_in 2 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model DLinear \
  --model_id CLS_Libras \
  --moving_avg 21 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Libras" \
  --seq_len 45 \
  --enc_in 2 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model DLinear \
  --model_id CLS_Libras \
  --moving_avg 16 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Libras" \
  --seq_len 45 \
  --enc_in 2 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model DLinear \
  --model_id CLS_Libras \
  --moving_avg 14 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/Libras" \
  --seq_len 45 \
  --enc_in 2 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model DLinear \
  --model_id CLS_Libras \
  --moving_avg 12 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10