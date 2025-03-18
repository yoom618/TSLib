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
  --root_path "${data_dir}/PEMS-SF" \
  --seq_len 144 \
  --enc_in 963 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  ...