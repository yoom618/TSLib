data_dir="/data/yoom618/TSLib/dataset"
checkpoint_dir="/data/yoom618/TSLib/checkpoints_best/DLinear"
gpu_id=0

python run.py \
  --use_gpu True \
  --gpu_type cuda \
  --gpu ${gpu_id} \
  --task_name classification \
  --data UEA \
  --root_path "${data_dir}/SelfRegulationSCP1" \
  --seq_len 896 \
  --enc_in 6 \
  --label_len 0 \
  --pred_len 0 \
  --c_out 0 \
  --checkpoints ${checkpoint_dir} \
  --model DLinear \
  --model_id CLS_SelfRegulationSCP1 \
  --moving_avg 448 \
  --is_training 0 \
  --batch_size 16 \
  --des Exp \
  --itr 1 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10