model_name="TSLANet"
dataset_name="SpokenArabicDigits"
tslib_dir="/data/username/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"

python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "04_15_53" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 128 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 14
