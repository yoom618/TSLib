# the best performing model for all UEA30 dataset
# if there is more than one model, we choose the one with the lowest model size or computation cost
model_name="TSLANet"
tslib_dir="/data/username/TSLib"
gpu_id=0

data_dir="${tslib_dir}/dataset"
checkpoint_dir="${tslib_dir}/checkpoints_best/${model_name}"

# ArticularyWordRecognition
dataset_name="ArticularyWordRecognition"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "06_41_37" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 64 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 36

# AtrialFibrillation
dataset_name="AtrialFibrillation"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "06_32_01" \
    --model_id "CLS_${dataset_name}" \
    --depth 2 \
    --emb_dim 256 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 128

# BasicMotions
dataset_name="BasicMotions"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "08_01_51" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 32 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 8

# CharacterTrajectories
dataset_name="CharacterTrajectories"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "13_02_38" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 64 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 37

# Cricket
dataset_name="Cricket"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "09_55_19" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 32 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 30

# DuckDuckGeese
dataset_name="DuckDuckGeese"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "23_45_23" \
    --model_id "CLS_${dataset_name}" \
    --depth 2 \
    --emb_dim 32 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 14

# EigenWorms
dataset_name="EigenWorms"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "12_59_31" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 64 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 450

# Epilepsy
dataset_name="Epilepsy"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "05_43_08" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 128 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 11

# ERing
dataset_name="ERing"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "05_50_58" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 32 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 13

# EthanolConcentration
dataset_name="EthanolConcentration"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "06_32_27" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 64 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 44

# FaceDetection
dataset_name="FaceDetection"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "01_35_56" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 32 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 16

# FingerMovements
dataset_name="FingerMovements"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "09_53_00" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 64 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 10

# HandMovementDirection
dataset_name="HandMovementDirection"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "10_23_41" \
    --model_id "CLS_${dataset_name}" \
    --depth 2 \
    --emb_dim 256 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 80

# Handwriting
dataset_name="Handwriting"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "07_37_01" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 256 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 4

# Heartbeat
dataset_name="Heartbeat"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "18_01_32" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 128 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 11

# InsectWingbeat
dataset_name="InsectWingbeat"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "23_54_36" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 64 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 3

# JapaneseVowels
dataset_name="JapaneseVowels"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "20_46_55" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 64 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 3

# Libras
dataset_name="Libras"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "16_36_09" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 64 \
    --mlp_ratio 2.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 5

# LSST
dataset_name="LSST"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "14_50_43" \
    --model_id "CLS_${dataset_name}" \
    --depth 2 \
    --emb_dim 32 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 2

# MotorImagery
dataset_name="MotorImagery"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "03_59_06" \
    --model_id "CLS_${dataset_name}" \
    --depth 2 \
    --emb_dim 64 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 225

# NATOPS
dataset_name="NATOPS"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "20_29_41" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 32 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 2

# PEMS-SF
dataset_name="PEMS-SF"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "17_54_24" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 256 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 4

# PenDigits
dataset_name="PenDigits"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "20_29_43" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 32 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 2

# PhonemeSpectra
dataset_name="PhonemeSpectra"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "14_50_16" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 64 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 6

# RacketSports
dataset_name="RacketSports"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "16_26_54" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 32 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 3

# SelfRegulationSCP1
dataset_name="SelfRegulationSCP1"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "21_42_48" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 128 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 224

# SelfRegulationSCP2
dataset_name="SelfRegulationSCP2"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "21_01_05" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 128 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 116

# SpokenArabicDigits
dataset_name="SpokenArabicDigits"
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

# StandWalkJump
dataset_name="StandWalkJump"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "23_55_27" \
    --model_id "CLS_${dataset_name}" \
    --depth 3 \
    --emb_dim 32 \
    --mlp_ratio 1.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 188

# UWaveGestureLibrary
dataset_name="UWaveGestureLibrary"
python -u _run_TSLANet/TSLANet_classification_test.py \
    --gpu ${gpu_id} \
    --data_type uea \
    --data_path "${data_dir}/${dataset_name}" \
    --data_name ${dataset_name} \
    --ckpt_path ${checkpoint_dir} \
    --ckpt_time "20_18_28" \
    --model_id "CLS_${dataset_name}" \
    --depth 1 \
    --emb_dim 64 \
    --mlp_ratio 3.0 \
    --masking_ratio 0.4 \
    --ICB True \
    --ASB True \
    --adaptive_filter True \
    --load_from_pretrained True \
    --patch_size 79