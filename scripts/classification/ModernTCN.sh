python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/EthanolConcentration/ \
    --data UEA \
    --model_id EthanolConcentration \
    --model ModernTCN \
    --ffn_ratio 1 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 1 1 \
    --large_size 13 13 \
    --small_size 5 5 \
    --dims 32 64 \
    --head_dropout 0.0 \
    --class_dropout 0.0 \
    --dropout 0.1 \
    --itr 1 \
    --learning_rate 0.0005 \
    --batch_size 64 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/FaceDetection/ \
    --data UEA \
    --model_id FaceDetection \
    --model ModernTCN \
    --ffn_ratio 1 \
    --patch_size 32 \
    --patch_stride 16 \
    --num_blocks 1 1 1 \
    --large_size 9 9 9 \
    --small_size 5 5 5 \
    --dims 32 64 128 \
    --head_dropout 0.0 \
    --dropout 0.5 \
    --class_dropout 0.1 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/Handwriting/ \
    --data UEA \
    --model_id Handwriting \
    --model ModernTCN \
    --ffn_ratio 4 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 \
    --large_size 13 \
    --small_size 5 \
    --dims 256 \
    --head_dropout 0.0 \
    --dropout 0.1 \
    --class_dropout 0.1 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/Heartbeat/ \
    --data UEA \
    --model_id Heartbeat \
    --model ModernTCN \
    --ffn_ratio 1 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 1 \
    --large_size 31 \
    --small_size 5 \
    --dims 256 \
    --head_dropout 0.0 \
    --dropout 0.3 \
    --class_dropout 0.0 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/JapaneseVowels/ \
    --data UEA \
    --model_id JapaneseVowels \
    --model ModernTCN \
    --ffn_ratio 2 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 1 \
    --large_size 21 19 \
    --small_size 5 5 \
    --dims 256 512 \
    --head_dropout 0.0 \
    --dropout 0.5 \
    --class_dropout 0.1 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/PEMS-SF/ \
    --data UEA \
    --model_id PEMS-SF \
    --model ModernTCN \
    --ffn_ratio 4 \
    --patch_size 48 \
    --patch_stride 24 \
    --num_blocks 2 \
    --large_size 91 \
    --small_size 5 \
    --dims 32 \
    --head_dropout 0.0 \
    --dropout 0.3 \
    --class_dropout 0.7 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/SelfRegulationSCP1/ \
    --data UEA \
    --model_id SelfRegulationSCP1 \
    --model ModernTCN \
    --ffn_ratio 1 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 \
    --large_size 13 \
    --small_size 5 \
    --dims 32 \
    --head_dropout 0.0 \
    --dropout 0.1 \
    --class_dropout 0.1 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/SelfRegulationSCP2/ \
    --data UEA \
    --model_id SelfRegulationSCP2 \
    --model ModernTCN \
    --ffn_ratio 4 \
    --patch_size 32 \
    --patch_stride 16 \
    --num_blocks 1 1 \
    --large_size 51 49 \
    --small_size 5 5 \
    --dims 64 128 \
    --head_dropout 0.0 \
    --dropout 0.3 \
    --class_dropout 0.1 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/SpokenArabicDigits/ \
    --data UEA \
    --model_id SpokenArabicDigits \
    --model ModernTCN \
    --ffn_ratio 4 \
    --patch_size 16 \
    --patch_stride 16 \
    --num_blocks 1 1 1 \
    --large_size 1 1 1 \
    --small_size 5 5 5 \
    --dims 32 64 128 \
    --head_dropout 0.0 \
    --dropout 0.3 \
    --class_dropout 0.1 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False

python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./all_datasets/waveGesture/ \
    --data UEA \
    --model_id waveGesture \
    --model ModernTCN \
    --ffn_ratio 1 \
    --patch_size 1 \
    --patch_stride 1 \
    --num_blocks 1 1 \
    --large_size 51 49 \
    --small_size 5 5 \
    --dims 128 256 \
    --head_dropout 0.0 \
    --dropout 0.3 \
    --class_dropout 0.1 \
    --itr 1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --train_epochs 100 \
    --patience 10 \
    --des Exp \
    --use_multi_scale False