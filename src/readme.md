运行流程
python train_phase1.py \
    --device 0 \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --learning_rate 5e-4 \
    --output_dir ../output_dir/phase1


python train_phase2.py \
    --device 0 \
    --perception_checkpoint ../output_dir/phase1-0.5/perception_best.pth \
    --qwen_path  \
    --num_train_epochs 5 \
    --train_batch_size 8 \
    --output_dir ../output_dir/phase2

python train_phase3_gen.py \
    --device 3 \
    --perception_checkpoint ../output_dir/phase1/perception_best.pth \
    --projector_checkpoint ../output_dir/phase2/projector_aligned.pth \
    --qwen_path  \
    --num_train_epochs 5 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir ../output_dir/phase3

python train_phase3_dpo.py \
  --device 3 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_workers 4 \
  --qwen_path  \
  --lora_shared_path ../output_dir/phase3/lora_adapter \
  --perception_checkpoint ../output_dir/phase1/perception_best.pth \
  --projector_checkpoint ../output_dir/phase2/projector_aligned.pth \
  --train_reasoning_file ./MMSD2.0dataset/data/text_json_final/train_reasoning_with_negatives.json \
  --valid_reasoning_file ./MMSD2.0dataset/data/text_json_final/valid_reasoning_with_negatives.json \
  --output_dir ../output_dir/phase3_dpo \
  --num_train_epochs 5 --beta 0.1


python train_phase4_end2end.py --num_train_epochs 5 --device 3 --lora_shared_path '../output_dir/phase3/lora_adapter'

