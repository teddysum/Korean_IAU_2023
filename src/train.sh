CUDA_VISIBLE_DEVICES=2 python sa_for_iau.py \
  --train_data ../data/nikluge-iau-2023-train.jsonl \
  --dev_data ../data/nikluge-iau-2023-dev.jsonl \
  --base_model klue/roberta-base \
  --do_train \
  --do_eval \
  --learning_rate 3e-6 \
  --eps 1e-8 \
  --num_train_epochs 10 \
  --model_path ../saved_models/ \
  --batch_size 8 \
  --max_len 512