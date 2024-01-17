if [ "$1" = "pororo" ]; then
  echo "Training on Pororo"
  DATA_DIR=../data/pororo_png
  OUTPUT_ROOT=./out/pororo/dalle-finetune-llava
  MODEL_CKPT='../ckpt/pororo/25_v1.pth'
  SENT_EMBED=512
  STORY_LEN=4
  LR=1e-4
  TRAIN_BS=1
  GRAD_ACC=4
elif [ "$1" = "flintstones" ]; then
  echo "Training on Flintstones"
  DATA_DIR=../data/flintstones
  OUTPUT_ROOT=./out/flintstones
  SENT_EMBED=512
  STORY_LEN=4
  LR=1e-5
  TRAIN_BS=1
  GRAD_ACC=4
elif [ "$1" = "didemo" ]; then
  echo "Training on DiDeMo"
  DATA_DIR=../data/didemo
  OUTPUT_ROOT=./out/didemo
  SENT_EMBED=512
  STORY_LEN=2
  TRAIN_BS=1
  GRAD_ACC=8
fi

LOG_DIR=../runs/dalle-finetune-llava

python ./train_t2i.py \
--model_name_or_path  $MODEL_CKPT \
--prefix_model_name_or_path './1.3B/' \
--tuning_mode story \
--dataset_name $1 \
--preseqlen 32 \
--condition \
--story_len $STORY_LEN \
--sent_embed $SENT_EMBED \
--prefix_dropout 0.2 \
--data_dir $DATA_DIR \
--dataloader_num_workers 4 \
--output_dir $OUTPUT_ROOT \
--log_dir $LOG_DIR \
--do_train --do_eval \
--per_gpu_train_batch_size $TRAIN_BS \
--per_gpu_eval_batch_size 2 \
--num_train_epochs 5 \
--gradient_accumulation_steps $GRAD_ACC \
--learning_rate $LR \
--logging_steps 50 \
--eval_steps 500 \
--generate_steps 1000 \
--background

