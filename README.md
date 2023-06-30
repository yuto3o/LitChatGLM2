# LitChatGLM2
Fine-tuning 🤖[ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) with 🤗[HF-LoRA](https://github.com/huggingface/peft) and ⚡[Lightning](https://github.com/Lightning-AI/lightning)

## Data
```json lines
{"instruction": "将以下句子从一种时态转化成另一种时态", "input": "他正在往商店走", "output": "他曾经往商店走过"}
{"instruction": "针对产品发布提出五种营销策略。", "input": "", "output": "1. 社交媒体活动。\n2. 电子邮件营销。\n3. 在线和离线广告。\n4. 推荐和评论。\n5. 合作名人推销。"}
...
```

## Train
set `model.bits` to 4 for QLoRA
```shell
CUDA_VISIBLE_DEVICES=0 python run_chatglm2.py fit \
    --model.pretrained_model_name_or_path THUDM/chatglm2-6b \
    --model.bits 4 \
    --model.learning_rate 2e-5 \
    --data.train_batch_size 16 \
    --data.eval_batch_size 16 \
    --data.max_source_length  256 \
    --data.max_target_length  256 \
    --data.train_data_path PATH_TO_TRAIN_DATA \
    --data.val_data_path PATH_TO_VAL_DATA \
    --trainer.max_epoch 10 \
    --trainer.val_check_interval 1000 \
    --trainer.accumulate_grad_batches 1 \
    --trainer.precision bf16-mixed
  
```

## Infer
```shell
CUDA_VISIBLE_DEVICES=1 python generate.py \
    --llm_model_file PATH_TO_LLM \
    --peft_model_file PATH_TO_LORA \
    --inp_file PATH_TO_INPUT_FILE \
    --out_file PATH_TO_OUTPUT_FILE
```
set `ref_file` and `k_shot` for simple K-Shot in-context learning