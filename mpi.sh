torchrun --nnodes=1 \
    --nproc_per_node=2 \
    lerobot/scripts/fsdp_train.py \
    --policy.type="qwen" \
    --output_dir="/data_16T/deepseek/0529" \
    --save_freq=10000 \
    --dataset.image_transforms.enable=false \
    --batch_size=4 \
    --dataset.repo_id="whatever" \
    --dataset.processor="/datassd_1T/qwen25vl/Qwen2.5-VL-7B-Instruct/" \
    --dataset.parent_dir="/data_16T/lerobot_openx/" \
    --policy.scheduler_warmup_steps=1000 \
    --data_mix="pizza_single" \
    --policy.scheduler_platform_steps=10000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.optimizer_lr=1e-3 \
    --policy.train_main_layers=0 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.train_awa=true \
    --policy.train_expert=true \
    --policy.train_full_vlm=true
    # --resume=true
    

