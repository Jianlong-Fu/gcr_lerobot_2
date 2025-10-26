torchrun --nnodes=1 \
    --nproc_per_node=2 \
    lerobot/scripts/fsdp_4_xvla.py \
    --policy.type="xvla" \
    --policy.encoder_name="/data_16T/deepseek/xvla_comp/Florence-2-large" \
    --output_dir="/data_16T/deepseek/1026" \
    --save_freq=5000 \
    --dataset.image_transforms.enable=false \
    --dataset.processor=none \
    --batch_size=16 \
    --gradient_accumulation_steps=1 \
    --dataset.repo_id="whatever" \
    --dataset.root="/data_16T/lerobot_openx/" \
    --policy.scheduler_warmup_steps=10 \
    --data_mix="cup_full_plus" \
    --policy.scheduler_platform_steps=10000 \
    --policy.scheduler_decay_steps=100000 \
    --policy.optimizer_lr=1e-3  \
    --uni_res=true \
    --uni_obs_tensor=true
    # --resume=true
    

