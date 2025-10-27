python lerobot/scripts/xvla_service.py \
    --policy.type="xvla" \
    --policy.encoder_name="/data_16T/deepseek/xvla_comp/Florence-2-large" \
    --dataset.repo_id="whatever" \
    --dataset.processor=none \
    --dataset.root="/data_16T/lerobot_openx/" \
    --data_mix="cup_full_plus" \
    --uni_res=true \
    --uni_obs_tensor=true
    
    # --dataset.root="/data_16T/lerobot_openx/" \
    # --data_mix="pizza"
    # --dataset.parent_dir="/data_16T/lerobot_openx/" \
    # --dataset.parent_dir="/data_16T/lerobot_openx/" \
    # --data_mix="pizza_single"
    