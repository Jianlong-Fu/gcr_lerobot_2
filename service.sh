python lerobot/scripts/pi0_service.py \
    --policy.type="pi0" \
    --dataset.repo_id="whatever" \
    --dataset.processor="/datassd_1T/qwen25vl/Qwen2.5-VL-7B-Instruct/" \
    --dataset.root="/data_16T/lerobot_openx/" \
    --data_mix="cup_full_plus" \
    --uni_res=true \
    --uni_obs_tensor=true
    
    # --dataset.root="/data_16T/lerobot_openx/" \
    # --data_mix="pizza"
    # --dataset.parent_dir="/data_16T/lerobot_openx/" \
    # --dataset.parent_dir="/data_16T/lerobot_openx/" \
    # --data_mix="pizza_single"
    