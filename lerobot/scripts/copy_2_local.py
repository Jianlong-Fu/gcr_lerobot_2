import os
import shutil

source = "/mnt/wangxiaofa/robot_dataset/lerobot-format/aloha-taskbox_lerobot/"

dst = "/scratch/amlt_code/gcr_lerobot_2/robot_dataset/lerobot-format/aloha-taskbox_lerobot/"

shutil.copytree(source, dst, dirs_exist_ok=True)