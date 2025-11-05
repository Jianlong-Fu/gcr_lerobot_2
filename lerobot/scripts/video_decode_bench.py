import os
import time
import numpy as np

from tqdm import tqdm

from lerobot.common.datasets.video_utils import (
    decode_video_frames_torchvision,
    decode_video_frames_torchcodec
)

if __name__ == "__main__":
    video_path = "/data_16T/lerobot_openx/aloha-taskbox_lerobot/videos/chunk-000/observation.images.cam_high/episode_000031.mp4"
    # video_path = "/scratch/amlt_code/gcr_lerobot_2/robot_dataset/lerobot-format/aloha-taskbox_lerobot/videos/chunk-000/observation.images.cam_high/episode_000031.mp4"
    rng = np.random.default_rng()
    diff = []
    total_start = time.time() 
    timestamps = [17.6500]
    for i in range(10):
        start = time.time()
        frames = decode_video_frames_torchcodec(video_path, timestamps, tolerance_s=1e-4, return_all=True, worker_count=16, return_type='numpy')
        print(f"torchcodec: {time.time() - start}")
    print(f"torch codec total: {time.time() - total_start}")
    print(frames[0].shape)
    
    total_start = time.time()
    for i in range(10):
        start = time.time()
        frames = decode_video_frames_torchvision(video_path, timestamps, tolerance_s=1e-4, return_all=True, return_type='numpy')
        print(f"torchvision: {time.time() - start}")
    print(frames[0].shape)
    # torchvision_length = len(frames)
    print(f"torch vision total: {time.time() - total_start}")
        
