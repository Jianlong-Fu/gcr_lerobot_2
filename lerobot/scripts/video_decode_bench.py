import os
import time

from tqdm import tqdm

from lerobot.common.datasets.video_utils import (
    decode_video_frames_torchvision,
    decode_video_frames_torchcodec,
    decode_video_frames_decord
)

if __name__ == "__main__":
    video_path = "/scratch/amlt_code/gcr_lerobot_2/robot_dataset/lerobot-format/aloha-taskbox_lerobot/videos/chunk-000/observation.images.cam_high/episode_000031.mp4"
    
    timestamps = [20.0]
    
    total_start = time.time()
    for i in tqdm(range(100)):
        start = time.time()
        frames = decode_video_frames_torchvision(video_path, timestamps, tolerance_s=1e-4, return_all=True)
        # print(f"torchvision: {time.time() - start}")

    print(f"torch vision total: {time.time() - total_start}")
      
    total_start = time.time()  
    for i in tqdm(range(100)):
        start = time.time()
        frames = decode_video_frames_torchcodec(video_path, timestamps, tolerance_s=1e-4, return_all=True)
        # print(f"torchcodec: {time.time() - start}")
        
    print(f"torch codec total: {time.time() - total_start}")
    
    # total_start = time.time()
    # for i in tqdm(range(100)):
    #     start = time.time()
    #     frames = decode_video_frames_decord(video_path, timestamps, tolerance_s=1e-4, return_all=True)
    #     # print(f"decord: {time.time() - start}")

    # print(f"decord total: {time.time() - total_start}")