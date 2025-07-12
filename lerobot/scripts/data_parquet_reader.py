import os
import glob
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import imageio

parquet_path = "/data_16T/lerobot_openx/all_aloha_including_instrumented_april_18/data/chunk-000/"
videopath = "/home/v-wenhuitan/pi_0_open/media/aloha_index.mp4"

parquet_files = os.listdir(parquet_path)

# file = os.path.join(parquet_path, parquet_files[0])
# df = pd.read_parquet(file)
# data = df.loc[1]
# for key, value in data.items():
#     print(key)

task_list = []

for file in tqdm(parquet_files):
    
    parquet_path_file = os.path.join(parquet_path, file)
    index = int(file.split("_")[1].split(".")[0])
    
    df = pd.read_parquet(parquet_path_file)
    frames = []

    for idx in df.index:
        row_data = df.loc[idx]
        task_info = row_data["task_index"]
        if task_info not in task_list:
            task_list.append(task_info)
            
parquet_path = "/data_16T/lerobot_openx/all_aloha_including_instrumented_april_18/data/chunk-001/"
parquet_files = os.listdir(parquet_path)
for file in tqdm(parquet_files):
    
    parquet_path_file = os.path.join(parquet_path, file)
    index = int(file.split("_")[1].split(".")[0])
    
    df = pd.read_parquet(parquet_path_file)
    frames = []

    for idx in df.index:
        row_data = df.loc[idx]
        task_info = row_data["task_index"]
        if task_info not in task_list:
            task_list.append(task_info)

print(task_list)
print(f"Total number of tasks: {len(task_list)}")

        # sample_img = row_data["observation.images.cam_high"]
        # np_image = np.frombuffer(sample_img["bytes"], np.uint8)
        # np_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        # frames.append(np_image)
        
    # width, height, channels = frames[0].shape
    # fps=30
    
    # actual_videopath = videopath.replace("index", str(index))

    # with imageio.get_writer(actual_videopath, fps=fps) as writer:
    #     for frame in frames:
    #         writer.append_data(frame)