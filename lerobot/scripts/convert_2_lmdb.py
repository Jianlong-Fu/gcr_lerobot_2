import polars as pl
import numpy as np
import os

# --- 准备一个示例 Parquet 文件 ---
file_path = "/data_16T/lerobot_openx/bridge_orig_lerobot/merged.parquet"
dataset_dir = "/data_16T/lerobot_openx/bridge_orig_lerobot"
# sample_df = pl.DataFrame({
#     "id": range(100),
#     "category": [f"cat_{i % 5}" for i in range(100)],
#     "value": np.random.rand(100)
# })
# sample_df.write_parquet(file_path)
# ------------------------------------


# 1. 读取整个文件
print("--- 1. 读取整个文件 ---")
df = pl.read_parquet(file_path)
print(df[0]['action'][0].to_numpy())
# print(df.head())

# # 2. 只读取特定列
# print("\n--- 2. 只读取特定列 ---")
# df_subset = pl.read_parquet(file_path, columns=["episode_index", "frame_index"])
# print(df_subset.head())

# # # 3. 读取前5行
# print("\n--- 3. 读取前5行 ---")
# df_preview = pl.read_parquet(file_path, n_rows=150)
# print(df_preview)

# # 4. 读取多个文件 (使用通配符)
# #    Polars 会自动将它们拼接成一个 DataFrame
# sample_df.write_parquet("sample_data_2.parquet")
# print("\n--- 4. 读取多个文件 ---")
# df_all = pl.read_parquet("sample_data*.parquet")
# print(f"从多个文件读取的总行数: {len(df_all)}")