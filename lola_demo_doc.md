# LoLA — Model Inference Guide

This document explains how to set up the environment, load the LoLA model, and run inference. It also describes the expected input/output formats and common issues you may encounter.

---

## 0. Prerequisites

The necessary files required to run the demo can be downloaded with the following commands:

```bash
# Necessary Dataset Download:
azcopy cp https://azsussc.blob.core.windows.net/v-wangxiaofa/robot_dataset/lerobot-format/aloha-busybox_lerobot_v2_1_combined_new/ /path/to/your/workdir/ --recursive

# Qwen2.5VL 7B-Instruct Model Download:
azcopy cp https://azsussc.blob.core.windows.net/v-wangxiaofa/qwen_params/Qwen2.5-VL-7B-Instruct/ /path/to/your/workdir/ --recursive

# LoLA Model Download:
azcopy cp https://azsussc.blob.core.windows.net/v-wangxiaofa/original_qw/1106_bb-aloha-lola_set02/step2000.pt /path/to/your/workdir/
```

---

## 1. Repository & Conda environment

Clone the repository and create a Conda environment:

```bash
git clone https://github.com/NsurrenderX/gcr_lerobot_2.git && cd gcr_lerobot_2
conda create -n lola python=3.10 -y
conda activate lola
```

---

## 2. Install required packages

Install system-level and Python dependencies (versions used during development are pinned below):

```bash
# FFmpeg
conda install -c conda-forge ffmpeg==7.1.1 -y

# Install package in editable mode with the optional pi0 extras
pip install -e ".[pi0]"

# Additional Python packages
pip install pytest polars qwen_vl_utils timm bitsandbytes scipy tabulate datasets==3.6.0
pip install torch==2.7.0 torchvision==0.22.0 torchcodec==0.5.0
pip install flash-attn --no-build-isolation
pip install numpy==1.26.4 transformers==4.51.3
```

> Note: these versions are what the demo expects. If you change versions, you may need to resolve incompatibilities (especially for `torch`, `torchvision`, `flash-attn` and `ffmpeg`).

---

## 3. Run the demo

1. Launch the background service used by the demo:

```bash
bash lola_service.sh
```

2. Run the demo script that loads the LoLA model and generates a demo action sequence:

```bash
python lerobot/scripts/lola_demo.py
```

The demo script (`lerobot/scripts/lola_demo.py`) includes a minimal example of preparing inputs and running the model.

---

## 4. Configuration variables to edit

Before running the demo, update the following variables in `lerobot/scripts/lola_demo.py`:

- `path_2_load`: Path to the LoLA model file (usually ends with `.pt`).
- `cfg.policy.qwen_path`: Path to the Qwen2.5-VL-7B-Instruct model directory (the original VLM checkpoint).
- `device`: Device where the model should be loaded (default: `cuda:0`).

Also check `lola_service.sh` and update these variables as needed:

- `dataset.processor`: Path to the Qwen2.5-VL-7B-Instruct model directory (same as `cfg.policy.qwen_path`).
- `dataset.parent_dir`: Directory that contains the dataset used by the service.

---

## 5. Input format (state & images)

The demo builds a `simulation_data` dictionary similar to the following:

```python
simulation_data = {
    "observation.state": torch.ones(32).to(dtype=torch.float32),
    "mean": state_mean,
    "std": state_std,
    "task": "Pick up the apple.",
}
```

### Robot state tensor (32 elements)

The `observation.state` tensor must follow this layout (indices shown):

- `LEFT_POS`: indices `[0, 1, 2]`
- `LEFT_QUAT`: indices `[3, 4, 5, 6]` (quaternion)
- `LEFT_GRIPPER`: index `[7]`
- `RIGHT_POS`: indices `[8, 9, 10]`
- `RIGHT_QUAT`: indices `[11, 12, 13, 14]` (quaternion)
- `RIGHT_GRIPPER`: index `[15]`
- `ZERO_PADDING`: indices `[16 .. 31]` (reserved / zero padding)

Replace the demo `torch.ones(32)` with your robot's real-time state vector when using LoLA in a real system.

---

## 6. Model output (action sequence)

When you run inference, the model returns an action sequence of shape `(50, 14)`:

- `50` — length of the action sequence (timesteps)
- `14` — dimensionality of each action vector

Action vector layout (size 14):

- `LEFT_POS`: `[0, 1, 2]`
- `LEFT_EULER`: `[3, 4, 5]` (Euler angles)
- `LEFT_GRIPPER`: `[6]`
- `RIGHT_POS`: `[7, 8, 9]`
- `RIGHT_EULER`: `[10, 11, 12]` (Euler angles)
- `RIGHT_GRIPPER`: `[13]`

> Note: The model outputs Euler angles for actions, while the input state uses quaternions for orientation. Convert formats as required by your controller.

---

## 7. Image views mapping

LoLA expects views keyed like this (as used in the `aloha-bb` dataset):

- `primary`: the `cam_high` view
- `secondary`: the `left_wrist` view
- `wrist`: the `right_wrist` view 

---

## 8. Minor issues and expected log messages

When loading the model you may see repeated log lines such as:

```
Training action expert
Training awa model
Training full vlm
Training vision encoder
training from scratch
```

These are printed because the model is initialized with default training settings. This is expected and harmless: the demo calls `lola.eval()` so it does **not** perform training during inference.

---


