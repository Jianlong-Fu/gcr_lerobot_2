import os
import cv2
import torch
import json
import time
import base64
import requests
import transformers
import numpy as np

from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request
from qwen_vl_utils import process_vision_info

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.utils import cycle
from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining

def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector

def dict_json_serlizable(d:dict):
    #convert all numpy ndarray in dict to list recursively
    for k,v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
            print("converted numpy ndarray:{} to list".format(k))
        elif isinstance(v, dict):
            dict_json_serlizable(v)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], np.ndarray):
                    v[i] = v[i].tolist()
                    print("converted numpy ndarray: {} to Obj.list".format(k))
    return d

def numpy_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # Convert NumPy array to a list
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def decode_b64_image(b64image):
    # Decode the base64 image string
    str_decode = base64.b64decode(b64image)
    np_image = np.frombuffer(str_decode, np.uint8)
    image_cv2 = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    # Get h,w of image
    h, w, _ = image_cv2.shape
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    #Crop the image retain the mid area to be a square
    # if h > w:
    #     image_cv2 = image_cv2[int((h-w)/2):int((h+w)/2), :, :]
    # elif w > h:
    #     image_cv2 = image_cv2[:, int((w-h)/2):int((w+h)/2), :]
    return image_cv2

def prepare_images(item: dict):
    vision = {
        "image": [],
        "video": None
    }
    
    assert "exp_id" in item, "item must contain exp_id"
    
    all_image_keys = ["primary"]
    present_img_keys = [key for key in all_image_keys if key in item]
    if len(present_img_keys) == 0:
        raise ValueError(
            f"item must contain at least one of the following keys: {all_image_keys}"
        )
    if item["exp_id"] in image_pool:
        image_pool[item["exp_id"]].append(item["primary"])
    else:
        image_pool[item["exp_id"]] = [item["primary"]]
    vision["image"].append(item["primary"][-1])
        
    video = item["primary"]
    video_length = len(video)
    for i in range(video_length):
        video[i] = video[i].resize((112, 112))
    
    vision["image"].append(item["secondary"])
    vision["image"].append(item["wrist"])
    default_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8))
    # vision["image"].append(default_image)
    # vision["image"].append(default_image)
    vision["video"] = video
    
    return vision

def prepare_language(vision, processor, item):
    def apply_template(text, vision=None):
        message = [
            {"role": "user",
            "content": [
                
            ],}
        ]
        if "video" in vision.keys():
            
            if vision["video"] is not None:
                message[0]["content"].append(
                    {
                        "type": "video",
                        "video": vision["video"],
                    },
                )
            
        for i in range(len(vision["image"])):
            message[0]["content"].append(
                {
                    "type": "image",
                    "image": vision["image"][i],
                }
            )
        message[0]["content"].append({"type": "text", "text": text})
        return processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
    text = item["task"]
    
    template = apply_template(text, vision)
    
    return template

def prepare_data(item, processor):
    
    vision = prepare_images(item)
    task = prepare_language(vision, processor, item)
    
    text = item["task"]
    
    message = [
        {
            "role": "user",
            "content": []
        }
    ]
        
    video = vision["video"]
    if video is not None:
        message[0]["content"].append(
            {
                "type": "video",
                "video": video,
            },
        )
    for i in range(len(vision["image"])):
        message[0]["content"].append(
            {
                "type": "image",
                "image": vision["image"][i],
            }
        )
    message[0]["content"].append({"type": "text", "text": text})
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        
    inputs = processor(
        text=task,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors = "pt",
        **video_kwargs,
    )
    
    input_ids = getattr(inputs, "input_ids", None)
    attention_mask = getattr(inputs, "attention_mask", None)
    pixel_values = getattr(inputs, "pixel_values", None)
    image_grid_thw = getattr(inputs, "image_grid_thw", None)
    pixel_values_videos = getattr(inputs, "pixel_values_videos", None)
    video_grid_thw = getattr(inputs, "video_grid_thw", None)
    second_per_grid_ts = getattr(inputs, "second_per_grid_ts", None)
    
    return_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
        "second_per_grid_ts": second_per_grid_ts,
    }

    return return_dict

def prepare_input(item, processor):
    vl_item = prepare_data(item, processor)
    
    item["observation.state"] = pad_vector(item["observation.state"], 32)
    item["observation.state"] = (item["observation.state"] - item["mean"]) / (item["std"] + 1e-8)
    state = torch.zeros_like(item["observation.state"])
    state_device = item["observation.state"].device
    state_dtype = item["observation.state"].dtype
    state[:8] = item["observation.state"][:8]
    state = state.to(device = state_device, dtype = state_dtype)
    item["observation.state"] = state
    
    data_dict = {
        "observation.state": item["observation.state"].unsqueeze(0),
        "action": None,
        **vl_item,
    }
    
    for k,v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(device)
        elif isinstance(v, list):
            if len(v) > 0:
                if isinstance(v[0], torch.Tensor):
                    device_list = [v[i].to(device) for i in range(len(v))]
                    data_dict[k] = device_list
    
    print("\n-------------------------------------------\n")
    print(f"input_ids: {data_dict['input_ids'].shape}")
    print(f"attention_mask: {data_dict['attention_mask'].shape}")
    print(f"pixel_values: {data_dict['pixel_values'].shape}")
    print(f"image_grid_thw: {data_dict['image_grid_thw']}")
    print(f"pixel_values_videos: {data_dict['pixel_values_videos'].shape}")
    print(f"video_grid_thw: {data_dict['video_grid_thw']}")
    print(f"second_per_grid_ts: {data_dict['second_per_grid_ts']}")

    return data_dict

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Cuda realtime vla model inference service!'

@app.route('/predict', methods=['POST'])
def predict():
    # task: str
    # image(s): in list type of base64 encoded string
    # exp_id: str
    # current state: list[float]

    # process the images into CV2.image or IMAGE.image
    # rerun the prepare_data process to get the desired input
    # global resource pool: image list stored in different exp_ids
    # global resource pool: processor
    global device, image_pool, processor, halo, state_mean, state_std, action_mean, action_std
    
    resp = request.get_json()
    
    item = {
        "observation.state": torch.tensor(resp["state"]),
        "mean": state_mean,
        "std": state_std,
        "task": resp["task"],
        "exp_id": resp["exp_id"]
    }
    
    images = resp['images'][0]
    item["primary"] = []
    for img in images:
        image_k4a_1 = decode_b64_image(img)
        image_k4a_1 = image_k4a_1[40:720,200:880,:] 
        image_k4a_1 = Image.fromarray(image_k4a_1).resize((224,224))
        # save image for visulization
        image_k4a_1.save("k4a.jpg")
        item["primary"].append(image_k4a_1)
    item["secondary"] = decode_b64_image(resp['images'][1])
    # item["secondary"] = item["secondary"][40:720,200:880,:]
    item["secondary"] = Image.fromarray(item["secondary"]).resize((224,224))
    item["secondary"].save("real_1.jpg")
    
    item["wrist"] = decode_b64_image(resp['images'][2])
    wrist_shape = item["wrist"].shape
    
    if wrist_shape[0] > wrist_shape[1]:
        # center crop
        item["wrist"] = item["wrist"][:,wrist_shape[1]//2-wrist_shape[0]//2:wrist_shape[1]//2+wrist_shape[0]//2,:]
    else:
        item["wrist"] = item["wrist"][wrist_shape[0]//2-wrist_shape[1]//2:wrist_shape[0]//2+wrist_shape[1]//2,:,:]
    item["wrist"] = Image.fromarray(item["wrist"]).resize((224,224))
    item["wrist"].save("real_2.jpg")
    
    input = prepare_input(item, processor)
    input["action.mean"] = action_mean
    input["action.std"] = action_std
    
    actions = halo.infer(input).tolist() # 1 * 50 *32
    # actions = actions[0] # 50 * 32
    actions = [row[:14] for row in actions[0]] # 50 * 7 eef pose
    # actions = [row[6:14] for row in actions[0]] # 50 * 7 joint
    print(actions[:5])
    
    response_dict = {
        "act": actions
    }
    return jsonify(response_dict)
# load qwen2.5vl's vl processor when calling __init__

@app.route('/modelbench', methods=['GET'])
def modelbench():
    global dataset
    benchloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    loader_cycler = cycle(benchloader)
    batch = next(loader_cycler)
    for k, v in batch.items():
        print(k)
    
    actions = halo.infer(batch).tolist() # 1 * 50 *32
    print(actions)
    return 'Model benchmarking service!'

@parser.wrap()
def start_service(cfg: TrainPipelineConfig):
    
    path_2_load = "/data_16T/deepseek/halo/step20000.pt"
    cfg.policy.qwen_path = "/datassd_1T/qwen25vl/Qwen2.5-VL-7B-Instruct/"
    
    image_transforms = (ImageTransforms(cfg.dataset.image_transforms))
    seed = cfg.seed
    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        image_transforms=image_transforms,
        seed=seed,
        data_mix=cfg.data_mix,
        vla2root_json="pizza.json",
        # vla2root_json="vla2root_bak_single.json"
    )
    action_mean = dataset.stats["action"]["mean"]
    action_std = dataset.stats["action"]["std"]
    print(action_mean, action_std)
    # action_mean[6] = 0.0
    # action_std[6] = 1.0
    
    state_mean = dataset.stats["observation.state"]["mean"]
    state_std = dataset.stats["observation.state"]["std"]
    
    processor = dataset.processor
    
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path
    )
    
    if path_2_load:
        model_state_dict = torch.load(path_2_load, map_location="cpu")
        key_to_remove = []
        for k, v in model_state_dict.items():
            if "awa_model.lm_head" in k or "qwen_expert.lm_head" in k:
                key_to_remove.append(k)
        for k in key_to_remove:
            del model_state_dict[k]
            
        policy.load_state_dict(model_state_dict, strict=True)
        del model_state_dict
        del key_to_remove
    
    for params in policy.parameters():
        params.data = params.data.bfloat16()
        
    halo = policy
    halo.eval()

    halo.to(device="cuda:0")
    
    device = "cuda:0"
    
    return device, halo, processor, state_mean, state_std, action_mean, action_std, dataset

if __name__ == '__main__':
    device, halo, processor, state_mean, state_std, action_mean, action_std, dataset = start_service()
    image_pool = {}
    app.run(host='0.0.0.0', port=7777)