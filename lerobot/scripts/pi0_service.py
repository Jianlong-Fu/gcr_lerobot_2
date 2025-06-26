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
from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining

def pil2tensor(pil_image):
    np_pil_image = np.array(pil_image).astype(np.float32)
    np_pil_image = np_pil_image.transpose((2, 0, 1)) / 255.0
    tensor_image = torch.from_numpy(np_pil_image)
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image
    

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
    # h, w, _ = image_cv2.shape
    # #Crop the image retain the mid area to be a square
    # if h > w:
    #     image_cv2 = image_cv2[int((h-w)/2):int((h+w)/2), :, :]
    # elif w > h:
    #     image_cv2 = image_cv2[:, int((w-h)/2):int((w+h)/2), :]
    return image_cv2

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
    global device, processor, pi0, state_mean, state_std, action_mean, action_std
    
    resp = request.get_json()
    
    item = {
        "observation.state": torch.tensor(resp["state"]),
        "task": [resp["task"]],
        "exp_id": resp["exp_id"]
    }
    
    images = resp['images'][0]
    item["primary"] = []
    for img in images:
        image_k4a_1 = decode_b64_image(img)
        image_k4a_1 = image_k4a_1[40:720,200:880,:] 
        image_k4a_1 = Image.fromarray(image_k4a_1).resize((224,224))
        item["primary"].append(image_k4a_1)
        
    item["observation.images.primary"] = item["primary"][-1]
    item["observation.images.primary"] = pil2tensor(item["observation.images.primary"])
    
    item["observation.images.secondary"] = decode_b64_image(resp['images'][1])
    item["observation.images.secondary"] = Image.fromarray(item["observation.images.secondary"]).resize((224,224))
    item["observation.images.secondary"] = pil2tensor(item["observation.images.secondary"])
    
    item["observation.images.wrist"] = decode_b64_image(resp['images'][2])
    wrist_shape = item["observation.images.wrist"].shape
    if wrist_shape[0] > wrist_shape[1]:
        # center crop
        item["observation.images.wrist"] = item["observation.images.wrist"][:,wrist_shape[1]//2-wrist_shape[0]//2:wrist_shape[1]//2+wrist_shape[0]//2,:]
    else:
        item["observation.images.wrist"] = item["observation.images.wrist"][wrist_shape[0]//2-wrist_shape[1]//2:wrist_shape[0]//2+wrist_shape[1]//2,:,:]
    item["observation.images.wrist"] = Image.fromarray(item["observation.images.wrist"]).resize((224,224))
    item["observation.images.wrist"] = pil2tensor(item["observation.images.wrist"])
    
    item["observation.state"] = (item["observation.state"] - state_mean) / (state_std + 1e-8)
    item["observation.state"] = item["observation.state"].unsqueeze(0).to(dtype=torch.bfloat16)
    
    # input = prepare_input(item, processor)
    # input["action.mean"] = action_mean
    # input["action.std"] = action_std
    
    for k,v in item.items():
        if isinstance(v, torch.Tensor):
            item[k] = v.to(device)
    
    actions = pi0.infer(item, noise=None).tolist() # 1 * 50 *32
    # actions = actions[0] # 50 * 32
    actions = np.array([row[:14] for row in actions[0]]) # 50 * 14 eef pose
    actions = (actions * (action_std.numpy() + 1e-8)) + action_mean.numpy()
    # actions = [row[6:14] for row in actions[0]] # 50 * 7 joint
    print(actions[:5])
    
    response_dict = {
        "act": actions.tolist()
    }
    return jsonify(response_dict)
# load qwen2.5vl's vl processor when calling __init__

@parser.wrap()
def start_service(cfg: TrainPipelineConfig):
    
    path_2_load = "/data_16T/deepseek/pi0pizza/mp_rank_00_model_states.pt"
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
    action_mean = dataset.stats["action"]["mean"][:14]
    dataset.stats["action"]["mean"] = dataset.stats["action"]["mean"][:14]
    action_std = dataset.stats["action"]["std"][:14]
    dataset.stats["action"]["std"] = dataset.stats["action"]["std"][:14]
    print(action_mean, action_std)
    # action_mean[6] = 0.0
    # action_std[6] = 1.0
    
    state_mean = dataset.stats["observation.state"]["mean"][:15]
    dataset.stats["observation.state"]["mean"] = dataset.stats["observation.state"]["mean"][:15]
    state_std = dataset.stats["observation.state"]["std"][:15]
    dataset.stats["observation.state"]["std"] = dataset.stats["observation.state"]["std"][:15]
    
    # processor = dataset.processor
    
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
            
        policy.load_state_dict(model_state_dict["module"], strict=True)
        del model_state_dict
        del key_to_remove
    
    for params in policy.parameters():
        params.data = params.data.bfloat16()
        
    pi0 = policy
    pi0.eval()

    pi0.to(device="cuda:0")
    
    device = "cuda:0"
    
    return device, pi0, state_mean, state_std, action_mean, action_std

if __name__ == '__main__':
    device, pi0, state_mean, state_std, action_mean, action_std = start_service()
    # image_pool = {}
    app.run(host='0.0.0.0', port=7777)