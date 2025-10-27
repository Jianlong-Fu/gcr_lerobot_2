import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoImageProcessor
from typing import Any, Dict, Tuple

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.common.policies.xvla.modeling_florence2 import Florence2ForConditionalGeneration
from lerobot.common.policies.xvla.processing_florence2 import Florence2Processor
from lerobot.common.policies.xvla.configuration_florence2 import Florence2Config
from lerobot.common.policies.xvla.components import (
    EE6DLoss,
    JointLoss,
    SoftPromptedTransformer,
    LanguagePreprocessor,
    ImagePreprocessor,
    find_domain_id
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype

LOSS_HUB = {
    "ee6d": EE6DLoss,
    "joint": JointLoss,
}

class XVLA(PreTrainedPolicy):
    """
    XVLA

    Overview
    --------
    - Visual-Language Encoder (VLMEncoder): encodes multi-view images and language tokens.
    - SoftPromptedTransformer: fuses (noisy) action sequences, proprioception, time, and VLM features.
    - Loss: fixed-parameter loss (EE6DLoss or JointLoss), no runtime config objects.

    Notes
    -----
    - Gripper channel indices are taken from the selected loss class (`.GRIPPER_IDX`).
    - During training/inference, gripper channels in proprio and noisy actions are zeroed.
    - At inference, a sigmoid is applied only on gripper channels.
    """
    config_class = XVLAConfig
    
    name = "xvla"
    
    def __init__(self, 
                config: XVLAConfig, 
                dataset_stats: dict[str, dict[str, Tensor]] | None = None,
                ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        
        self.config = config
        
        action_mode = self.config.action_mode.lower()
        assert action_mode in {"ee6d", "joint"}, "action_mode must be 'ee6d' or 'joint'"
        
        self.criterion = LOSS_HUB[action_mode]()
        
        # image_processor = AutoImageProcessor.from_pretrained(
        #         self.config.encoder_name
        #     )
        # tokenizer = AutoTokenizer.from_pretrained(
        #         self.config.encoder_name
        #     )
        # vlm_processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)
        vlm_config = Florence2Config.from_pretrained(self.config.encoder_name)
        self.vlm = Florence2ForConditionalGeneration.from_pretrained(
            self.config.encoder_name,
            config=vlm_config,
            local_files_only=True,
            # trust_remote_code=True
            )
        # self.vlm = AutoModelForCausalLM.from_pretrained(
        #         self.config.encoder_name,
        #         torch_dtype="auto",
        #         local_files_only=True,
        #         trust_remote_code=True
        #     )
        
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head
                
        self.transformer = SoftPromptedTransformer(
            hidden_size=self.config.hidden_size,
            multi_model_input_size = self.vlm.config.projection_dim,
            depth = self.config.depth,
            num_heads=self.config.num_heads,
            dim_action = self.criterion.DIM_ACTION,
            dim_propio = self.criterion.DIM_ACTION,
            len_soft_prompts=self.config.len_soft_prompts,
            dim_time = self.config.dim_time,
            max_len_seq=self.config.max_len_seq,
            use_hetero_proj=self.config.use_hetero_proj
        )
        
        # I/O preprocessors (implementations are project-specific)
        self.text_preprocessor = LanguagePreprocessor(encoder_name=self.config.encoder_name)
        self.image_preprocessor = ImagePreprocessor()
        
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V], bool or 0/1
    ) -> Dict[str, torch.Tensor]:
        """
        Produce VLM token features and auxiliary visual tokens from multi-view inputs.

        Parameters
        ----------
        input_ids : LongTensor, shape [B, L]
            Token ids for the text prompt/instruction.
        pixel_values : FloatTensor, shape [B, V, C, H, W]
            Image batch with V views per sample.
        image_mask : Tensor, shape [B, V]
            Mask indicating which views are valid (True/1) vs padded (False/0).

        Returns
        -------
        Dict[str, Tensor]
            {
            "vlm_features": FloatTensor [B, T_enc, D],  # encoder token sequence
            "aux_visual_inputs": FloatTensor [B, (V-1)*N, D]  # flattened features for views 1..V-1
            }
        """
        B, V = pixel_values.shape[:2]

        # Flatten views, select valid images, encode
        flat_mask = image_mask.view(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)                    # [B*V, C, H, W]

        num_valid = int(flat_mask.sum().item())
        assert num_valid > 0, "At least one image must be valid."

        valid_images = flat_images[flat_mask]                   # [#valid, C, H, W]
        
        valid_feats = self.vlm._encode_image(valid_images)    # [#valid, N, D]
        N, D = valid_feats.shape[1:]

        # Reconstruct dense [B, V, N, D] tensor
        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)        # [B, V, N, D]

        # Text embeddings
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids) # [B, L, D]

        # Merge first-view visual tokens with text embeddings
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],  # [B, N, D]
            inputs_embeds,         # [B, L, D]
        )

        # Run encoder to get token-level features
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        # Remaining views (1..V-1) flattened as auxiliary inputs
        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # [B, (V-1)*N, D]

        return {
            "vlm_features": enc_out,
            "aux_visual_inputs": aux_visual_inputs,
        }

    # ------------------------------ utilities -------------------------------
    def mask_gripper(self, proprio: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zero gripper channels in `proprio` and `action`.

        Parameters
        ----------
        proprio : Tensor, shape [B, dim_proprio]
        action : Tensor, shape [B, T, dim_action]

        Returns
        -------
        proprio_m : Tensor
            Proprio with gripper channels zeroed.
        action_m : Tensor
            Action with gripper channels zeroed.
        """
        idx = self.criterion.GRIPPER_IDX
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., idx] = 0.0
        action_m[..., idx] = 0.0
        return proprio_m, action_m

    # ------------------------------ training --------------------------------
    def forward(
        self,
        batch: dict[str, Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Parameters
        ----------
        input_ids : LongTensor, shape [B, L]
            Token IDs for language input.
        image_input : FloatTensor, shape [B, V, C, H, W]
            Multi-view images.
        image_mask : Tensor, shape [B, V]
            Mask for views (1 = valid, 0 = pad).
        domain_id : LongTensor, shape [B]
            Domain/task IDs.
        proprio : Tensor, shape [B, dim_proprio]
            Proprioceptive state.
        action : Tensor, shape [B, T=num_actions, dim_action]
            Ground-truth action sequence.

        Returns
        -------
        Dict[str, Tensor]
            Loss dictionary (keys depend on loss type), including "total_loss".
        """
        
        image_input = batch["image_input"]
        image_mask = batch["image_mask"]
        device = image_input.device
        language_instruction = self.text_preprocessor.encode_language(batch["task"])
        input_ids = language_instruction["input_ids"].to(device)
        
        proprio = batch['observation.state']
        action = batch["action"]
        domain_id = [find_domain_id(dataset_name) for dataset_name in batch["dataset_name"]]
        domain_id = torch.tensor(domain_id, device=device)
        # input_ids: torch.LongTensor,
        # image_input: torch.FloatTensor,
        # image_mask: torch.Tensor,
        # domain_id: torch.Tensor,
        # proprio: torch.Tensor,
        # action: torch.Tensor,
        enc = self.forward_vlm(input_ids=input_ids, pixel_values=image_input, image_mask=image_mask)

        B = input_ids.shape[0]
        # Per-sample time in (0, 1); avoid exactly 1.0
        time = (torch.rand(1, device=input_ids.device) + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        # Mix GT action with noise
        action_with_noise = torch.randn_like(action) * time.view(-1, 1, 1) + action * (1 - time).view(-1, 1, 1)

        # Mask gripper channels
        proprio_m, action_with_noise_m = self.mask_gripper(proprio, action_with_noise)

        # Predict actions
        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_with_noise_m,
            t=time,
            proprio=proprio_m,
            **enc,
        )
        return self.criterion(pred_action, action) 
    
    def get_optim_params(self) -> dict:
        return self.parameters()
    
    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.num_actions)
    
    @torch.no_grad
    def select_action(self, 
                    batch: dict[str, Tensor],
                    steps: int = 10
                ) -> Dict[str, torch.Tensor]:
        """
        Iterative denoising sampler (linear schedule).

        Parameters
        ----------
        input_ids : LongTensor, shape [B, L]
            Token IDs for language input.
        image_input : FloatTensor, shape [B, V, C, H, W]
            Multi-view images.
        image_mask : Tensor, shape [B, V]
            Mask for views (1 = valid, 0 = pad).
        domain_id : LongTensor, shape [B]
            Domain/task IDs.
        proprio : Tensor, shape [B, dim_proprio]
            Proprioceptive state.
        steps : int
            Number of denoising iterations.

        Returns
        -------
        Tensor, shape [B, T=num_actions, dim_action]
            Predicted action sequence; sigmoid applied only on gripper channels.
        """
        self.eval()
        image_input = batch["image_input"]
        image_mask = batch["image_mask"]
        device = image_input.device
        language_instruction = self.text_preprocessor.encode_language(batch["task"])
        input_ids = language_instruction["input_ids"].to(device)
        
        proprio = batch['observation.state']
        # action = batch["action"]
        domain_id = [find_domain_id(dataset_name) for dataset_name in batch["dataset_name"]]
        domain_id = torch.tensor(domain_id, device=device)
        
        enc = self.forward_vlm(input_ids=input_ids, pixel_values=image_input, image_mask=image_mask)
        
        B = input_ids.shape[0]
        device = input_ids.device
        x1 = torch.randn(B, self.config.num_actions, self.criterion.DIM_ACTION, device=device)
        action = torch.zeros(B, self.config.num_actions, self.criterion.DIM_ACTION, device=device)
        
        steps = max(int(steps), 1)
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=device)
            action_with_noise = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, action_with_noise_m = self.mask_gripper(proprio, action_with_noise)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=action_with_noise_m,
                t=t,
                proprio=proprio_m,
                **enc,
            )

        idx = self.criterion.GRIPPER_IDX
        action[..., idx] = torch.sigmoid(action[..., idx])
        return action
        # for t in range(self.config.num_actions):