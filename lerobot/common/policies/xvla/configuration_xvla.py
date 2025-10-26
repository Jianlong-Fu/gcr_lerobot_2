from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

@PreTrainedConfig.register_subclass("xvla")
@dataclass
class XVLAConfig(PreTrainedConfig):
    """Configuration for XVLA policy."""
    
    depth: int = 24
    hidden_size: int = 1024
    num_heads: int = 16
    num_actions: int = 30
    num_domains: int = 128
    len_soft_prompts: int = 32
    use_hetero_proj: bool = False
    action_mode: str = "ee6d"
    encoder_name: str = "microsoft/Florence-2-large"
    dim_time: int = 32
    max_len_seq: int = 512
    chunk_size = num_actions
    
    empty_cameras: int = 0
    
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    
    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-2
    
    optimizer_lr_vlmcoef: float = 1.0

    scheduler_warmup_steps: int = 2_000
    scheduler_decay_steps: int = -1
    scheduler_decay_lr: float = 1e-5
    scheduler_platform_steps: int = -1
    
    def __post_init__(self):
        super().__post_init__()
    
    def validate_features(self) -> None:
        # TODO: implement value error
        # if not self.image_features and not self.env_state_feature:
        #     raise ValueError("You must provide at least one image or the environment state among the inputs.")

        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera
            
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )
        
    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
        
    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.num_actions))
    
    @property
    def reward_delta_indices(self) -> None:
        return None