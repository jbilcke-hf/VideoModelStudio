import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from utils import parse_bool_env

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
ASK_USER_TO_DUPLICATE_SPACE = parse_bool_env(os.getenv("ASK_USER_TO_DUPLICATE_SPACE"))

# Base storage path
STORAGE_PATH = Path(os.environ.get('STORAGE_PATH', '.data'))

# Subdirectories for different data types
VIDEOS_TO_SPLIT_PATH = STORAGE_PATH / "videos_to_split"    # Raw uploaded/downloaded files
STAGING_PATH = STORAGE_PATH / "staging"                    # This is where files that are captioned or need captioning are waiting
TRAINING_PATH = STORAGE_PATH / "training"                  # Folder containing the final training dataset
TRAINING_VIDEOS_PATH = TRAINING_PATH / "videos"            # Captioned clips ready for training
MODEL_PATH = STORAGE_PATH / "model"                        # Model checkpoints and files
OUTPUT_PATH = STORAGE_PATH / "output"                  # Training outputs and logs
LOG_FILE_PATH = OUTPUT_PATH / "last_session.log"

# On the production server we can afford to preload the big model
PRELOAD_CAPTIONING_MODEL = parse_bool_env(os.environ.get('PRELOAD_CAPTIONING_MODEL'))

CAPTIONING_MODEL = "lmms-lab/LLaVA-Video-7B-Qwen2"

DEFAULT_PROMPT_PREFIX = "In the style of TOK, "

# This is only use to debug things in local
USE_MOCK_CAPTIONING_MODEL = parse_bool_env(os.environ.get('USE_MOCK_CAPTIONING_MODEL'))

DEFAULT_CAPTIONING_BOT_INSTRUCTIONS = "Please write a full description of the following video: camera (close-up shot, medium-shot..), genre (music video, horror movie scene, video game footage, go pro footage, japanese anime, noir film, science-fiction, action movie, documentary..), characters (physical appearance, look, skin, facial features, haircut, clothing), scene (action, positions, movements), location (indoor, outdoor, place, building, country..), time and lighting (natural, golden hour, night time, LED lights, kelvin temperature etc), weather and climate (dusty, rainy, fog, haze, snowing..), era/settings"
       
# Create directories
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
VIDEOS_TO_SPLIT_PATH.mkdir(parents=True, exist_ok=True)
STAGING_PATH.mkdir(parents=True, exist_ok=True)
TRAINING_PATH.mkdir(parents=True, exist_ok=True)
TRAINING_VIDEOS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Image normalization settings
NORMALIZE_IMAGES_TO = os.environ.get('NORMALIZE_IMAGES_TO', 'png').lower()
if NORMALIZE_IMAGES_TO not in ['png', 'jpg']:
    raise ValueError("NORMALIZE_IMAGES_TO must be either 'png' or 'jpg'")
JPEG_QUALITY = int(os.environ.get('JPEG_QUALITY', '97'))

MODEL_TYPES = {
    "HunyuanVideo (LoRA)": "hunyuan_video", 
    "LTX-Video (LoRA)": "ltx_video"
}


# it is best to use resolutions that are powers of 8
# The resolution should be divisible by 32
# so we cannot use 1080, 540 etc as they are not divisible by 32
MEDIUM_19_9_RATIO_WIDTH = 768 # 32 * 24
MEDIUM_19_9_RATIO_HEIGHT = 512 # 32 * 16

# 1920 = 32 * 60 (divided by 2: 960 = 32 * 30)
# 1920 = 32 * 60 (divided by 2: 960 = 32 * 30)
# 1056 = 32 * 33 (divided by 2: 544 = 17 * 32)
# 1024 = 32 * 32 (divided by 2: 512 = 16 * 32)
# it is important that the resolution buckets properly cover the training dataset,
# or else that we exclude from the dataset videos that are out of this range
# right now, finetrainers will crash if that happens, so the workaround is to have more buckets in here

NB_FRAMES_1 = 1  #  1
NB_FRAMES_9 = 8 + 1 # 8 + 1
NB_FRAMES_17 = 8 * 2 + 1 # 16 + 1
NB_FRAMES_32 = 8 * 4 + 1  # 32 + 1
NB_FRAMES_48 = 8 * 6 + 1 # 48 + 1
NB_FRAMES_64 = 8 * 8 + 1  # 64 + 1
NB_FRAMES_80 = 8 * 10 + 1  # 80 + 1
NB_FRAMES_96 = 8 * 12 + 1  # 96 + 1
NB_FRAMES_112 = 8 * 14 + 1  # 112 + 1
NB_FRAMES_128 = 8 * 16 + 1  # 128 + 1
NB_FRAMES_144 = 8 * 18 + 1  # 144 + 1
NB_FRAMES_160  = 8 * 20 + 1  # 160 + 1
NB_FRAMES_176 = 8 * 22 + 1  # 176 + 1
NB_FRAMES_192 = 8 * 24 + 1  # 192 + 1
NB_FRAMES_224 = 8 * 28 + 1  # 224 + 1
NB_FRAMES_256 = 8 * 32 + 1  # 256 + 1
# 256 isn't a lot by the way, especially with 60 FPS videos.. 
# can we crank it and put more frames in here?

SMALL_TRAINING_BUCKETS = [
    (NB_FRAMES_1,   MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 1
    (NB_FRAMES_9,   MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 8 + 1
    (NB_FRAMES_17,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 16 + 1
    (NB_FRAMES_32,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 32 + 1
    (NB_FRAMES_48,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 48 + 1
    (NB_FRAMES_64,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 64 + 1
    (NB_FRAMES_80,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 80 + 1
    (NB_FRAMES_96,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 96 + 1
    (NB_FRAMES_112, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 112 + 1
    (NB_FRAMES_128, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 128 + 1
    (NB_FRAMES_144, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 144 + 1
    (NB_FRAMES_160, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 160 + 1
    (NB_FRAMES_176, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 176 + 1
    (NB_FRAMES_192, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 192 + 1
    (NB_FRAMES_224, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 224 + 1
    (NB_FRAMES_256, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 256 + 1
]

MEDIUM_19_9_RATIO_WIDTH = 928 # 32 * 29
MEDIUM_19_9_RATIO_HEIGHT = 512 # 32 * 16

MEDIUM_19_9_RATIO_BUCKETS = [
    (NB_FRAMES_1,   MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), #  1
    (NB_FRAMES_9,   MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 8 + 1
    (NB_FRAMES_17,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 16 + 1
    (NB_FRAMES_32,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 32 + 1
    (NB_FRAMES_48,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 48 + 1
    (NB_FRAMES_64,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 64 + 1
    (NB_FRAMES_80,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 80 + 1
    (NB_FRAMES_96,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 96 + 1
    (NB_FRAMES_112, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 112 + 1
    (NB_FRAMES_128, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 128 + 1
    (NB_FRAMES_144, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 144 + 1
    (NB_FRAMES_160, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 160 + 1
    (NB_FRAMES_176, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 176 + 1
    (NB_FRAMES_192, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 192 + 1
    (NB_FRAMES_224, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 224 + 1
    (NB_FRAMES_256, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 256 + 1
]

TRAINING_PRESETS = {
    "HunyuanVideo (normal)": {
        "model_type": "hunyuan_video",
        "lora_rank": "128",
        "lora_alpha": "128",
        "num_epochs": 70,
        "batch_size": 1,
        "learning_rate": 2e-5,
        "save_iterations": 500,
        "training_buckets": SMALL_TRAINING_BUCKETS,
    },
    "LTX-Video (normal)": {
        "model_type": "ltx_video", 
        "lora_rank": "128",
        "lora_alpha": "128",
        "num_epochs": 70,
        "batch_size": 1,
        "learning_rate": 3e-5,
        "save_iterations": 500,
        "training_buckets": SMALL_TRAINING_BUCKETS,
    },
    "LTX-Video (16:9, HQ)": {
        "model_type": "ltx_video",
        "lora_rank": "256", 
        "lora_alpha": "128",
        "num_epochs": 50,
        "batch_size": 1,
        "learning_rate": 3e-5,
        "save_iterations": 200,
        "training_buckets": MEDIUM_19_9_RATIO_BUCKETS,
    }
}

@dataclass
class TrainingConfig:
    """Configuration class for finetrainers training"""
    
    # Required arguments must come first
    model_name: str
    pretrained_model_name_or_path: str
    data_root: str
    output_dir: str
    
    # Optional arguments follow
    revision: Optional[str] = None
    variant: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Dataset arguments

    # note: video_column and caption_column serve a dual purpose,
    # when using the CSV mode they have to be CSV column names,
    # otherwise they have to be filename (relative to the data_root dir path)
    video_column: str = "videos.txt"
    caption_column: str = "prompts.txt"

    id_token: Optional[str] = None
    video_resolution_buckets: List[Tuple[int, int, int]] = field(default_factory=lambda: SMALL_TRAINING_BUCKETS)
    video_reshape_mode: str = "center"
    caption_dropout_p: float = 0.05
    caption_dropout_technique: str = "empty"
    precompute_conditions: bool = False
    
    # Diffusion arguments
    flow_resolution_shifting: bool = False
    flow_weighting_scheme: str = "none"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_mode_scale: float = 1.29
    
    # Training arguments
    training_type: str = "lora"
    seed: int = 42
    mixed_precision: str = "bf16"
    batch_size: int = 1
    train_epochs: int = 70
    lora_rank: int = 128
    lora_alpha: int = 128
    target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    checkpointing_steps: int = 500
    checkpointing_limit: Optional[int] = 2
    resume_from_checkpoint: Optional[str] = None
    enable_slicing: bool = True
    enable_tiling: bool = True

    # Optimizer arguments
    optimizer: str = "adamw"
    lr: float = 3e-5
    scale_lr: bool = False
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1e-4
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Miscellaneous arguments
    tracker_name: str = "finetrainers"
    report_to: str = "wandb"
    nccl_timeout: int = 1800

    @classmethod
    def hunyuan_video_lora(cls, data_path: str, output_path: str, buckets=None) -> 'TrainingConfig':
        """Configuration for Hunyuan video-to-video LoRA training"""
        return cls(
            model_name="hunyuan_video",
            pretrained_model_name_or_path="hunyuanvideo-community/HunyuanVideo",
            data_root=data_path,
            output_dir=output_path,
            batch_size=1,
            train_epochs=70,
            lr=2e-5,
            gradient_checkpointing=True,
            id_token="afkx",
            gradient_accumulation_steps=1,
            lora_rank=128,
            lora_alpha=128,
            video_resolution_buckets=buckets or SMALL_TRAINING_BUCKETS,
            caption_dropout_p=0.05,
            flow_weighting_scheme="none"  # Hunyuan specific
        )
    
    @classmethod
    def ltx_video_lora(cls, data_path: str, output_path: str, buckets=None) -> 'TrainingConfig':
        """Configuration for LTX-Video LoRA training"""
        return cls(
            model_name="ltx_video",
            pretrained_model_name_or_path="Lightricks/LTX-Video",
            data_root=data_path,
            output_dir=output_path,
            batch_size=1,
            train_epochs=40,
            lr=3e-5,
            gradient_checkpointing=True,
            id_token="BW_STYLE",
            gradient_accumulation_steps=4,
            lora_rank=128,
            lora_alpha=128,
            video_resolution_buckets=buckets or SMALL_TRAINING_BUCKETS,
            caption_dropout_p=0.05,
            flow_weighting_scheme="logit_normal"  # LTX specific
        )

    def to_args_list(self) -> List[str]:
        """Convert config to command line arguments list"""
        args = []
        
        # Model arguments 

        # Add model_name (required argument)
        args.extend(["--model_name", self.model_name])
        
        args.extend(["--pretrained_model_name_or_path", self.pretrained_model_name_or_path])
        if self.revision:
            args.extend(["--revision", self.revision])
        if self.variant:
            args.extend(["--variant", self.variant]) 
        if self.cache_dir:
            args.extend(["--cache_dir", self.cache_dir])

        # Dataset arguments
        args.extend(["--data_root", self.data_root])
        args.extend(["--video_column", self.video_column])
        args.extend(["--caption_column", self.caption_column])
        if self.id_token:
            args.extend(["--id_token", self.id_token])
            
        # Add video resolution buckets
        if self.video_resolution_buckets:
            bucket_strs = [f"{f}x{h}x{w}" for f, h, w in self.video_resolution_buckets]
            args.extend(["--video_resolution_buckets"] + bucket_strs)
            
        if self.video_reshape_mode:
            args.extend(["--video_reshape_mode", self.video_reshape_mode])
            
        args.extend(["--caption_dropout_p", str(self.caption_dropout_p)])
        args.extend(["--caption_dropout_technique", self.caption_dropout_technique])
        if self.precompute_conditions:
            args.append("--precompute_conditions")

        # Diffusion arguments
        if self.flow_resolution_shifting:
            args.append("--flow_resolution_shifting")
        args.extend(["--flow_weighting_scheme", self.flow_weighting_scheme])
        args.extend(["--flow_logit_mean", str(self.flow_logit_mean)])
        args.extend(["--flow_logit_std", str(self.flow_logit_std)])
        args.extend(["--flow_mode_scale", str(self.flow_mode_scale)])

        # Training arguments
        args.extend(["--training_type", self.training_type])
        args.extend(["--seed", str(self.seed)])
        
        # we don't use this,  because mixed precision is handled by accelerate launch, not by the training script itself.
        #args.extend(["--mixed_precision", self.mixed_precision])
        
        args.extend(["--batch_size", str(self.batch_size)])
        args.extend(["--train_epochs", str(self.train_epochs)])
        args.extend(["--rank", str(self.lora_rank)])
        args.extend(["--lora_alpha", str(self.lora_alpha)])
        args.extend(["--target_modules"] + self.target_modules)
        args.extend(["--gradient_accumulation_steps", str(self.gradient_accumulation_steps)])
        if self.gradient_checkpointing:
            args.append("--gradient_checkpointing")
        args.extend(["--checkpointing_steps", str(self.checkpointing_steps)])
        if self.checkpointing_limit:
            args.extend(["--checkpointing_limit", str(self.checkpointing_limit)])
        if self.resume_from_checkpoint:
            args.extend(["--resume_from_checkpoint", self.resume_from_checkpoint])
        if self.enable_slicing:
            args.append("--enable_slicing")
        if self.enable_tiling:
            args.append("--enable_tiling")

        # Optimizer arguments
        args.extend(["--optimizer", self.optimizer])
        args.extend(["--lr", str(self.lr)])
        if self.scale_lr:
            args.append("--scale_lr")
        args.extend(["--lr_scheduler", self.lr_scheduler])
        args.extend(["--lr_warmup_steps", str(self.lr_warmup_steps)])
        args.extend(["--lr_num_cycles", str(self.lr_num_cycles)])
        args.extend(["--lr_power", str(self.lr_power)])
        args.extend(["--beta1", str(self.beta1)])
        args.extend(["--beta2", str(self.beta2)])
        args.extend(["--weight_decay", str(self.weight_decay)])
        args.extend(["--epsilon", str(self.epsilon)])
        args.extend(["--max_grad_norm", str(self.max_grad_norm)])

        # Miscellaneous arguments
        args.extend(["--tracker_name", self.tracker_name])
        args.extend(["--output_dir", self.output_dir])
        args.extend(["--report_to", self.report_to])
        args.extend(["--nccl_timeout", str(self.nccl_timeout)])

        # normally this is disabled by default, but there was a bug in finetrainers
        # so I had to fix it in trainer.py to make sure we check for push_to-hub
        #args.append("--push_to_hub")
        #args.extend(["--hub_token", str(False)])
        #args.extend(["--hub_model_id", str(False)])

        # If you are using LLM-captioned videos, it is common to see many unwanted starting phrases like
        # "In this video, ...", "This video features ...", etc.
        # To remove a simple subset of these phrases, you can specify
        # --remove_common_llm_caption_prefixes when starting training.
        args.append("--remove_common_llm_caption_prefixes")

        return args