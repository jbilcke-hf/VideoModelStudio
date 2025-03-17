import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import torch
import math

def parse_bool_env(env_value: Optional[str]) -> bool:
    """Parse environment variable string to boolean
    
    Handles various true/false string representations:
    - True: "true", "True", "TRUE", "1", etc
    - False: "false", "False", "FALSE", "0", "", None
    """
    if not env_value:
        return False
    return str(env_value).lower() in ('true', '1', 't', 'y', 'yes')

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

DEFAULT_CAPTIONING_BOT_INSTRUCTIONS = "Please write a full video description. Be synthetic and methodically list camera (close-up shot, medium-shot..), genre (music video, horror movie scene, video game footage, go pro footage, japanese anime, noir film, science-fiction, action movie, documentary..), characters (physical appearance, look, skin, facial features, haircut, clothing), scene (action, positions, movements), location (indoor, outdoor, place, building, country..), time and lighting (natural, golden hour, night time, LED lights, kelvin temperature etc), weather and climate (dusty, rainy, fog, haze, snowing..), era/settings."
       
# Create directories
STORAGE_PATH.mkdir(parents=True, exist_ok=True)
VIDEOS_TO_SPLIT_PATH.mkdir(parents=True, exist_ok=True)
STAGING_PATH.mkdir(parents=True, exist_ok=True)
TRAINING_PATH.mkdir(parents=True, exist_ok=True)
TRAINING_VIDEOS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# To secure public instances
VMS_ADMIN_PASSWORD = os.environ.get('VMS_ADMIN_PASSWORD', '')

# Image normalization settings
NORMALIZE_IMAGES_TO = os.environ.get('NORMALIZE_IMAGES_TO', 'png').lower()
if NORMALIZE_IMAGES_TO not in ['png', 'jpg']:
    raise ValueError("NORMALIZE_IMAGES_TO must be either 'png' or 'jpg'")
JPEG_QUALITY = int(os.environ.get('JPEG_QUALITY', '97'))

MODEL_TYPES = {
    "HunyuanVideo": "hunyuan_video", 
    "LTX-Video": "ltx_video",
    "Wan": "wan"
}

# Training types
TRAINING_TYPES = {
    "LoRA Finetune": "lora",
    "Full Finetune": "full-finetune"
}

# Model versions for each model type
MODEL_VERSIONS = {
    "wan": {
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": {
            "name": "Wan 2.1 T2V 1.3B (text-only, smaller)",
            "type": "text-to-video",
            "description": "Faster, smaller model (1.3B parameters)"
        },
        "Wan-AI/Wan2.1-T2V-14B-Diffusers": {
            "name": "Wan 2.1 T2V 14B (text-only, larger)",
            "type": "text-to-video",
            "description": "Higher quality but slower (14B parameters)"
        },
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": {
            "name": "Wan 2.1 I2V 480p (image+text)",
            "type": "image-to-video",
            "description": "Image conditioning at 480p resolution"
        },
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": {
            "name": "Wan 2.1 I2V 720p (image+text)",
            "type": "image-to-video",
            "description": "Image conditioning at 720p resolution"
        }
    },
    "ltx_video": {
        "Lightricks/LTX-Video": {
            "name": "LTX Video (official)",
            "type": "text-to-video",
            "description": "Official LTX Video model"
        }
    },
    "hunyuan_video": {
        "hunyuanvideo-community/HunyuanVideo": {
            "name": "Hunyuan Video (official)",
            "type": "text-to-video",
            "description": "Official Hunyuan Video model"
        }
    }
}

DEFAULT_SEED = 42

DEFAULT_REMOVE_COMMON_LLM_CAPTION_PREFIXES = True

DEFAULT_DATASET_TYPE = "video"
DEFAULT_TRAINING_TYPE = "lora"

DEFAULT_RESHAPE_MODE = "bicubic"

DEFAULT_MIXED_PRECISION = "bf16"



DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS = 200

DEFAULT_LORA_RANK = 128
DEFAULT_LORA_RANK_STR = str(DEFAULT_LORA_RANK)

DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_ALPHA_STR = str(DEFAULT_LORA_ALPHA)

DEFAULT_CAPTION_DROPOUT_P = 0.05

DEFAULT_BATCH_SIZE = 1

DEFAULT_LEARNING_RATE = 3e-5

# GPU SETTINGS
DEFAULT_NUM_GPUS = 1
DEFAULT_MAX_GPUS = min(8, torch.cuda.device_count() if torch.cuda.is_available() else 1)
DEFAULT_PRECOMPUTATION_ITEMS = 512

DEFAULT_NB_TRAINING_STEPS = 1000

# For this value, it is recommended to use about 20 to 40% of the number of training steps
DEFAULT_NB_LR_WARMUP_STEPS = math.ceil(0.20 * DEFAULT_NB_TRAINING_STEPS)  # 20% of training steps

# Whether to automatically restart a training job after a server reboot or not
DEFAULT_AUTO_RESUME = False

# For validation
DEFAULT_VALIDATION_NB_STEPS = 50
DEFAULT_VALIDATION_HEIGHT = 512
DEFAULT_VALIDATION_WIDTH = 768
DEFAULT_VALIDATION_NB_FRAMES = 49
DEFAULT_VALIDATION_FRAMERATE = 8

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
NB_FRAMES_33 = 8 * 4 + 1  # 32 + 1
NB_FRAMES_49 = 8 * 6 + 1 # 48 + 1
NB_FRAMES_65 = 8 * 8 + 1  # 64 + 1
NB_FRAMES_81 = 8 * 10 + 1  # 80 + 1
NB_FRAMES_97 = 8 * 12 + 1  # 96 + 1
NB_FRAMES_113 = 8 * 14 + 1  # 112 + 1
NB_FRAMES_129 = 8 * 16 + 1  # 128 + 1
NB_FRAMES_145 = 8 * 18 + 1  # 144 + 1
NB_FRAMES_161  = 8 * 20 + 1  # 160 + 1
NB_FRAMES_177 = 8 * 22 + 1  # 176 + 1
NB_FRAMES_193 = 8 * 24 + 1  # 192 + 1
NB_FRAMES_225 = 8 * 28 + 1  # 224 + 1
NB_FRAMES_257 = 8 * 32 + 1  # 256 + 1
# 256 isn't a lot by the way, especially with 60 FPS videos.. 
# can we crank it and put more frames in here?

NB_FRAMES_273 = 8 * 34 + 1  # 272 + 1
NB_FRAMES_289 = 8 * 36 + 1  # 288 + 1
NB_FRAMES_305 = 8 * 38 + 1  # 304 + 1
NB_FRAMES_321 = 8 * 40 + 1  # 320 + 1
NB_FRAMES_337 = 8 * 42 + 1  # 336 + 1
NB_FRAMES_353 = 8 * 44 + 1  # 352 + 1
NB_FRAMES_369 = 8 * 46 + 1  # 368 + 1
NB_FRAMES_385 = 8 * 48 + 1  # 384 + 1
NB_FRAMES_401 = 8 * 50 + 1  # 400 + 1

SMALL_TRAINING_BUCKETS = [
    (NB_FRAMES_1,   MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 1
    (NB_FRAMES_9,   MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 8 + 1
    (NB_FRAMES_17,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 16 + 1
    (NB_FRAMES_33,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 32 + 1
    (NB_FRAMES_49,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 48 + 1
    (NB_FRAMES_65,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 64 + 1
    (NB_FRAMES_81,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 80 + 1
    (NB_FRAMES_97,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 96 + 1
    (NB_FRAMES_113, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 112 + 1
    (NB_FRAMES_129, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 128 + 1
    (NB_FRAMES_145, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 144 + 1
    (NB_FRAMES_161, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 160 + 1
    (NB_FRAMES_177, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 176 + 1
    (NB_FRAMES_193, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 192 + 1
    (NB_FRAMES_225, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 224 + 1
    (NB_FRAMES_257, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 256 + 1
]

MEDIUM_19_9_RATIO_WIDTH = 928 # 32 * 29
MEDIUM_19_9_RATIO_HEIGHT = 512 # 32 * 16

MEDIUM_19_9_RATIO_BUCKETS = [
    (NB_FRAMES_1,   MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), #  1
    (NB_FRAMES_9,   MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 8 + 1
    (NB_FRAMES_17,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 16 + 1
    (NB_FRAMES_33,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 32 + 1
    (NB_FRAMES_49,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 48 + 1
    (NB_FRAMES_65,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 64 + 1
    (NB_FRAMES_81,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 80 + 1
    (NB_FRAMES_97,  MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 96 + 1
    (NB_FRAMES_113, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 112 + 1
    (NB_FRAMES_129, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 128 + 1
    (NB_FRAMES_145, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 144 + 1
    (NB_FRAMES_161, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 160 + 1
    (NB_FRAMES_177, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 176 + 1
    (NB_FRAMES_193, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 192 + 1
    (NB_FRAMES_225, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 224 + 1
    (NB_FRAMES_257, MEDIUM_19_9_RATIO_HEIGHT, MEDIUM_19_9_RATIO_WIDTH), # 256 + 1
]

# Updated training presets to include Wan-2.1-T2V and support both LoRA and full-finetune
TRAINING_PRESETS = {
    "HunyuanVideo (normal)": {
        "model_type": "hunyuan_video",
        "training_type": "lora",
        "lora_rank": DEFAULT_LORA_RANK_STR,
        "lora_alpha": DEFAULT_LORA_ALPHA_STR,
        "train_steps": DEFAULT_NB_TRAINING_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": 2e-5,
        "save_iterations": DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
        "training_buckets": SMALL_TRAINING_BUCKETS,
        "flow_weighting_scheme": "none",
        "num_gpus": DEFAULT_NUM_GPUS,
        "precomputation_items": DEFAULT_PRECOMPUTATION_ITEMS,
        "lr_warmup_steps": DEFAULT_NB_LR_WARMUP_STEPS,
    },
    "LTX-Video (normal)": {
        "model_type": "ltx_video", 
        "training_type": "lora",
        "lora_rank": DEFAULT_LORA_RANK_STR,
        "lora_alpha": DEFAULT_LORA_ALPHA_STR,
        "train_steps": DEFAULT_NB_TRAINING_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "save_iterations": DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
        "training_buckets": SMALL_TRAINING_BUCKETS,
        "flow_weighting_scheme": "none",
        "num_gpus": DEFAULT_NUM_GPUS,
        "precomputation_items": DEFAULT_PRECOMPUTATION_ITEMS,
        "lr_warmup_steps": DEFAULT_NB_LR_WARMUP_STEPS,
    },
    "LTX-Video (16:9, HQ)": {
        "model_type": "ltx_video",
        "training_type": "lora",
        "lora_rank": "256", 
        "lora_alpha": DEFAULT_LORA_ALPHA_STR,
        "train_steps": DEFAULT_NB_TRAINING_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "save_iterations": DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
        "training_buckets": MEDIUM_19_9_RATIO_BUCKETS,
        "flow_weighting_scheme": "logit_normal",
        "num_gpus": DEFAULT_NUM_GPUS,
        "precomputation_items": DEFAULT_PRECOMPUTATION_ITEMS,
        "lr_warmup_steps": DEFAULT_NB_LR_WARMUP_STEPS,
    },
    "LTX-Video (Full Finetune)": {
        "model_type": "ltx_video",
        "training_type": "full-finetune",
        "train_steps": DEFAULT_NB_TRAINING_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "save_iterations": DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
        "training_buckets": SMALL_TRAINING_BUCKETS,
        "flow_weighting_scheme": "logit_normal",
        "num_gpus": DEFAULT_NUM_GPUS,
        "precomputation_items": DEFAULT_PRECOMPUTATION_ITEMS,
        "lr_warmup_steps": DEFAULT_NB_LR_WARMUP_STEPS,
    },
    "Wan-2.1-T2V (normal)": {
        "model_type": "wan",
        "training_type": "lora",
        "lora_rank": "32",
        "lora_alpha": "32",
        "train_steps": DEFAULT_NB_TRAINING_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": 5e-5,
        "save_iterations": DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
        "training_buckets": SMALL_TRAINING_BUCKETS,
        "flow_weighting_scheme": "logit_normal",
        "num_gpus": DEFAULT_NUM_GPUS,
        "precomputation_items": DEFAULT_PRECOMPUTATION_ITEMS,
        "lr_warmup_steps": DEFAULT_NB_LR_WARMUP_STEPS,
    },
    "Wan-2.1-T2V (HQ)": {
        "model_type": "wan",
        "training_type": "lora",
        "lora_rank": "64",
        "lora_alpha": "64",
        "train_steps": DEFAULT_NB_TRAINING_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "save_iterations": DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS,
        "training_buckets": MEDIUM_19_9_RATIO_BUCKETS,
        "flow_weighting_scheme": "logit_normal",
        "num_gpus": DEFAULT_NUM_GPUS,
        "precomputation_items": DEFAULT_PRECOMPUTATION_ITEMS,
        "lr_warmup_steps": DEFAULT_NB_LR_WARMUP_STEPS,
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
    version: Optional[str] = None
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
    caption_dropout_p: float = DEFAULT_CAPTION_DROPOUT_P
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
    seed: int = DEFAULT_SEED
    mixed_precision: str = "bf16"
    batch_size: int = 1
    train_steps: int = DEFAULT_NB_TRAINING_STEPS
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    checkpointing_steps: int = DEFAULT_SAVE_CHECKPOINT_EVERY_N_STEPS
    checkpointing_limit: Optional[int] = 2
    resume_from_checkpoint: Optional[str] = None
    enable_slicing: bool = True
    enable_tiling: bool = True

    # Optimizer arguments
    optimizer: str = "adamw"
    lr: float = DEFAULT_LEARNING_RATE
    scale_lr: bool = False
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = DEFAULT_NB_LR_WARMUP_STEPS
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
            train_steps=DEFAULT_NB_TRAINING_STEPS,
            lr=2e-5,
            gradient_checkpointing=True,
            id_token=None,
            gradient_accumulation_steps=1,
            lora_rank=DEFAULT_LORA_RANK,
            lora_alpha=DEFAULT_LORA_ALPHA,
            video_resolution_buckets=buckets or SMALL_TRAINING_BUCKETS,
            caption_dropout_p=DEFAULT_CAPTION_DROPOUT_P,
            flow_weighting_scheme="none",  # Hunyuan specific
            training_type="lora"
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
            train_steps=DEFAULT_NB_TRAINING_STEPS,
            lr=DEFAULT_LEARNING_RATE,
            gradient_checkpointing=True,
            id_token=None,
            gradient_accumulation_steps=4,
            lora_rank=DEFAULT_LORA_RANK,
            lora_alpha=DEFAULT_LORA_ALPHA,
            video_resolution_buckets=buckets or SMALL_TRAINING_BUCKETS,
            caption_dropout_p=DEFAULT_CAPTION_DROPOUT_P,
            flow_weighting_scheme="logit_normal",  # LTX specific
            training_type="lora"
        )
        
    @classmethod
    def ltx_video_full_finetune(cls, data_path: str, output_path: str, buckets=None) -> 'TrainingConfig':
        """Configuration for LTX-Video full finetune training"""
        return cls(
            model_name="ltx_video",
            pretrained_model_name_or_path="Lightricks/LTX-Video",
            data_root=data_path,
            output_dir=output_path,
            batch_size=1,
            train_steps=DEFAULT_NB_TRAINING_STEPS,
            lr=1e-5,
            gradient_checkpointing=True,
            id_token=None,
            gradient_accumulation_steps=1,
            video_resolution_buckets=buckets or SMALL_TRAINING_BUCKETS,
            caption_dropout_p=DEFAULT_CAPTION_DROPOUT_P,
            flow_weighting_scheme="logit_normal",  # LTX specific
            training_type="full-finetune"
        )
        
    @classmethod
    def wan_lora(cls, data_path: str, output_path: str, buckets=None) -> 'TrainingConfig':
        """Configuration for Wan T2V LoRA training"""
        return cls(
            model_name="wan",
            pretrained_model_name_or_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            data_root=data_path,
            output_dir=output_path,
            batch_size=1,
            train_steps=DEFAULT_NB_TRAINING_STEPS,
            lr=5e-5,
            gradient_checkpointing=True,
            id_token=None,
            gradient_accumulation_steps=1,
            lora_rank=32,
            lora_alpha=32,
            target_modules=["blocks.*(to_q|to_k|to_v|to_out.0)"],  # Wan-specific target modules
            video_resolution_buckets=buckets or SMALL_TRAINING_BUCKETS,
            caption_dropout_p=DEFAULT_CAPTION_DROPOUT_P,
            flow_weighting_scheme="logit_normal",  # Wan specific
            training_type="lora"
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
        if self.version:
            args.extend(["--variant", self.version]) 
        if self.cache_dir:
            args.extend(["--cache_dir", self.cache_dir])

        # Dataset arguments
        args.extend(["--dataset_config", self.data_root])
        
        # Add ID token if specified
        if self.id_token:
            args.extend(["--id_token", self.id_token])
            
        # Add video resolution buckets
        if self.video_resolution_buckets:
            bucket_strs = [f"{f}x{h}x{w}" for f, h, w in self.video_resolution_buckets]
            args.extend(["--video_resolution_buckets"] + bucket_strs)
            
        args.extend(["--caption_dropout_p", str(self.caption_dropout_p)])
        args.extend(["--caption_dropout_technique", self.caption_dropout_technique])
        if self.precompute_conditions:
            args.append("--precompute_conditions")

        if hasattr(self, 'precomputation_items') and self.precomputation_items:
            args.extend(["--precomputation_items", str(self.precomputation_items)])
            
        # Diffusion arguments
        if self.flow_resolution_shifting:
            args.append("--flow_resolution_shifting")
        args.extend(["--flow_weighting_scheme", self.flow_weighting_scheme])
        args.extend(["--flow_logit_mean", str(self.flow_logit_mean)])
        args.extend(["--flow_logit_std", str(self.flow_logit_std)])
        args.extend(["--flow_mode_scale", str(self.flow_mode_scale)])

        # Training arguments
        args.extend(["--training_type",self.training_type])
        args.extend(["--seed", str(self.seed)])
        
        # We don't use this, because mixed precision is handled by accelerate launch, not by the training script itself.
        #args.extend(["--mixed_precision", self.mixed_precision])
        
        args.extend(["--batch_size", str(self.batch_size)])
        args.extend(["--train_steps", str(self.train_steps)])
        
        # LoRA specific arguments
        if self.training_type == "lora":
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