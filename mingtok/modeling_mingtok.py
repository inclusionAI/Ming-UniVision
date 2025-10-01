
from omegaconf import MISSING, OmegaConf

import os
import math
import contextlib

import torch
import torch.nn as nn
from einops import rearrange

from .vision_transformer import build_low_level_encoder, build_semantic_decoder, build_pixel_decoder

from dataclasses import dataclass
from typing import Optional

from transformers import PretrainedConfig, PreTrainedModel


@dataclass
class LowLevelEncoderConfig:
    img_size: int
    patch_size: int
    depth: int
    embed_dim: int
    ffn_layer: str
    out_dim: int

@dataclass
class SemanticDecoderConfig:
    in_dim: int
    patch_size: int
    embed_dim: int
    decoder_depth: int
    ffn_layer: str

@dataclass
class PixelDecoderConfig:
    patch_size: int
    decoder_depth: int
    norm_pix_loss: bool
    embed_dim: int
    loss_type: str

@dataclass
class MingTokConfigPlain:
    low_level_encoder: LowLevelEncoderConfig
    semantic_decoder: SemanticDecoderConfig
    pixel_decoder: PixelDecoderConfig
    pretrained_checkpoint: str
    model_dtype: str = "bf16"

    scaling_factor: float = 1.0
    mean: float = 0.0

class MingTokConfig(PretrainedConfig):
    model_type = "mingtok"

    def __init__(
        self,
        low_level_encoder=None,
        semantic_decoder=None,
        pixel_decoder=None,
        pretrained_checkpoint=None,
        model_dtype="bf16",
        scaling_factor=1.0,
        mean=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.low_level_encoder = low_level_encoder or {}
        self.semantic_decoder = semantic_decoder or {}
        self.pixel_decoder = pixel_decoder or {}
        self.pretrained_checkpoint = pretrained_checkpoint
        self.model_dtype = model_dtype
        self.scaling_factor = scaling_factor
        self.mean = mean

    @classmethod
    def from_omegaconf(cls, cfg):
        return cls(
            low_level_encoder=cfg.low_level_encoder,
            semantic_decoder=cfg.semantic_decoder,
            pixel_decoder=cfg.pixel_decoder,
            pretrained_checkpoint=cfg.get("pretrained_checkpoint", None),
            model_dtype=cfg.get("model_dtype", "bf16"),
            scaling_factor=cfg.get("scaling_factor", 1.0),
            mean=cfg.get("mean", 0.0),
        )

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

class MingTok(PreTrainedModel):
    config_class = MingTokConfig
    base_model_prefix = "mingtok"

    def __init__(self, config: MingTokConfig):
        super().__init__(config)

        self.config = config

        self.latent_dim = self.config.low_level_encoder.get("out_dim", 32)
        self.feature_dim = self.config.semantic_decoder.get("embed_dim", 1024)
        self.patch_size = self.config.low_level_encoder.get("patch_size", 32)

        self.model_dtype = DTYPE_MAP[config.model_dtype]

        self.low_level_encoder = build_low_level_encoder(config.low_level_encoder)

        self.semantic_decoder = build_semantic_decoder(config.semantic_decoder)

        self.pixel_decoder = build_pixel_decoder(config.pixel_decoder)

        self.sem_to_pix = nn.Linear(
            self.semantic_decoder.num_features,
            self.pixel_decoder.num_features * ((self.semantic_decoder.patch_size // self.pixel_decoder.patch_size)**2)
        )

        self.scaling_factor = self.config.scaling_factor
        self.mean = self.config.mean

        self.post_init()
        self.maybe_load_checkpoint()

    def maybe_load_checkpoint(self):
        ckpt_path = getattr(self.config, "pretrained_checkpoint", None)
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading pretrained weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("target_backbone")}
            msg = self.load_state_dict(filtered_state_dict, strict=False)
            print(f"Missing: {msg[0]}, Unexpected: {msg[1]}")
        else:
            print("not loading visual backbone weights")

    @property
    def num_features(self):
        return self.visual_encoder.num_features
    
    @property
    def device(self):
        # avoid iter all parameters
        for _, param in self.named_parameters():
            return param.device

    def forward_enc_dec(self, x):
        features = self.forward(x)
        recon_image = self.forward_pixel_decoder(features['x_norm_patchtokens'])
        return recon_image

    
    def forward(self, x):
        with self.maybe_autocast(self.model_dtype):
            latent = self.low_level_encoder(x)
            features = self.semantic_decoder(latent)
            return {
                "x_norm_patchtokens": features['x_norm_patchtokens'],
                "latent": (latent - self.mean) / self.scaling_factor
            }
    
    def forward_feature_decoder(self, hidden_states, past_key_values=None):
        with self.maybe_autocast(self.model_dtype):
            with torch.no_grad():
                hidden_states = hidden_states * self.scaling_factor + self.mean
                return self.semantic_decoder(
                    hidden_states, 
                    use_cache=True, 
                    past_key_values=past_key_values, 
                    cache_position=None
                )
            
    def forward_feature_decoder_wo_cache(self, hidden_states):
        return self.semantic_decoder(hidden_states, use_cache=False)
    
    def forward_pixel_decoder(self, x):
        with self.maybe_autocast(torch.float32):

            if self.sem_to_pix is not None:
                x_enc_for_recon = self.sem_to_pix(x)
                x_enc_for_recon = rearrange(x_enc_for_recon, "b (h w) (x y c) -> b (h x w y) c", 
                h=int(math.sqrt(x_enc_for_recon.shape[1])), 
                w=int(math.sqrt(x_enc_for_recon.shape[1])), 
                x=self.semantic_decoder.patch_size // self.pixel_decoder.patch_size, 
                y=self.semantic_decoder.patch_size // self.pixel_decoder.patch_size)
            
            x_recon = self.pixel_decoder(x_enc_for_recon)

            o = self.pixel_decoder.unpatchify(x_recon)

            image = o.clamp_(-1, 1)

            return image

    def maybe_autocast(self, dtype=torch.float16, enable=True):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = (self.device != torch.device("cpu") and enable)

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
