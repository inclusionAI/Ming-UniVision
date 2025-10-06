#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from configuration_bailingmm import MingUniVisionConfig
from modeling_utils import patch_continuous_features, build_modality_mask

# audio encoder
from funasr.models.sanm.encoder import SANMEncoder
from modeling_bailing_moe import BailingMoeForCausalLM
from modeling_utils import Transpose, encode_audio_segments

# vision encoder
import sys
from omegaconf import MISSING, OmegaConf
from mingtok.modeling_mingtok import MingTokConfig, MingTok
from mingtok.utils import CenterCropProcessor
import json

from diff_loss_rf_swiglu import RectifiedFlowLoss

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MingUniVisionConfig"

DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>"
DEFAULT_GEN_IM_START_TOKEN = "<gen_image>"
DEFAULT_GEN_IM_END_TOKEN = "</gen_image>"
GLM_USER_PREFIX = "<role>HUMAN</role>"
GLM_ASSISTANT_PREFIX = "<role>ASSISTANT</role>"


@dataclass
class BailingMMCausalLMOutputWithPast(ModelOutput):
    """
    Base class for BailingMM causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class MingUniVisionForConditionalGeneration(PreTrainedModel):
    config_class = MingUniVisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingAudioModel"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: MingUniVisionConfig,
    ):
        super().__init__(config)
        self.config: MingUniVisionConfig = config
        self.llm_dytpe = torch.bfloat16

        # Load mingtok.
        self.vision = MingTok.from_pretrained("inclusionAI/MingTok-Vision")
        print('self.vision.feature_dim', self.vision.feature_dim)
        print('self.vision.image_emb_dim_for_gen', self.vision.latent_dim)

        # Make bailingmoe.
        assert self.config.llm_config is not None
        self.model = BailingMoeForCausalLM(self.config.llm_config)

        # Make linear_proj.
        mlp_modules_img = [nn.Linear(self.vision.feature_dim, self.model.config.hidden_size)]
        for _ in range(1, self.config.mlp_depth):
            mlp_modules_img.append(nn.GELU())
            mlp_modules_img.append(nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size))
        self.linear_proj = nn.Sequential(*mlp_modules_img)

        # Make vis_head and diffusion head.
        assert self.config.vishead_diffloss_config is not None
        self.config.vishead_diffloss_config['hidden_size'] = self.model.config.hidden_size
        self.config.vishead_diffloss_config['image_emb_dim_for_gen'] = self.vision.latent_dim
        self.model.setup_vishead_diffloss(**self.config.vishead_diffloss_config)

        self.tokenizer = None
        self.past_key_values = None
        self.past_attention_mask = None
        self.past_text_uncond_attention_mask = None
        self.past_uncond_attention_mask = None

        self.post_init()

    def extract_image_feature(self, pixel_values, grid_thw):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            eva_image_feat = self.vision(pixel_values)
            if isinstance(eva_image_feat, dict):
                eva_image_feat = eva_image_feat['x_norm_patchtokens']
            eva_image_feat = eva_image_feat.float()
            image_embeds = self.linear_proj(eva_image_feat)
        return image_embeds

    def extract_audio_feature(self, audio_feats, audio_feats_lengths, use_whisper_encoder=False):
        audio_embeds, _, audio_embeds_lengths = encode_audio_segments(
            encoder=self.audio,
            proj_layer=self.linear_proj_audio,
            wav_feats=audio_feats,
            wav_feats_lengths=audio_feats_lengths,
            audio_config=self.config.audio_config
        )
        if self.config.audio_config.norm_query_embeds:
            audio_embeds = F.normalize(audio_embeds, dim=2)  # [-1, 256, 2048]
        return audio_embeds.to(audio_feats.dtype), audio_embeds_lengths
    
    def prompt_wrap_vision(self, input_ids, inputs_embeds, vision_embeds, image_token_id=None):
        if vision_embeds is None or input_ids is None:
            return inputs_embeds

        if len(vision_embeds.shape) == 3:
            vision_embeds = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        self.config.llm_config.image_patch_token = image_token_id if image_token_id is not None else self.config.llm_config.image_patch_token
        n_image_tokens = (input_ids == self.config.llm_config.image_patch_token).sum().item()
        n_image_features = vision_embeds.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        image_router_mask =  (
            (input_ids == self.config.llm_config.image_patch_token)
            .unsqueeze(-1)
            .to(inputs_embeds.device)
        ) 
        image_mask = image_router_mask.expand_as(inputs_embeds)
        image_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_router_mask = image_router_mask.squeeze(-1)
        return inputs_embeds, image_router_mask

    def prompt_wrap_audio(self, input_ids, inputs_embeds, audio_embeds, audio_embeds_lengths, placeholder_audio_loc_lens):
        inputs_embeds = patch_continuous_features(
           input_embeddings=inputs_embeds, placeholder_loc_lens=placeholder_audio_loc_lens,
           encoded_feats=audio_embeds, encoded_feat_lens=audio_embeds_lengths,
        )
        audio_router_mask = build_modality_mask(placeholder_audio_loc_lens, inputs_embeds.shape[:-1]).to(inputs_embeds.device)
        return inputs_embeds, audio_router_mask
     
    def prompt_wrap_navit(self, input_ids, query_embeds_image=None, query_embeds_video=None, query_embeds_audio=None,
        query_embeds_audio_lengths=None, placeholder_audio_loc_lens=None, target_embeds=None):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None and target_embeds is None:
            return inputs_embeds

        # import pdb; pdb.set_trace()
        image_mask = None
        audio_mask = None
        if query_embeds_image is not None:
            inputs_embeds, image_mask = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_image)
        if query_embeds_video is not None:
            inputs_embeds, image_mask = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_video)
        if query_embeds_audio is not None:
            inputs_embeds, audio_mask = self.prompt_wrap_audio(
                input_ids, inputs_embeds, query_embeds_audio, query_embeds_audio_lengths, placeholder_audio_loc_lens,
            )
        return inputs_embeds, image_mask, audio_mask

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        uncond_attention_mask: Optional[torch.Tensor] = None,
        text_uncond_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_feats: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        output_image_prefix: Optional[str] = "output",
        image_gen_temperature: Optional[float] = 1.0,
        image_gen_text_cfg: Optional[float] = 3.0,
        image_gen_image_cfg: Optional[float] = 1.1,
        **generate_kwargs,
    ):
        if self.past_key_values is not None and past_key_values is None:
            past_key_values = self.past_key_values
        if self.past_attention_mask is not None:
            attention_mask = torch.cat((self.past_attention_mask, attention_mask),dim=1)
            uncond_attention_mask = torch.cat((self.past_uncond_attention_mask, uncond_attention_mask), dim=1)
            text_uncond_attention_mask = torch.cat((self.past_text_uncond_attention_mask, text_uncond_attention_mask), dim=1)
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if (image_embeds is None and video_embeds is None and audio_embeds is None) or input_ids.size(1) == 1:
                words_embeddings = self.model.get_input_embeddings()(input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1))
                image_mask = None
                audio_mask = None
            else:
                words_embeddings, image_mask, audio_mask = self.prompt_wrap_navit(
                        input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1), image_embeds, video_embeds, audio_embeds,
                        audio_embeds_lengths, audio_placeholder_loc_lens, None
                )
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                text_uncond_attention_mask=text_uncond_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=words_embeddings,
                use_cache=use_cache,
                image_mask=image_mask,
                audio_mask=audio_mask,
                rope_deltas=None,
                latent_to_sem_func=self.vision.forward_feature_decoder,
                linear_proj=self.linear_proj,
                sem_to_pix_func=self.vision.forward_pixel_decoder,
                output_image_prefix=output_image_prefix,
                return_dict_in_generate=True,
                image_gen_temperature=image_gen_temperature,
                image_gen_text_cfg=image_gen_text_cfg,
                image_gen_image_cfg=image_gen_image_cfg,
                **generate_kwargs,
            )   

        # save states for future rounds
        self.past_key_values = outputs.past_key_values
        cache_length = self.past_key_values.get_seq_length()
        pad_attn_mask = torch.ones(attention_mask.shape[0], cache_length-attention_mask.shape[1], dtype=attention_mask.dtype, device=attention_mask.device)
        pad_uncond_attn_mask = torch.zeros(attention_mask.shape[0], cache_length-attention_mask.shape[1], dtype=attention_mask.dtype, device=attention_mask.device)

        import os
        past_mode = os.environ.get('PAST_MODE', "DROP")
        if past_mode == "KEEP":
            self.past_attention_mask = torch.cat((
                        attention_mask, pad_attn_mask
                    ), dim=1)
            self.past_text_uncond_attention_mask = torch.cat((
                        text_uncond_attention_mask, pad_attn_mask
                    ), dim=1)
            self.past_uncond_attention_mask = torch.cat((
                        uncond_attention_mask, pad_uncond_attn_mask
                    ), dim=1)
        elif past_mode == "DROP":
            self.past_attention_mask = torch.cat((
                        attention_mask, pad_attn_mask
                    ), dim=1)
            self.past_text_uncond_attention_mask = torch.cat((
                        attention_mask, pad_attn_mask
                    ), dim=1)
            self.past_uncond_attention_mask = torch.cat((
                        attention_mask, pad_uncond_attn_mask
                    ), dim=1)
        self.model.reset_image_gen_status() # clear generated_images per round
        return outputs.sequences

    def reset_inner_state(self):
        self.past_key_values = None
        self.past_attention_mask = None
        self.past_text_uncond_attention_mask = None
        self.past_uncond_attention_mask = None
        self.model.reset_image_gen_status()