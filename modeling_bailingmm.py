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
        self.vision = MingTok.from_pretrained("./models/MingTok-Vision")
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
    def unified_image_generation(
        self,
        caption,
        image_gen_cfg=1.0,
        with_separator=False,
        cfg_schedule="constant",
        image_gen_temperature=1.0,
        cfg_renorm_type=None,
        time_shifting_factor=None,
    ):
        gen_start_token = DEFAULT_IM_START_TOKEN
        prompt = "Please generate the corresponding image based on the description."
        if with_separator:
            text_input = prompt + "\n" + caption
        else:
            text_input = prompt + caption

        features = self.preprocess_text_for_image_generation(
            text_input,
            gen_start_token=gen_start_token,
            uncond_prompt="",
            # uncond_prompt=prompt + "\n" if with_separator else prompt,
        )

        print(f"input string: {self.tokenizer.decode(features['input_ids'])}", flush=True)
        if image_gen_cfg > 1.0:
            features['input_ids'] = torch.stack((features['input_ids'], features['input_ids']), dim=0)
            features['attention_mask'] = torch.stack((features['attention_mask'], features['uncond_attention_mask']), dim=0)

        image_tensor = self.forward_for_image_generation(
            features,
            image_gen_cfg=image_gen_cfg,
            cfg_schedule=cfg_schedule,
            image_gen_temperature=image_gen_temperature,
            cfg_renorm_type=cfg_renorm_type,
            time_shifting_factor=time_shifting_factor,
        )
        return image_tensor

    def preprocess_text_for_image_generation(
        self, 
        text_input,
        gen_start_token,
        uncond_prompt
    ):
        roles = ["human", "gpt"]
        sources = [
            {"from": "human", "value": text_input},
            {"from": "gpt", "value": gen_start_token},
        ]

        # Skip the first one if it is not from human
        if sources[0]["from"] != "human":
            sources = sources[1:]

        input_text = ""
        input_ids = []
        attention_mask = []
        uncond_text_ids = []
        uncond_attention_mask = []

        self.usr_prefix = GLM_USER_PREFIX
        self.assistant_prefix = GLM_ASSISTANT_PREFIX
        self.usr_prefix_id = (self.tokenizer(self.usr_prefix, return_tensors="pt")["input_ids"][0]).tolist()
        self.assistant_prefix_id = (self.tokenizer(self.assistant_prefix, return_tensors="pt")["input_ids"][0]).tolist()

        for j, sentence in enumerate(sources):
            role = sentence["from"]
            if j % 2 == 0:
                assert role == roles[0]  # user
                question = sentence["value"]

                input_text += self.usr_prefix
                input_ids.extend(self.usr_prefix_id)
                uncond_text_ids.extend(self.usr_prefix_id)

                input_text += question  # <role>HUMAN</role>Please generate the corresponding image based on the description.\nDraw a beautiful girl.'
                question_id = (self.tokenizer(question, return_tensors="pt")["input_ids"][0]).tolist()
                input_ids.extend(question_id)

                uncond_question_id = (self.tokenizer(uncond_prompt, return_tensors="pt")["input_ids"][0]).tolist()
                uncond_text_ids.extend(uncond_question_id)

                # 获取当前 user turn 处理完后 input_ids 的总长度
                # 这是为当前 turn 创建的所有 mask 的目标长度
                current_turn_len = len(input_ids)

                # 标准 attention_mask 总是全1
                attention_mask.extend([1] * current_turn_len)
                uncond_end_idx = len(self.usr_prefix_id) + len(uncond_question_id)
                # uncond_attention_mask: 只有 prefix 和 uncond_prompt 是 1
                uncond_attention_mask.extend([1] * uncond_end_idx + [0] * (current_turn_len - uncond_end_idx))
                assert len(attention_mask) == len(uncond_attention_mask)

            else:  # assistant
                assert role == roles[1]
                input_text += self.assistant_prefix
                input_ids.extend(self.assistant_prefix_id)

                input_text += sentence["value"]  # '<role>HUMAN</role>Please generate the corresponding image based on the description.\nDraw a beautiful girl.<role>ASSISTANT</role><image>'
                answer_id = (self.tokenizer(sentence["value"], return_tensors="pt")["input_ids"][0]).tolist()

                input_ids.extend(answer_id)
                attention_mask.extend([1] * (len(input_ids) - len(attention_mask)))
                uncond_attention_mask.extend([1] * (len(input_ids) - len(uncond_attention_mask)))
                assert len(attention_mask) == len(uncond_attention_mask)

        assert len(attention_mask) == len(input_ids)
        assert len(attention_mask) == len(uncond_attention_mask) 

        return dict(
            input_ids=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask),
            uncond_attention_mask=torch.tensor(uncond_attention_mask),
            input_text=input_text,  # just for debug, '<role>HUMAN</role>Please generate the corresponding image based on the description.\nDraw a beautiful girl.<role>ASSISTANT</role><image>'
        )

    @torch.no_grad()
    def forward_for_image_generation(
        self,
        samples,
        image_gen_cfg=1.0,
        cfg_schedule="constant",
        image_gen_temperature=1.0,
        cfg_renorm_type=None,
        time_shifting_factor=None
    ):
        cur_text = samples['input_ids'].cuda()
        attention_mask = samples['attention_mask'].cuda()
        if len(cur_text.shape) == 1:
            cur_text = cur_text.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        input_embeds = self.model.model.word_embeddings(cur_text)

        output_tokens = []
        past_key_values = None
        feat_dec_past_key_values = None

        visual_position_indicators = cur_text == self.model.config.image_start_token
        if self.model.config.image_patch_token in cur_text:
            im_patch_mask = cur_text == self.model.config.image_patch_token
            visual_position_indicators = torch.logical_or(visual_position_indicators, im_patch_mask)
        max_token_length_for_gen = 256

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for token_idx in tqdm(range(max_token_length_for_gen)):
                position_ids = attention_mask.long().cumsum(-1) - 1
                if past_key_values is not None:
                    position_ids = position_ids[:, -1:]

                if cfg_schedule == "linear":
                    cfg_iter = 1 + (image_gen_cfg - 1) * (256 - token_idx) / 256
                elif cfg_schedule == "linear-reverse":
                    cfg_iter = 1 + (image_gen_cfg - 1) * token_idx / 255
                elif cfg_schedule == "constant":
                    cfg_iter = image_gen_cfg
                else:
                    raise NotImplementedError
                output_model = self.model.forward_for_image_generation_inner(
                    inputs_embeds=input_embeds,
                    labels=None,
                    labels_img=None,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    image_token_id=token_idx,
                    image_gen_temperature=image_gen_temperature,
                    image_gen_cfg=cfg_iter,
                    visual_position_indicators=visual_position_indicators,
                    cfg_renorm_type=cfg_renorm_type,
                    time_shifting_factor=time_shifting_factor,
                    use_cache=True,
                )
                output_token_gen = output_model[0]
                past_key_values = output_model[1]

                feat_dec_out = self.vision.forward_feature_decoder(
                    output_token_gen,
                    past_key_values=feat_dec_past_key_values,
                )
                feat_dec_past_key_values = feat_dec_out['past_key_values']
                output_token = feat_dec_out['x_norm_patchtokens']
                output_tokens.append(output_token)

                input_embeds = self.linear_proj(output_token)

                attention_mask = torch.cat(
                    (attention_mask, torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype, device=attention_mask.device)),
                    dim=-1
                )
                visual_position_indicators = torch.ones(input_embeds.shape[:2], dtype=torch.bool)

        image_tensor = self.vision.forward_pixel_decoder(torch.cat(output_tokens, dim=1))
        return image_tensor

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        image_gen: Optional[bool] = False,
        image_gen_steps: Optional[int] = 30,
        image_gen_seed: Optional[int] = 0,
        image_gen_cfg: Optional[float] = 3.5,
        image_gen_height: Optional[int] = 512,
        image_gen_width: Optional[int] = 512,
        image_gen_prompt: Optional[str] = "",
        **generate_kwargs,
    ):
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        
        if image_gen:
            cfg_schedule = "constant"
            image_gen_temperature = 1.0
            cfg_renorm_type = None
            time_shifting_factor = None
            image_gen_cfg = 3.5
            with_separator = True

            image_tensor = self.unified_image_generation(
                caption=image_gen_prompt,
                image_gen_cfg=image_gen_cfg,
                with_separator=with_separator,
                cfg_schedule=cfg_schedule,
                image_gen_temperature=image_gen_temperature,
                cfg_renorm_type=cfg_renorm_type,
                time_shifting_factor=time_shifting_factor,
            )
            return image_tensor

        else:
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
                # import pdb; pdb.set_trace()
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=words_embeddings,
                    use_cache=use_cache,
                    image_mask=image_mask,
                    audio_mask=audio_mask,
                    rope_deltas=None,
                    **generate_kwargs,
                )
        return outputs
