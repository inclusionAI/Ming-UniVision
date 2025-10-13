# coding=utf-8
# Copyright 2022 shunxing1234 and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLM model configuration """

from typing import Dict

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

GLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shunxing1234/GLM": "https://huggingface.co/shunxing1234/GLM/resolve/main/config.json",
    # See all GLM models at https://huggingface.co/models?filter=glm
}


class GLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GLMModel`].
    It is used to instantiate an GLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the GLM [shunxing1234/GLM-base-cased](https://huggingface.co/shunxing1234/GLM-base-cased) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the GLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~GLMModel`] or
            [`~TFGLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~GLMModel`] or
            [`~TFGLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        last_logits_l2_alpha ('float', *optional*, defaults to -1.0):
            Whether use l2 norm for last output logits.
            If < 0, will not compute last logits l2 norm,
            elif == 0, will compute l2 norm but not plus in the loss,
            while > 0, will plus this loss in the total loss.
        rotary_type (`str` or `function`, *optional*, defaults to `"none"`):
            The Rotary Embedding type to used in SelfAttention.
            If string, `"none"`, `"1d"`, `"2d"` are supported.
        unidirectional ('bool', *optional*, defaults to `False`):
            Whether or not the model is train with prefix LM or causal LM.
        Example:

    ```python
    >>> from transformers import GLMModel, GLMConfig

    >>> # Initializing a GLM shunxing1234/GLM-base-cased style configuration
    >>> configuration = GLMConfig()

    >>> # Initializing a model from the shunxing1234/GLM-base-cased style configuration
    >>> model = GLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "glm"
    attribute_map = {"num_hidden_layers": "num_layers"}

    def __init__(
        self,
        num_layers=24,
        vocab_size=30592,
        hidden_size=1024,
        num_experts=1,
        expert_capacity=None,
        moe_config: Dict = {},
        num_attention_heads=16,
        num_key_value_heads=0,
        embedding_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        max_sequence_length=512,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        parallel_output=True,
        relative_encoding=False,
        block_position_encoding=True,
        output_predict=False,
        spell_length=None,
        spell_func="lstm",
        attention_scale=1.0,
        initializer_range=0.02,
        pool_token="cls",
        max_memory_length=0,
        bf16=True,
        intermediate_size=None,
        last_logits_l2_alpha=-1.0,
        rotary_type='none',
        use_rmsnorm=False,
        use_atorch_rmsnorm=False,
        use_swiglu=False,
        rope_scaling=1.0,
        use_cache=True,
        focused_attention=False,
        cache_in_memory=False,
        attention_grouping=None,
        output_hidden_states=False,
        tie_word_embeddings=True,
        unidirectional=False,
        use_bias=True,
        use_qkv_bias=False,
        mlp_version='v1',
        norm_softmax=False,
        norm_head=False,
        num_decoder_image_token=1024,
        num_decoder_audio_token=512,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.moe_config = moe_config
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        self.output_predict = output_predict
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale
        self.initializer_range = initializer_range
        self.pool_token = pool_token
        self.max_memory_length = max_memory_length
        self.bf16 = bf16
        self.intermediate_size = intermediate_size
        self.last_logits_l2_alpha = last_logits_l2_alpha
        self.rotary_type = rotary_type
        self.use_rmsnorm = use_rmsnorm
        self.use_atorch_rmsnorm = use_atorch_rmsnorm
        self.use_swiglu = use_swiglu
        self.rope_scaling = rope_scaling
        self.use_cache = use_cache
        self.focused_attention = focused_attention
        self.cache_in_memory = cache_in_memory
        self.attention_grouping = attention_grouping
        self.unidirectional = unidirectional
        self.use_bias = use_bias
        self.use_qkv_bias = use_qkv_bias
        self.mlp_version = mlp_version
        self.norm_softmax = norm_softmax
        self.norm_head = norm_head
        self.num_decoder_image_token = num_decoder_image_token
        self.num_decoder_audio_token = num_decoder_audio_token

        super().__init__(output_hidden_states=output_hidden_states, tie_word_embeddings=tie_word_embeddings, **kwargs)

