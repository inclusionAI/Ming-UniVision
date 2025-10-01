from vllm.model_executor.models.qwen2 import *
from vllm import ModelRegistry
from vllm.model_executor import SamplingMetadata

from vllm.attention import AttentionMetadata
from vllm.sequence import IntermediateTensors
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, Sampler
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.profiling import ProcessorInputs
from vllm.inputs import INPUT_REGISTRY, InputContext, token_inputs
from vllm.multimodal.inputs import MultiModalInputs, PlaceholderRange

from typing import Optional, List, Union, Iterable, Tuple, Dict
from functools import lru_cache, partial
import torch
from torch import nn
import torch.nn.functional as F
import copy
import json
import sys
from vllm.model_executor.models.interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)

from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptUpdate)
from vllm.multimodal.inputs import (MultiModalFieldConfig,
                                    MultiModalKwargs, MultiModalDataDict)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from transformers import PretrainedConfig, BatchFeature, ProcessorMixin
from collections.abc import Mapping, Sequence
import traceback
from vllm.inputs import InputProcessingContext

class MingTalkerProcessingInfo(BaseProcessingInfo):

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
    
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}


class MingTalkerMultiModalProcessor(BaseMultiModalProcessor[MingTalkerProcessingInfo]):
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio=MultiModalFieldConfig.batched("audio"),
        )
    
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audio_inputs = mm_data.get("audio", None)
        
        mm_inputs = {}
        if audio_inputs is not None:
            mm_inputs['audio'] = audio_inputs
        
        return BatchFeature({
            "input_ids": prompt,
            **mm_inputs,
        })
    
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        return []

    
    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        ...

    
    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:

        mm_items = self._to_mm_items(dict((k, v.float().cpu().numpy()) for k, v in mm_data.items()))

        mm_hashes = (self._hash_mm_items(mm_items, hf_processor_mm_kwargs)
                     if return_mm_hashes else None)

        
        processed_data = self._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=hf_processor_mm_kwargs,
        )

        field_configs = self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs=hf_processor_mm_kwargs)

        mm_kwargs = MultiModalKwargs.from_hf_inputs(processed_data, field_configs)
        
        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders={"audio": [PlaceholderRange(offset=0, length=len(prompt), is_embed=None)]},
        )

class MingTalkerDummyInputsBuilder(BaseDummyInputsBuilder):
    """Framework compatibility only - for memory profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> list[int]:
        ...
    
    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        ...
    
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        audio_count = mm_counts.get("audio", 0)
        return ProcessorInputs(
            prompt_text = [0] * 1,
            mm_data = {"audio": torch.rand((1, 896)).cpu()}
        )


from vllm.model_executor.models.utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

@MULTIMODAL_REGISTRY.register_processor(
    MingTalkerMultiModalProcessor,
    info=MingTalkerProcessingInfo,
    dummy_inputs=MingTalkerDummyInputsBuilder,
)
class MingTalkerForCausalLM(nn.Module, SupportsPP, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config
        self.text_pad_id = config.text_pad_id

        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)
        
        # 初始化sampler
        self.sampler = Sampler()

        self.make_empty_intermediate_tensors = (self.model.make_empty_intermediate_tensors)
        
    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None:
            return multimodal_embeddings[0]
        else:
            inputs_embeds = self.model.get_input_embeddings(input_ids) + self.model.get_input_embeddings(torch.ones_like(input_ids).to(input_ids) * self.text_pad_id)
        return inputs_embeds
    

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        audio_embeds = kwargs.get('audio', None)
        return audio_embeds.squeeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        if intermediate_tensors is not None:
            inputs_embeds = None
            traceback.print_stack()
        elif inputs_embeds is None:
            audio = kwargs.get("audio", None)
            if audio is not None:
                intermediate_tensors = IntermediateTensors({"audio_embeddings": audio})
                # traceback.print_stack()
            inputs_embeds = self.get_input_embeddings(input_ids)
            input_ids = None

        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.model.compute_logits(hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        
        next_tokens = self.sampler(logits, sampling_metadata)

        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        processed_weights = []
        for k, v in weights:
            if not k.startswith('model.'):
                continue
            processed_weights.append((k, v))
            
        loader = AutoWeightsLoader(self)
        return loader.load_weights(processed_weights)
