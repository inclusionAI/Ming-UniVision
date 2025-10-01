from vllm import ModelRegistry
from ming_talker import MingTalkerForCausalLM
from vllm.inputs import TokensPrompt as LLMInputs
from vllm import LLM, SamplingParams

ModelRegistry.register_model("MingTalkerForCausalLM", MingTalkerForCausalLM)

from vllm import LLMEngine, SamplingParams, RequestOutput
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoTokenizer
import uuid 

from vllm.inputs.data import TokensPrompt
import torch

import os, sys


import asyncio
import json
import time
import torch
import ssl
import math
from enum import Enum
from argparse import Namespace
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Awaitable, Callable

from pydantic import BaseModel

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.outlines_decoding import GuidedDecodingMode
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser,
                        random_uuid)
from vllm.version import __version__ as VLLM_VERSION
from vllm.inputs import TokensPrompt

def construct_vllm_and_tokenizer(model_path:str, enforce_eager: bool = False, gpu_memory_utilization=0.2):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    engine_args = AsyncEngineArgs(
        model=model_path,
        trust_remote_code=True, 
        enforce_eager=enforce_eager, 
        gpu_memory_utilization=gpu_memory_utilization, 
        disable_custom_all_reduce=True, 
        tensor_parallel_size=1,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine, tokenizer

async def default_async_func() -> bool:
    return False  # 默认返回 False

def vllm_infer_generator(
    vllm_engine: AsyncLLMEngine, 
    prompt_token_ids, 
    prompt_embeds, 
    request_id, 
    sampling_params=None, 
    is_cancelled: Callable[[], Awaitable[bool]] = default_async_func):

    requests = TokensPrompt({
        "prompt_token_ids": prompt_token_ids,
        "multi_modal_data": {"audio": prompt_embeds.unsqueeze(0).cpu()}
    })
    
    if sampling_params == None:
      sampling_params = SamplingParams(
          temperature=0.0,
          top_k=50,
          skip_special_tokens=True,
          max_tokens=1024,
          stop=["<audio_eos>"]  # Qwen2的正确终止符[1,4](@ref)
      )

    output_generator = vllm_engine.generate(requests, sampling_params=sampling_params, request_id=request_id)

    # Streaming case
    async def stream_results(results_generator) -> AsyncGenerator[dict, None]:
        async for request_output in results_generator:
            text_outputs = []
            token_ids = []
            finish_reason = []
            
            for output in request_output.outputs:
                text_outputs.append(output.text)
                if output.token_ids:
                    token_ids.append(output.token_ids[-1])
                finish_reason.append(output.finish_reason if output.finish_reason else None)

            ret = {
                "text": request_output.outputs[0].text,
                "token_ids": request_output.outputs[0].token_ids[-1],
                "finish_reason": request_output.outputs[0].finish_reason if request_output.outputs[0].finish_reason else None,
            }
            yield ret
    
    return stream_results(output_generator)

def get_vllm_request_id():
    return str(uuid.uuid4())


# 以下代码用于单元测试

async def warmup_vllm_engine(vllm_engine, prompt_token_ids, prompt_embeds):
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=50,
        skip_special_tokens=True,
        max_tokens=1024,
        stop=["<audio_eos>"]  # Qwen2的正确终止符[1,4](@ref)
    )
    output = vllm_infer_generator(vllm_engine, prompt_token_ids, prompt_embeds, get_vllm_request_id(), sampling_params=sampling_params)
    async for ret in output:
        pass

async def test_single_request_token_time_cost(vllm_engine, prompt_token_ids, prompt_embeds):
    
    # 单测单个 request  情况下的平均 token 耗时
    times = 10
    start_time = time.perf_counter()
    token_count = 0
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=50,
        skip_special_tokens=True,
        max_tokens=1024,
        stop=["<audio_eos>"]  # Qwen2的正确终止符[1,4](@ref)
    )

    for _ in range(times):
        results = []
        output = vllm_infer_generator(vllm_engine, prompt_token_ids, prompt_embeds, get_vllm_request_id(), sampling_params=sampling_params)
        async for ret in output:
            token_count += 1
            results.append(ret)

    print(f"token count: {token_count}, {[it['token_ids'] for it in results]}")
    
    end_time = time.perf_counter()
    print(f"per token with single request generate time cost: {(end_time - start_time) * 1000 / token_count}")


async def test_multi_request_token_count(vllm_engine, prompt_token_ids, prompt_embeds):

    # 单测多个 request 并发情况下的平均 token 耗时
    
    start_time = time.perf_counter()


    async def test_multi_request_token_count_task():
        token_count = 0
        sampling_params = SamplingParams(
            temperature=0.0,
            top_k=50,
            skip_special_tokens=True,
            max_tokens=1024,
            stop=["<audio_eos>"]  # Qwen2的正确终止符[1,4](@ref)
        )
        times = 4
        for _ in range(times):
            output = vllm_infer_generator(vllm_engine, prompt_token_ids, prompt_embeds, get_vllm_request_id(), sampling_params=sampling_params)
            async for _ in output:
                token_count += 1

        return token_count
    
    concurrent_num = 4
    token_counts = await asyncio.gather(*[test_multi_request_token_count_task() for _ in range(concurrent_num)])

    print(f"token count: {sum(token_counts)}")
    end_time = time.perf_counter()
    print(f"per token with multi request generate time cost: {(end_time - start_time) * 1000 / sum(token_counts)}")


async def main():
    model_path = "/video_hy2/workspace/weilong.cwl/metax_models/bailingv4_moe_lite_addmetax_0716/talker"

    vllm_engine, _ = construct_vllm_and_tokenizer(model_path=model_path, enforce_eager=False, gpu_memory_utilization=0.4)
    prompt_token_ids = torch.load('/mnt1/zhoubofan/prompt_token_ids.pt')
    prompt_embeds = torch.load('/mnt1/zhoubofan/prompt_embeds.pt')
    
    await warmup_vllm_engine(vllm_engine, prompt_token_ids, prompt_embeds)

    await test_single_request_token_time_cost(vllm_engine, prompt_token_ids, prompt_embeds)
    # await test_multi_request_token_count(vllm_engine, prompt_token_ids, prompt_embeds)


    


if __name__ == "__main__":
    asyncio.run(main())


