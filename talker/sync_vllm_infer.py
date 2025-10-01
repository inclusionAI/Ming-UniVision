from vllm import ModelRegistry
from talker.ming_talker import MingTalkerForCausalLM
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
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Awaitable, Callable, Generator

from pydantic import BaseModel
from vllm.entrypoints.llm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, LLMEngine
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

def construct_vllm(model_path:str, enforce_eager: bool = False, gpu_memory_utilization=0.2):
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True, 
        enforce_eager=enforce_eager, 
        gpu_memory_utilization=gpu_memory_utilization, 
        disable_custom_all_reduce=True, 
        tensor_parallel_size=1,
    )

    llm_engine = LLMEngine.from_engine_args(engine_args)
    return llm_engine

def vllm_infer_generator(
    llm_engine: LLMEngine,
    prompt_token_ids, 
    prompt_embeds, 
    request_id=None,
    sampling_params=None):
    """
    请注意不要把一个 llm_engine 同时用在多个 vllm_infer_generator 中执行
    """

    request = TokensPrompt({
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

    llm_engine.add_request(request_id=request_id, prompt=request, params=sampling_params)

    # Streaming case
    def stream_results(llm_engine) -> Generator[Dict[str, Any], None, None]:
      while llm_engine.has_unfinished_requests(): 
        start = time.time()
        request_outputs = llm_engine.step()
        
        for request_output in request_outputs:
            # 仅处理当前请求ID的输出
            if request_output.request_id != request_id:
                continue
                
            output = request_output.outputs[0]
            yield {
                "text": output.text,
                "token_ids": output.token_ids[-1] if output.token_ids else None,
                "finish_reason": output.finish_reason
            }
    
    return stream_results(llm_engine)

def get_vllm_request_id():
    return str(uuid.uuid4())


# 以下代码用于单元测试

def warmup_vllm_engine(vllm_engine, prompt_token_ids, prompt_embeds):
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=50,
        skip_special_tokens=True,
        max_tokens=1024,
        stop=["<audio_eos>"]  # Qwen2的正确终止符[1,4](@ref)
    )
    output = vllm_infer_generator(vllm_engine, prompt_token_ids, prompt_embeds, get_vllm_request_id(), sampling_params=sampling_params)
    for ret in output:
        pass

def test_single_request_token_time_cost(vllm_engine, prompt_token_ids, prompt_embeds):
    
    # 单测单个 request  情况下的平均 token 耗时
    times = 1
    start_time = time.perf_counter()
    token_count = 0
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=50,
        min_tokens=10,
        skip_special_tokens=True,
        max_tokens=1024,
        stop=["<audio_eos>"]  # Qwen2的正确终止符[1,4](@ref)
    )

    for _ in range(times):
        results = []
        output = vllm_infer_generator(vllm_engine, prompt_token_ids, prompt_embeds, get_vllm_request_id(), sampling_params=sampling_params)
        for ret in output:
            token_count += 1
            results.append(ret)

    print(f"token count: {token_count}, {[it['token_ids'] for it in results]}")
    
    end_time = time.perf_counter()
    print(f"per token with single request generate time cost: {(end_time - start_time) * 1000 / token_count}")


def main():
    model_path = "/video_hy2/workspace/weilong.cwl/metax_models/bailingv4_moe_lite_addmetax_0716/talker"
    
    print("first")
    vllm_engine = construct_vllm(model_path=model_path, enforce_eager=False, gpu_memory_utilization=0.2)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        total = torch.cuda.get_device_properties(device).total_memory  # 总显存
        allocated = torch.cuda.memory_allocated(device)  # 已分配显存
        reserved = torch.cuda.memory_reserved(device)   # 保留显存
        free = total - allocated - reserved             # 剩余显存
        print(f"总显存: {total/1024**3:.2f} GB")
        print(f"已分配: {allocated/1024**3:.2f} GB")
        print(f"保留显存: {reserved/1024**3:.2f} GB")
        print(f"剩余显存: {free/1024**3:.2f} GB")


    prompt_token_ids = torch.load('/mnt1/zhoubofan/prompt_token_ids.pt')
    prompt_embeds = torch.load('/mnt1/zhoubofan/prompt_embeds.pt')
    
    warmup_vllm_engine(vllm_engine, prompt_token_ids, prompt_embeds)

    test_single_request_token_time_cost(vllm_engine, prompt_token_ids, prompt_embeds)
    # test_multi_request_token_count(vllm_engine, prompt_token_ids, prompt_embeds)


if __name__ == "__main__":
    main()


