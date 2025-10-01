"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
from vllm import ModelRegistry
from ming_talker import MingTalkerForCausalLM

ModelRegistry.register_model("MingTalkerForCausalLM", MingTalkerForCausalLM)

import asyncio
import json
import torch
import ssl
import math
from enum import Enum
from argparse import Namespace
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
from vllm.model_executor.guided_decoding.outlines_decoding import GuidedDecodingMode
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser, # iterate_with_cancellation,
                        random_uuid)
from vllm.version import __version__ as VLLM_VERSION
from vllm.inputs import TokensPrompt

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

logger = init_logger("vllm.entrypoints.api_server")


class GuidedDecodingParams:
    guided_decoding_backend: str
    guided_choice: Optional[List[str]]
    guided_json: Optional[Union[str, dict, BaseModel]]
    guided_regex: Optional[str]
    guided_grammar: Optional[str]
    guided_whitespace_pattern: Optional[str]


class StatusCodeEnum(Enum):
    """状态码枚举类"""

    OK = (0, 'success')
    ERROR = (-1, 'system_error')
    SERVER_ERR = (500, 'service_error')
    NECESSARY_PARAM_ERR = (4001, 'invalid_params')
    PARAM_ERR = (4002, 'params_error')

    @property
    def code(self):
        """获取状态码"""
        return self.value[0]

    @property
    def errmsg(self):
        """获取状态码信息"""
        return self.value[1]

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.get("/abort")
async def abort() -> Response:
    ...


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt_token_ids: List of token IDs for the prompt
    - prompt_embeds: Audio embeddings for the prompt
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt_token_ids = request_dict.pop("prompt_token_ids", None)
    prompt_embeds = request_dict.pop("prompt_embeds", None)
    if prompt_embeds and not isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = torch.tensor(prompt_embeds)
    
    if not prompt_token_ids:
        logger.error(f"param error, request: {request_dict}")
        ret = {"code": StatusCodeEnum.NECESSARY_PARAM_ERR.code, "msg": "缺少必传参数prompt_token_ids", "text": ""}
        return JSONResponse(ret)
        
    stream = request_dict.pop("stream", False)
    trace_id = request_dict.pop("trace_id", None)
    request_id = trace_id if trace_id else random_uuid()

    try:
        sampling_params = SamplingParams(**request_dict)
    except (ValueError, TypeError) as e:
        ret = {"code": StatusCodeEnum.PARAM_ERR.code, "msg": str(e), "text": ""}
        return JSONResponse(ret)

    try:
        # Create TokensPrompt with both token_ids and embeddings
        requests = TokensPrompt({
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"audio": prompt_embeds.unsqueeze(0)},
        })

        results_generator = engine.generate(requests, sampling_params, request_id)
        # results_generator = iterate_with_cancellation(
        #     results_generator, is_cancelled=request.is_disconnected)

        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async for request_output in results_generator:
                prompt_lens = []
                text_outputs = []
                token_ids = []
                finish_reason = []
                cumulative_logprob = []
                num_generated_tokens = []

                for output in request_output.outputs:
                    text_outputs.append(output.text)
                    if output.token_ids:
                        token_ids.append(output.token_ids[-1])
                    prompt_lens.append(len(request_output.prompt_token_ids))
                    finish_reason.append(output.finish_reason if output.finish_reason else 'false')
                    cumulative_logprob.append(inf2zero(output.cumulative_logprob))
                    num_generated_tokens.append(len(output.token_ids))

                if request_output.finished and request_output.metrics:
                    logger.info(f"{request_output.request_id} {request_output.metrics}")

                ret = {
                    "code": StatusCodeEnum.OK.code,
                    "msg": StatusCodeEnum.OK.errmsg,
                    "text": text_outputs,
                    "token_ids": token_ids,
                    "finish_reason": finish_reason,
                    "cumulative_logprob": cumulative_logprob,
                    "num_prompt_tokens": prompt_lens,
                    "num_generated_tokens": num_generated_tokens
                }
                yield (json.dumps(ret) + "\0").encode("utf-8")

        if stream:
            return StreamingResponse(stream_results())

        # Non-streaming case
        final_output = None
        try:
            async for request_output in results_generator:
                final_output = request_output
        except asyncio.CancelledError:
            return Response(status_code=499)

        assert final_output is not None
        text_outputs = []
        token_ids = []
        finish_reason = []

        for output in final_output.outputs:
            text_outputs.append(output.text)
            token_ids.append(list(output.token_ids))
            prompt_lens.append(len(final_output.prompt_token_ids))
            finish_reason.append(output.finish_reason if output.finish_reason else 'false')
            cumulative_logprob.append(inf2zero(output.cumulative_logprob))
            num_generated_tokens.append(len(output.token_ids))

        if request_output.finished and request_output.metrics:
            logger.info(f"{request_output.request_id} {request_output.metrics}")

        ret = {
            "code": StatusCodeEnum.OK.code,
            "msg": StatusCodeEnum.OK.errmsg,
            "text": text_outputs,
            "token_ids": token_ids[:-1],
            "finish_reason": finish_reason,
            "cumulative_logprob": cumulative_logprob,
            "num_prompt_tokens": prompt_lens,
            "num_generated_tokens": num_generated_tokens
        }
        return JSONResponse(ret)
    except Exception as e:
        logger.error("server error ", e)
        ret = {"code": StatusCodeEnum.SERVER_ERR.code, "msg": str(e), "text": ""}
        return JSONResponse(ret)

def inf2zero(val):
    if not val or math.isinf(val) or math.isnan(val):
        return 0.0
    return val


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s %(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": "/home/admin/logs/vllm_gw.log"
        },
        "access": {
            "formatter": "access",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": "/home/admin/logs/vllm_gw.log"

        },
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}


def build_app(args: Namespace) -> FastAPI:
    global app

    app.root_path = args.root_path
    return app


async def init_app(
        args: Namespace,
        llm_engine: Optional[AsyncLLMEngine] = None,
) -> FastAPI:
    app = build_app(args)

    global engine

    engine_args = AsyncEngineArgs.from_cli_args(args)
    print(f"init_app llm_engine: {llm_engine}, {engine_args}")
    engine = (llm_engine if llm_engine is not None else AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.API_SERVER))

    if not hasattr(app.state, 'engine_client'):
        app.state.engine_client = engine  # 动态适配新版

    return app


async def run_server(args: Namespace,
                     llm_engine: Optional[AsyncLLMEngine] = None,
                     **uvicorn_kwargs: Any) -> None:
    logger.info("vLLM bailing API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    app = await init_app(args, llm_engine)
    assert engine is not None

    shutdown_task = await serve_http(
        app,
        sock=None,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=parser.check_port, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))