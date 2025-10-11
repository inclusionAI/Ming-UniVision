import logging
import os
import warnings
import torch

from torch import Tensor
from torch import nn

from typing import Optional, Tuple, List

from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func
else:
    flash_attn_func = None

XFORMERS_ENABLED = os.environ.get("XFORMERS_ENABLED")
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind, LowerTriangularMask

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.layer_idx = layer_idx

        if flash_attn_func is not None:
            self.flash_attn = True
        else:
            self.flash_attn = False

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if XFORMERS_AVAILABLE:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = unbind(qkv, 2)

            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        else:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            if self.flash_attn and x.dtype!=torch.float32:

                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

                q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]

                x = flash_attn_func(q, k, v)
                x = x.reshape([B, N, C])

                x = self.proj(x)
                x = self.proj_drop(x)
                return x
            else:
                return super().forward(x)
            
class CausalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if flash_attn_func is not None:
            self.flash_attn = True
        else:
            self.flash_attn = False

        self.layer_idx = layer_idx

    def forward(
        self, 
        x: Tensor, 
        past_key_value: Optional[List[torch.FloatTensor]]=None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        attn = q @ k.transpose(-2, -1)

        attn_mask = self.prepare_causal_attention_mask(shape=(B,N), device=x.device)
        attn = attn.masked_fill(attn_mask.to(torch.bool), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, past_key_value)
    
    def prepare_causal_attention_mask(self, shape, device: torch.device) -> Tensor:
        """
            shape: (batch_size, sequence_length)
        """
        batch_size, seq_len = shape
        attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)

        return attention_mask.to(device)

class MemEffCausalAttention(CausalAttention):
    def forward(
        self, 
        x: Tensor, 
        attn_bias=None, 
        past_key_value: Optional[List[torch.FloatTensor]]=None,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Tuple:
        if XFORMERS_AVAILABLE:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

            q, k, v = unbind(qkv, 2)

            if past_key_value is not None:
                cache_kwargs = {"cache_position": cache_position}
                k, v = past_key_value.update(
                    k.transpose(1, 2), 
                    v.transpose(1, 2), 
                    self.layer_idx, 
                    cache_kwargs)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                x = memory_efficient_attention(q, k, v)
            else:
                x = memory_efficient_attention(q, k, v, attn_bias=LowerTriangularMask())
            x = x.reshape([B, N, C])

            x = self.proj(x)
            x = self.proj_drop(x)
            return (x, past_key_value)
        else:
            if not self.flash_attn:
                if attn_bias is not None:
                    raise AssertionError("xFormers is required for using nested tensors")
                return super().forward(x)
            else:
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

                q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]

                if past_key_value is not None:
                    cache_kwargs = {"cache_position": cache_position}
                    k, v = past_key_value.update(
                        k.transpose(1, 2), 
                        v.transpose(1, 2), 
                        self.layer_idx, 
                        cache_kwargs)
                    k = k.transpose(1, 2)
                    v = v.transpose(1, 2)
                    if q.dtype not in [torch.float16, torch.bfloat16]:
                        q = q.to(torch.bfloat16)
                        k = k.to(torch.bfloat16)
                        v = v.to(torch.bfloat16)
                    x = flash_attn_func(q, k, v, causal=True)
                else:
                    x = flash_attn_func(q, k, v, causal=True)
                x = x.reshape([B, N, C])

                x = self.proj(x)
                x = self.proj_drop(x)
                return (x, past_key_value)

            
            
