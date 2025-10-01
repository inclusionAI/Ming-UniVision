from functools import partial
import math
import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint

from einops import rearrange

from .layers import Mlp, PatchEmbed, SwiGLUFFNFused, Attention, MemEffAttention, NestedTensorBlock as Block, CausalBlock
from .layers import CausalAttention, MemEffCausalAttention

from transformers.cache_utils import DynamicCache

logger = logging.getLogger("mingtok")

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x
    
    def forward_with_mid_feats(self, x, out=None):
        for b in self:
            x = b(x)
            if out is None:
                out = (x,)
            else:
                out += (x,)
        return x, out

class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        initialization_method="trunc_normal",
        out_dim=None,
        out_projection_layer=None
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # with class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for _ in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        assert out_dim is not None
        self.out_dim = out_dim
        self.out_projection_layer = out_projection_layer
        self.downsample_ratio = 1

        self.out_norm = norm_layer(embed_dim)
        self.out_act = nn.GELU()
        self.out_proj = nn.Linear(embed_dim, out_dim)
    
    def forward_out_layer(self, x):
        x_shortcut = rearrange(x, "b n (c h) -> b n c h", c=self.out_dim).mean(-1)
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_proj(x)
        return x_shortcut + x
    
    def get_num_patches(self, h, w):
        return self.patch_embed.num_patches

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        patch_pos_embed = pos_embed[:, :-1]
        class_pos_embed = pos_embed[:, -1]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((patch_pos_embed, class_pos_embed.unsqueeze(0)), dim=1).to(previous_dtype)

    # with class token
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((x, self.cls_token.expand(x.shape[0], -1, -1)), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
    
        for blk in self.blocks:
            x = blk(x)

        x_norm = self.forward_out_layer(x)

        return x_norm

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        loss_type="L2",
        norm_pix_loss=True,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        initialization_method="trunc_normal",
        require_head=False,
        num_register_tokens=0,
        with_cls_token=True,
        in_dim=None,
        in_projection_layer=None,
        enable_lpips_loss=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.loss_type = loss_type

        self.norm_pix_loss = norm_pix_loss

        self.initialization_method = initialization_method

        self.in_dim = in_dim
        self.in_projection_layer = in_projection_layer
        if in_dim is None:
            in_dim = self.embed_dim
            self.in_proj = None
        else:
            if in_projection_layer is None:
                self.in_proj = nn.Linear(in_dim, embed_dim)

        self.num_cls_tokens = with_cls_token * 1
        print(f"num cls tokens {self.num_cls_tokens}")

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                layer_idx=i,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.require_head = require_head
        if require_head:
            self.head = nn.Linear(embed_dim, patch_size**2 * 3, bias=True)
        else:
            self.head = nn.Identity()

    def get_num_patches(self, h=None, w=None):
        return self.patch_embed.num_patches
    
    def forward_in_projection_layer(self, x):
        if self.in_proj is not None:
            x_shortcut = rearrange(
                x.unsqueeze(-1).repeat(1,1,1,int(self.embed_dim//self.in_dim)),
                "b n c h -> b n (c h)"
            )
            x = self.in_proj(x) + x_shortcut
        return x

    def forward_features(
        self, 
        x, 
        masks=None, 
        use_cache=False, 
        past_key_values=None, 
        cache_position=None
    ):

        x = self.forward_in_projection_layer(x)

        B, N, C = x.shape

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + N, device=x.device
            )

        hidden_states = x
        for blk in self.blocks:
            if use_cache:
                if self.chunked_blocks:
                    for blk_inner in blk:
                        layer_outputs = blk_inner(
                            hidden_states,
                            use_cache=use_cache,
                            past_key_value=past_key_values,
                            cache_position=cache_position
                        )
                        hidden_states = layer_outputs[0]
                        next_decoder_cache = layer_outputs[1]
                else:
                    layer_outputs = blk(
                        hidden_states,
                        use_cache=use_cache,
                        past_key_value=past_key_values,
                        cache_position=cache_position
                    )
                    hidden_states = layer_outputs[0]
                    next_decoder_cache = layer_outputs[1]
            else:
                hidden_states = blk(hidden_states)

        next_cache = next_decoder_cache if use_cache else None

        x_norm = self.norm(hidden_states)
        if self.num_cls_tokens > 0 and N > 1:
            x_norm_patch_tokens = x_norm[:, : -self.num_cls_tokens]
            return {
                "x_norm_patchtokens": x_norm_patch_tokens,
                "x_norm_clstoken": x_norm[:, -self.num_cls_tokens],
                "x_norm": x_norm,
                "x_prenorm": x,
            }
        elif N > 1:
            return {
                "x_norm_patchtokens": x_norm,
                "x_norm_clstoken": None,
                "x_prenorm": x,
            }
        else:
            return {
                "x_prenorm": x,
                "x_norm_patchtokens": x_norm,
                "past_key_values": next_cache,
            }
        
    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if self.require_head:
            return self.head(ret['x_norm_patchtokens'])
        else:
            return ret
        
    def get_num_layer(self, var_name):
        if any([var_name in ("cls_token", "pos_embed"), var_name.startswith("patch_embed")]):
            print(f"{var_name}: 0")
            return 0
        elif var_name.startswith("in_proj"):
            return 0
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[2])
            print(f"{var_name}: {layer_id + 1}")
            return layer_id+1
        else:
            print(f"{var_name}: {len(self.blocks[0])}")
            return len(self.blocks[0])

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        dtype = imgs.dtype
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x.to(dtype)
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def compute_reconstruction_loss(self, imgs, pred, mask=None):
        target = self.patchify(imgs)
        if mask is not None:
            b, hw_mask = mask.shape
            h_mask = w_mask = int(math.sqrt(hw_mask))
            hw_tgt = target.shape[1]

            repeat_num = int(math.sqrt(hw_tgt // hw_mask))

            mask = mask.reshape(b, h_mask, 1, w_mask, 1).repeat(1,1,repeat_num,1,repeat_num).reshape(b, -1)
            mask = ~mask
        if self.loss_type == "L2":
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5
            
            loss = (pred - target) ** 2
        elif self.loss_type == "L1":
            pred = 2. * (pred - pred.min()) / (pred.max() - pred.min()) - 1.
            loss = torch.abs(pred - target)
        elif self.loss_type == "L1-tanh":
            pred = torch.tanh(pred)
            loss = torch.abs(pred - target)
        elif self.loss_type == "L1-plain":
            loss = torch.abs(pred - target)
        else:
            raise NotImplementedError
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if mask is None:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on kept patches
        return loss
    
    def compute_lpips_loss(self, imgs, pred):
        pred = self.unpatchify(pred)
        if self.perceptual_loss is not None:
            loss = self.perceptual_loss(pred, imgs).mean()
        else:
            loss = 0.0
        return loss

def decoder(patch_size=16, embed_dim=1024, depth=1, loss_type="L2", norm_pix_loss=True, fa_enable=False, frozen=False, **kwargs):
    if fa_enable:
        attn_class = MemEffAttention
    else:
        attn_class = Attention
    model = TransformerDecoder(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=embed_dim//64,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=attn_class),
        loss_type=loss_type,
        norm_pix_loss=norm_pix_loss,
        require_head=True,
        num_register_tokens=0,
        with_cls_token=False,
        **kwargs,
    )
    if frozen:
        for name, param in model.named_parameters():
            param.requires_grad = False
        model = model.eval()
        model.training = disabled_train
        logging.info("freeze decoder")
    return model

def causal_decoder(
    patch_size=16, 
    num_register_tokens=0, 
    embed_dim=1024, 
    depth=1, 
    loss_type="L2", 
    norm_pix_loss=True, 
    fa_enable=False, 
    frozen=False, 
    with_cls_token=True, 
    **kwargs):
    if fa_enable:
        attn_class = MemEffCausalAttention
    else:
        attn_class = CausalAttention
    model = TransformerDecoder(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=embed_dim//64,
        mlp_ratio=4,
        block_fn=partial(CausalBlock, attn_class=attn_class),
        loss_type=loss_type,
        num_register_tokens=num_register_tokens,
        with_cls_token=with_cls_token,
        norm_pix_loss=norm_pix_loss,
        require_head=False,
        **kwargs,
    )
    if frozen:
        for name, param in model.named_parameters():
            param.requires_grad = False
        model = model.eval()
        model.training = disabled_train
        logging.info("causal decoder frozen")
        model.frozen = True
    else:
        logging.info("causal decoder unlocked")
        model.frozen = False
    return model

def build_low_level_encoder(backbone_config):

    img_size = backbone_config.get("img_size", 224)
    embed_dim = backbone_config.get("embed_dim", 1024)
    depth = backbone_config.get("depth", 24)
    patch_size = backbone_config.get("patch_size", 16)
    ffn_layer = backbone_config.get("ffn_layer", "mlp")
    initialization_method = backbone_config.get("initialization_method", "trunc_normal")
    fa_enable = backbone_config.get("fa_enable", True)
    frozen = backbone_config.get("frozen", True)
    out_dim = backbone_config.get("out_dim", None)

    if fa_enable:
        attn_class = MemEffAttention
    else:
        attn_class = Attention
    model = VisionTransformerEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=embed_dim//64,
        mlp_ratio=4,
        ffn_layer=ffn_layer,
        block_fn=partial(Block, attn_class=attn_class),
        initialization_method=initialization_method,
        out_dim=out_dim,
    )

    if frozen:
        for name, param in model.named_parameters():
            param.requires_grad = False
        model = model.eval()
        model.training = disabled_train
    return model

def build_semantic_decoder(decoder_config):

    patch_size = decoder_config.get("patch_size", 16)
    embed_dim = decoder_config.get("embed_dim", 1024)
    depth = decoder_config.get("decoder_depth", 1)
    ffn_layer = decoder_config.get("ffn_layer", "mlp")
    drop_path_rate = decoder_config.get("drop_path_rate", 0.0)
    norm_pix_loss = decoder_config.get("norm_pix_loss", True)
    initialization_method = decoder_config.get("initialization_method", "trunc_normal")
    fa_enable = decoder_config.get("fa_enable", True)
    frozen = decoder_config.get("frozen", False)
    in_dim = decoder_config.get("in_dim", None)
    in_projection_layer = decoder_config.get("in_projection_layer", None)
    return causal_decoder(patch_size, embed_dim=embed_dim, depth=depth, loss_type=None, norm_pix_loss=norm_pix_loss, drop_path_rate=drop_path_rate, initialization_method=initialization_method, fa_enable=fa_enable, frozen=frozen, ffn_layer=ffn_layer, in_dim=in_dim, in_projection_layer=in_projection_layer)

def build_pixel_decoder(decoder_config):
    patch_size = decoder_config.get("patch_size", 16)
    embed_dim = decoder_config.get("embed_dim", 1024)
    depth = decoder_config.get("decoder_depth", 1)
    drop_path_rate = decoder_config.get("drop_path_rate", 0.0)
    loss_type = decoder_config.get("loss_type", "L2")
    norm_pix_loss = decoder_config.get("norm_pix_loss", True)
    initialization_method = decoder_config.get("initialization_method", "trunc_normal")
    fa_enable = decoder_config.get("fa_enable", True)
    frozen = decoder_config.get("frozen", False)
    enable_lpips_loss = decoder_config.get("enable_lpips_loss", False)
    return decoder(patch_size, embed_dim=embed_dim, depth=depth, loss_type=loss_type, norm_pix_loss=norm_pix_loss, drop_path_rate=drop_path_rate, initialization_method=initialization_method, fa_enable=fa_enable, frozen=frozen, enable_lpips_loss=enable_lpips_loss)