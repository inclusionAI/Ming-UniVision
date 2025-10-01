import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

import os
from typing import Callable, Optional
import warnings

from torch import Tensor
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import SwiGLU

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (SwiGLU)")
    else:
        warnings.warn("xFormers is disabled (SwiGLU)")
        raise ImportError
except ImportError:
    SwiGLU = SwiGLUFFN
    XFORMERS_AVAILABLE = False

    warnings.warn("xFormers is not available (SwiGLU)")


class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )


class RectifiedFlowLoss(nn.Module):
    """Rectified Flow Loss - 基于确定性ODE路径的生成模型"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, mlp_mult=1., grad_checkpointing=False):
        super(RectifiedFlowLoss, self).__init__()
        self.in_channels = target_channels

        # 转换num_sampling_steps从字符串为整数(与原DiffLoss保持兼容)
        self.num_sampling_steps = int(num_sampling_steps) if isinstance(num_sampling_steps, str) else num_sampling_steps
        print(f"num sampling steps for rf head: {self.num_sampling_steps}")

        # 保持原有网络结构不变，只更改输出通道数(不需要预测方差)
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # 只需要预测速度场
            z_channels=z_channels,
            num_res_blocks=depth,
            mlp_mult=mlp_mult,
            grad_checkpointing=grad_checkpointing
        )

        self.t_sample_strategy = "uniform"
    
    def set_t_sample_strategy(self, strategy="uniform"):
        print(f"Setting t_sample_strategy to {strategy}")
        self.t_sample_strategy = strategy

    def forward(self, target, z, mask=None, repa_target=None):
        """使用与原DiffLoss相同的参数接口计算Rectified Flow损失"""
        batch_size = target.shape[0]
        
        # 随机采样时间点 t ∈ [0, 1]
        if self.t_sample_strategy == "uniform":
            t = torch.rand(batch_size, device=target.device)
        elif self.t_sample_strategy == "lognorm":
            t_mid = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=target.device)
            t = 1 / (1 + torch.exp(-t_mid))

        # # 生成标准正态分布，然后映射到 [0, 1]
        sigma = 0.5
        beta = 0
        # t = torch.sigmoid(torch.randn(batch_size, device=target.device)*sigma+beta)   

        # 在t处插值得到x_t = (1-t)*x_0 + t*x_1，其中x_1是噪声，x_0是真实数据
        noise = torch.randn_like(target)
        t_view = t.view(-1, *([1] * (target.dim() - 1)))  # 适应任何维度的目标
        x_t = (1 - t_view) * target + t_view * noise

        # 计算真实速度场 v = x_0 - x_1
        true_velocity = target - noise

        # 预测速度场，保持与原始代码相同的接口
        pred_velocity = self.net(x_t, t, z)

        # 计算L2损失
        loss = torch.nn.functional.mse_loss(pred_velocity, true_velocity, reduction='none')
        loss = loss.mean(dim=-1)  # 在特征维度上平均

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    def sample(self, z, temperature=1.0, cfg=1.0, cfg_renorm_type=None, time_shifting_factor=None):
        """保持与原DiffLoss相同的接口进行采样"""
        batch_size = z.shape[0]
        device = z.device
        img_cfg = False
        text_cfg = cfg 
        image_cfg = 2.5
        cfg_mode = 0

        # 从环境变量获取CFG配置参数
        import os
        cfg_mode = int(os.environ.get('CFG_MODE', '1'))
        text_cfg = float(os.environ.get('TEXT_CFG', str(cfg)))
        image_cfg = float(os.environ.get('IMAGE_CFG', str(cfg)))
        # print('cfgmode:',cfg_mode,',text_cfg:',text_cfg,',image_cfg:',image_cfg)
        b_num = 4
        # import pdb; pdb.set_trace()

        # 初始化为纯噪声
        if cfg != 1.0 and not img_cfg:
            # 使用分类器引导时
            # print(f"Using cfg: {cfg}, temperature: {temperature}")
            noise = torch.randn(batch_size // 2, self.in_channels, device=device)
            noise = torch.cat([noise, noise], dim=0) * temperature
            use_cfg = True
            b_num=2
        elif cfg!=1.0 and img_cfg:
            noise = torch.randn(batch_size // 4, self.in_channels, device=device)
            noise = torch.cat([noise, noise,noise,noise], dim=0) * temperature
            use_cfg = True
            b_num=4
        else:
            # 不使用分类器引导
            noise = torch.randn(batch_size, self.in_channels, device=device) * temperature
            use_cfg = False
            b_num=1        
            
        # 使用欧拉法求解ODE
        x = noise
        steps = self.num_sampling_steps
        # import ipdb;ipdb.set_trace()
        if time_shifting_factor:
            time_steps = torch.linspace(0.0, 1.0, steps + 1, device=device)
            time_steps = time_steps / (time_steps + time_shifting_factor - time_shifting_factor * time_steps)
            time_steps = 1 - time_steps
            step_size = time_steps[:-1] - time_steps[1:]
            time_steps = time_steps[:-1]
        else:
            time_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)[:-1]
            step_size = 1.0 / steps
        
        for tid, t in enumerate(time_steps):
            # 创建batch的时间步
            t_batch = torch.ones(batch_size, device=device) * t
            
            # 预测速度场
            with torch.no_grad():
                if use_cfg:
                    # 分类器引导
                    half = x[: batch_size // b_num]
                    if b_num==2:
                        combined = torch.cat([half, half], dim=0)
                    elif b_num==4:
                        combined = torch.cat([half, half, half, half], dim=0)
                    v_combined = self.net(combined, t_batch, z)

                    if b_num==4:
                        v_full, v_uncond, v_img, v_text = v_combined.chunk(4)

                        # 根据 cfg_mode 选择组合公式
                        if cfg_mode == 1:
                            # M1: 链式引导，先图像，再文本
                            v_guided = v_uncond + image_cfg * (v_img - v_uncond) + text_cfg * (v_full - v_img)
                        elif cfg_mode == 2:
                            # M2: 示例中的一种模式
                            v_guided = v_uncond + 3 * (v_img - v_uncond) + text_cfg * (v_full - v_img)
                        elif cfg_mode == 3:
                            # M3: 文本条件作为基准进行图像引导
                            v_guided = v_text + image_cfg * (v_full - v_text)
                        elif cfg_mode == 4:
                            # M4: 链式引导，先文本，再图像
                            v_guided = v_uncond + text_cfg * (v_text - v_uncond) + image_cfg * (v_full - v_text)
                        elif cfg_mode == 5:
                            # M5: 图像条件作为基准进行文本引导
                            v_guided = v_img + text_cfg * (v_full - v_img)
                        elif cfg_mode == 6:
                            # M6: 平行引导，所有增量都基于 uncond
                            v_guided = v_uncond + text_cfg * (v_text - v_uncond) + image_cfg * (v_img - v_uncond)
                        elif cfg_mode == 0:
                            v_guided = v_uncond + text_cfg * (v_full - v_uncond)
                        elif cfg_mode == 10:
                            o = v_img + text_cfg * (v_full - v_img)
                            v_guided = v_text + image_cfg * (o - v_text)
                        else:
                            raise ValueError(f"Unsupported cfg_mode: {cfg_mode}")

                        v = v_guided
                        v = torch.cat([v, v, v, v], dim=0)
                    else:
                        v_cond, v_uncond = torch.split(v_combined, batch_size // 2, dim=0)
                        v = v_uncond + cfg * (v_cond - v_uncond)
                        if cfg_renorm_type == "channel":
                            # cfg renorm
                            norm_v_cond = torch.norm(v_cond, dim=-1, keepdim=True)
                            norm_v = torch.norm(v, dim=-1, keepdim=True)
                            scale = (norm_v_cond / norm_v + 1e-8).clamp(min=0.0, max=1.0)
                            v = v * scale
                        # 重复以匹配原始形状
                        v = torch.cat([v, v], dim=0)
                else:
                    v = self.net(x, t_batch, z)

            # Euler步进：x_{t-dt} = x_t - v_t * dt
            if isinstance(step_size, float):
                x = x + v * step_size
            else:
                x = x + v * step_size[tid]
            
        return x

# 保留原有代码的其他类和函数(SimpleMLPAdaLN, TimestepEmbedder, ResBlock, FinalLayer等)
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self._freqs = None
        self._dim_for_freqs = None
        self._max_period_for_freqs = None

    def freqs(self, device, dim, max_period=10000):
        if self._freqs is None or max_period != self._max_period_for_freqs or dim != self._dim_for_freqs:
            print("Building freqs")
            self._dim_for_freqs = dim
            self._max_period_for_freqs = max_period

            half = dim // 2
            self._freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=device)
        return self._freqs

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        # freqs = torch.exp(
        #     -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        # ).to(device=t.device)
        args = t[:, None].float() * self.freqs(t.device, dim, max_period)[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels,
        mlp_mult=1.,
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        print(f"hidden feature for ResBlock: {channels*mlp_mult}")
        self.mlp = SwiGLUFFNFused(
            in_features=channels,
            hidden_features=int(channels*mlp_mult),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        mlp_mult=1.,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                mlp_mult,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # def forward(self, x, t, c):
    #     """
    #     Apply the model to an input batch.
    #     :param x: an [N x C] Tensor of inputs.
    #     :param t: a 1-D batch of timesteps.
    #     :param c: conditioning from AR transformer.
    #     :return: an [N x C] Tensor of outputs.
    #     """
    #     # import ipdb;ipdb.set_trace()
    #     x = self.input_proj(x)
    #     t = self.time_embed(t)
    #     c = self.cond_embed(c)

    #     y = t + c

    #     if self.grad_checkpointing and not torch.jit.is_scripting():
    #         for block in self.res_blocks:
    #             x = checkpoint(block, x, y)
    #     else:
    #         for block in self.res_blocks:
    #             x = block(x, y)

    #     return self.final_layer(x, y)

    # 以下是SimpleMLPAdaLN的forward方法，需要适应时间步t是标量而非整数
    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps (0 to 1).
        :param c: conditioning from AR transformer.
        :return: velocity field prediction
        """
        x = self.input_proj(x)
        # 将t从[0,1]映射到网络期望的范围
        t = t * 1000  # 简单缩放让embedding有意义
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

if __name__ == "__main__":
    # a = RectifiedFlowLoss(
    #     target_channels=32, 
    #     z_channels=3584, 
    #     depth=12, 
    #     width=3072, 
    #     num_sampling_steps="16", 
    #     mlp_mult=4., 
    #     grad_checkpointing=False)
    # num = 0
    # for k, v in a.named_parameters():
    #     num += v.numel()
    # print(num)
    a = TimestepEmbedder(1024)
    print("running b")
    b = a(torch.rand(8, device="cpu"))
    print("running c")
    c = a(torch.rand(8, device="cpu"))
    print("running d")
    d = a(torch.rand(8, device="cpu"))
    print("ok")

    # ckpt = torch.load("/video_hy2/workspace/yuandan.zdd/diffusion_ar_vgen/logs_diffonly/test_h20_uniae_huge_64node_new1024_gpt2.1_rope2_vae_uniae_huge_uniaediffonly_pishi_1e-42025-04-14_21:39:13/checkpoint-40.pth", map_location='cpu')
    # import pdb; pdb.set_trace()
    # print("")
