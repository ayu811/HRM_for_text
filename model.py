# HRM-LLM (H: MQA Transformer, L: MLA Transformer) with 1-step gradient + deep supervision
# Author: you
# License: MIT

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def _init_lecun_trunc_normal_(tensor: torch.Tensor, std: float = 1.0, trunc: float = 2.0):
    # LeCun normal (fan-in) with truncation ≈ JAX lecun_normal + trunc
    fan_in = tensor.size(1) if tensor.ndim > 1 else tensor.numel()
    s = std / math.sqrt(fan_in)
    with torch.no_grad():
        size = tensor.shape
        tmp = torch.randn(size, device=tensor.device) * s
        tmp = torch.clamp(tmp, -trunc * s, trunc * s)
        tensor.copy_(tmp)
    return tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x


# -------- RoPE ----------
def apply_rope(q: torch.Tensor, k: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
    # q,k: [B, H, T, Dh]
    def rotate(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_ = (q * rope_cos) + (rotate(q) * rope_sin)
    k_ = (k * rope_cos) + (rotate(k) * rope_sin)
    return q_, k_


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("cos_cached", torch.stack((cos, cos), dim=-1).reshape(max_seq_len, head_dim), persistent=False)
        self.register_buffer("sin_cached", torch.stack((sin, sin), dim=-1).reshape(max_seq_len, head_dim), persistent=False)

    def get_rope(self, x: torch.Tensor, tlen: int):
        cos = self.cos_cached[:tlen].to(x.device).unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
        sin = self.sin_cached[:tlen].to(x.device).unsqueeze(0).unsqueeze(0)
        return cos, sin


# ---------------------------------------------------------
# Attention Variants
# ---------------------------------------------------------

class MultiQueryAttention(nn.Module):
    """
    H-Module: MQA (single K/V head; many Q heads)
    Expects mask_h: [B,1,T,T] (causal * pad)
    """
    def __init__(self, d_model: int, n_heads: int, rope: Rotary, attn_dropout: float = 0.0, resid_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)
        self.rope = rope

        _init_lecun_trunc_normal_(self.q_proj.weight)
        _init_lecun_trunc_normal_(self.k_proj.weight)
        _init_lecun_trunc_normal_(self.v_proj.weight)
        _init_lecun_trunc_normal_(self.o_proj.weight)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,T,Dh]
        k = self.k_proj(x).unsqueeze(1)  # [B,1,T,Dh]
        v = self.v_proj(x).unsqueeze(1)  # [B,1,T,Dh]

        cos, sin = self.rope.get_rope(q, T)
        q, k = apply_rope(q, k, cos, sin)

        att = torch.einsum("bhtd,busd->bhts", q, k) * (1.0 / math.sqrt(self.head_dim))  # [B,H,T,T]
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        p = F.softmax(att, dim=-1)
        p = self.attn_drop(p)
        out = torch.einsum("bhts,busd->bhtd", p, v)  # [B,H,T,Dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        out = self.resid_drop(self.o_proj(out))
        return out


class MLAAttention(nn.Module):
    """
    L-Module: Multi-Head Latent Attention (token->M latent aggregation -> token)
    Expects mask_l: [B,1,T,1] (pad only for Stage1)
    """
    def __init__(self, d_model: int, n_heads: int, n_latents: int, rope: Rotary, attn_dropout: float = 0.0, resid_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_latents = n_latents
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.latent_code = nn.Parameter(torch.randn(n_heads, n_latents, self.head_dim))  # [H, M, Dh]

        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)
        self.rope = rope

        _init_lecun_trunc_normal_(self.q_proj.weight)
        _init_lecun_trunc_normal_(self.k_proj.weight)
        _init_lecun_trunc_normal_(self.v_proj.weight)
        _init_lecun_trunc_normal_(self.o_proj.weight)
        with torch.no_grad():
            self.latent_code.mul_(0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        H, M, Dh = self.n_heads, self.n_latents, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)  # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)  # [B,H,T,Dh]
        v = self.v_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)  # [B,H,T,Dh]

        cos, sin = self.rope.get_rope(q, T)
        q, k = apply_rope(q, k, cos, sin)

        # -------- Stage 1: Token -> Latent aggregation per head ----------
        lc = self.latent_code.unsqueeze(0)  # [1,H,M,Dh]
        assign_logits = torch.einsum("bhtd,bhmd->bhtm", k, lc) * (1.0 / math.sqrt(Dh))  # [B,H,T,M]
        if mask is not None:
            # mask: [B,1,T,1] -> broadcast to [B,H,T,M]
            assign_logits = assign_logits.masked_fill(mask == 0, float("-inf"))
        assign = F.softmax(assign_logits, dim=2)  # normalize over T
        k_lat = torch.einsum("bhtm,bhtd->bhmd", assign, k)
        v_lat = torch.einsum("bhtm,bhtd->bhmd", assign, v)

        # -------- Stage 2: Token attends to Latents ----------
        att2 = torch.einsum("bhtd,bhmd->bhtm", q, k_lat) * (1.0 / math.sqrt(Dh))
        p2 = F.softmax(att2, dim=-1)
        p2 = self.attn_drop(p2)
        out = torch.einsum("bhtm,bhmd->bhtd", p2, v_lat)  # [B,H,T,Dh]

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        out = self.resid_drop(self.o_proj(out))
        return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden_mult * d_model)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        _init_lecun_trunc_normal_(self.w1.weight)
        _init_lecun_trunc_normal_(self.w2.weight)
        _init_lecun_trunc_normal_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, attn: nn.Module, ffn_mult: float, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = attn
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ffn_mult, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------
# HRM Stacks (H and L)
# ---------------------------------------------------------

@dataclass
class HRMConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_heads_h: int = 12
    n_heads_l: int = 12
    n_layers_h: int = 6
    n_layers_l: int = 6
    n_latents_l: int = 32            # MLA latent slots per head
    ffn_mult: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    max_seq_len: int = 2048
    # compute schedule
    N_cycles: int = 2                # H cycles
    T_steps: int = 4                 # L steps per H
    # deep supervision segments (fixed for pretrain)
    segments: int = 2
    # misc
    tie_lm_head: bool = True
    pad_id: int = 0


class HRMStackH(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.rope = Rotary(cfg.d_model // cfg.n_heads_h, cfg.max_seq_len)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model,
                  MultiQueryAttention(cfg.d_model, cfg.n_heads_h, rope=self.rope, attn_dropout=cfg.attn_dropout, resid_dropout=cfg.dropout),
                  cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers_h)
        ])
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for b in self.blocks:
            x = b(x, mask)
        return self.norm(x)


class HRMStackL(nn.Module):
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.rope = Rotary(cfg.d_model // cfg.n_heads_l, cfg.max_seq_len)
        self.blocks = nn.ModuleList([
            Block(cfg.d_model,
                  MLAAttention(cfg.d_model, cfg.n_heads_l, cfg.n_latents_l, rope=self.rope,
                               attn_dropout=cfg.attn_dropout, resid_dropout=cfg.dropout),
                  cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers_l)
        ])
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for b in self.blocks:
            x = b(x, mask)
        return self.norm(x)


# ---------------------------------------------------------
# HRM Model (next-token LM)
# ---------------------------------------------------------

class HRMModel(nn.Module):
    """
    Sequence LM variant of HRM.
    - Two recurrent stacks:
      * L-stack runs T_steps times per H cycle
      * H-stack runs once per cycle
    - 1-step gradient approximation: (N*T_steps−1) を no_grad、最後の 1 ステップのみ勾配追跡
    - Deep Supervision: 'segments' 回繰り返し、区切りで detach
    """
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.h_stack = HRMStackH(cfg)
        self.l_stack = HRMStackL(cfg)
        self.to_h_from_l = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.to_l_from_h = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_lm_head:
            self.lm_head.weight = self.embed.weight

        # Q-head for ACT（事前学習では凍結）
        self.q_head = nn.Linear(cfg.d_model, 2, bias=True)

        _init_lecun_trunc_normal_(self.embed.weight)
        _init_lecun_trunc_normal_(self.to_h_from_l.weight)
        _init_lecun_trunc_normal_(self.to_l_from_h.weight)
        if not cfg.tie_lm_head:
            _init_lecun_trunc_normal_(self.lm_head.weight)
        _init_lecun_trunc_normal_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    # --------- State init ----------
    def init_state(self, B: int, T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.cfg.d_model
        zH = torch.randn(B, T, d, device=device).clamp_(-2.0, 2.0)
        zL = torch.randn(B, T, d, device=device).clamp_(-2.0, 2.0)
        return zH, zL

    # --------- Single HRM segment with 1-step gradient ----------
    def hrm_segment(self, x_emb: torch.Tensor,
                    mask_h: torch.Tensor, mask_l: torch.Tensor,
                    zH: torch.Tensor, zL: torch.Tensor,
                    N: int, T_steps: int, one_step_grad: bool = True) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            x_emb: [B,T,C] token embeddings
            mask_h: [B,1,T,T] (causal * pad) for H/MQA
            mask_l: [B,1,T,1] (pad) for L/MLA
            zH, zL: previous hidden states [B,T,C]
        Returns:
            (zH_new, zL_new), logits, q_values
        """
        def l_update(zL_in, zH_cur):
            x_l = x_emb + self.to_l_from_h(zH_cur)
            return self.l_stack(x_l + zL_in, mask=mask_l)

        def h_update(zH_in, zL_cur):
            x_h = x_emb + self.to_h_from_l(zL_cur)
            return self.h_stack(x_h + zH_in, mask=mask_h)

        if one_step_grad:
            with torch.no_grad():
                zH_tmp, zL_tmp = zH, zL
                total = N * T_steps
                for i in range(total - 1):
                    zL_tmp = l_update(zL_tmp, zH_tmp)
                    if (i + 1) % T_steps == 0:
                        zH_tmp = h_update(zH_tmp, zL_tmp)
                zH_nograd, zL_nograd = zH_tmp.detach(), zL_tmp.detach()

            zL_last = l_update(zL_nograd, zH_nograd)
            zH_last = h_update(zH_nograd, zL_last)
        else:
            zH_last, zL_last = zH, zL
            total = N * T_steps
            for i in range(total):
                zL_last = l_update(zL_last, zH_last)
                if (i + 1) % T_steps == 0:
                    zH_last = h_update(zH_last, zL_last)

        logits = self.lm_head(zH_last)           # [B,T,V]
        q_values = self.q_head(zH_last.mean(dim=1))  # [B,2]
        return (zH_last, zL_last), logits, q_values

    # --------- Forward ----------
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                segments: Optional[int] = None,
                one_step_grad: bool = True,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        cfg = self.cfg
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=input_ids.device)

        # マスク作成：H 用（因果×Pad）、L 用（Pad）
        causal = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        mask_h = causal * pad_mask                           # [B,1,T,T]
        mask_l = attention_mask.unsqueeze(1).unsqueeze(-1)   # [B,1,T,1]

        x_emb = self.embed(input_ids)

        if state is None:
            zH, zL = self.init_state(B, T, x_emb.device)
        else:
            zH, zL = state

        S = segments if segments is not None else cfg.segments
        logits = None
        q_values = None

        for s in range(S):
            (zH, zL), logits, q_values = self.hrm_segment(
                x_emb, mask_h, mask_l, zH, zL, cfg.N_cycles, cfg.T_steps, one_step_grad=one_step_grad
            )
            if s != S - 1:
                zH = zH.detach()
                zL = zL.detach()

        out = {"logits": logits, "q_values": q_values, "state": (zH, zL)}

        # 言語モデリング損失（右シフト）
        if labels is not None:
            pred = logits[:, :-1, :].contiguous()
            tgt = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                pred.view(-1, pred.size(-1)),
                tgt.view(-1),
                ignore_index=cfg.pad_id,
            )
            out["loss"] = loss

        return out

    # --------- Utility ----------
    def freeze_halting(self):
        for p in self.q_head.parameters():
            p.requires_grad = False
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.zero_()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
        self.eval()
        state = None
        out = input_ids
        for _ in range(max_new_tokens):
            x = out[:, -self.cfg.max_seq_len:]
            res = self.forward(x, state=state, one_step_grad=False)
            logits = res["logits"][:, -1, :]
            state = res["state"]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
        return out

