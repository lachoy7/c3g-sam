import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_mask = mask.bool()
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class InstillAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        cfg=None,
        freeze_qk: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.freeze_qk = freeze_qk
        self.inner_dim = inner_dim

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        feature_dim = cfg.feature_dim if cfg.different_learnable_tokens else dim
        self.to_anotherv = nn.Linear(feature_dim, inner_dim, bias=False)

        self.to_yout = (
            nn.Sequential(nn.Linear(inner_dim, feature_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        if freeze_qk:
            for param in (*self.to_q.parameters(), *self.to_k.parameters()):
                param.requires_grad = False

        self._last_qkv: torch.Tensor | None = None

    def _project_qkv(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.freeze_qk:
            with torch.no_grad():
                q = self.to_q(x_norm)
                k = self.to_k(x_norm)
            v = self.to_v(x_norm)
        else:
            q = self.to_q(x_norm)
            k = self.to_k(x_norm)
            v = self.to_v(x_norm)
        self._last_qkv = torch.cat([q, k, v], dim=-1)
        return q, k, v

    def forward(self, x, y, mask=None):
        x_norm = self.norm(x)
        q, k, v = self._project_qkv(x_norm)
        another_v = self.to_anotherv(y)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        another_v = rearrange(another_v, "b n (h d) -> b h n d", h=self.heads)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_mask = mask.bool()
        else:
            attn_mask = None

        # Force the numerically-stable math SDPA backend: with frozen q/k
        # (requires_grad=False) and only the value path trainable, the
        # memory-efficient backend's backward can return NaN value gradients.
        with sdpa_kernel(SDPBackend.MATH):
            x_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        x_out = rearrange(x_out, "b h n d -> b n (h d)")
        x_out = self.to_out(x_out)

        with sdpa_kernel(SDPBackend.MATH):
            y_out = F.scaled_dot_product_attention(
                q,
                k,
                another_v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        y_out = rearrange(y_out, "b h n d -> b n (h d)")
        y_out = self.to_yout(y_out)

        return x_out, y_out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, cfg=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.cfg = cfg
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, mask=None, context_feature=None):
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x) + x
        return self.norm(x)


class InstillTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        cfg=None,
        freeze_instill_qk: bool = False,
    ):
        super().__init__()
        feature_mlp_dim = (
            cfg.feature_dim * 2 if cfg.different_learnable_tokens else mlp_dim
        )
        feature_dim = cfg.feature_dim if cfg.different_learnable_tokens else dim
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.cfg = cfg
        self.y_norm = nn.LayerNorm(feature_dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        InstillAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            cfg=cfg,
                            freeze_qk=freeze_instill_qk,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                        FeedForward(feature_dim, feature_mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, mask=None, context_feature=None):
        for attn, ff1, ff2 in self.layers:
            x_attn, y_attn = attn(x, context_feature, mask)
            x = x_attn + x
            context_feature = y_attn + context_feature
            x = ff1(x) + x
            context_feature = ff2(context_feature) + context_feature
        return self.norm(x), self.y_norm(context_feature)


def remap_instill_to_qkv_checkpoint(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map legacy fused ``to_qkv.weight`` keys to ``to_q`` / ``to_k`` / ``to_v``."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(".to_qkv.weight"):
            inner = value.shape[0] // 3
            prefix = key[: -len(".to_qkv.weight")]
            out[f"{prefix}.to_q.weight"] = value[:inner].clone()
            out[f"{prefix}.to_k.weight"] = value[inner : 2 * inner].clone()
            out[f"{prefix}.to_v.weight"] = value[2 * inner :].clone()
        elif ".to_qkv." in key:
            continue
        else:
            out[key] = value
    return out


def freeze_instill_attention_qk(instill_transformer: InstillTransformer) -> None:
    """No-op when ``freeze_instill_qk`` was set at construction (preferred path)."""
    for module_list in instill_transformer.layers:
        attn = module_list[0]
        if not isinstance(attn, InstillAttention):
            continue
        attn.freeze_qk = True
        for param in (*attn.to_q.parameters(), *attn.to_k.parameters()):
            param.requires_grad = False
