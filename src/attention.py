import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.models as tvm
import math
xformers = None
has_xformers = False
# modified from https://github.com/jadore801120/attention-is-all-you-need-pytorch
class CrossAttention(nn.Module):
    def __init__(self, 
                 embedding_dim=64,
                 num_heads=4,
                 dropout=0.1,
                 use_softmax=True):
        super(CrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_softmax = use_softmax
        
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, target_item, history_sequence, mask=None):
        batch_size, seq_len, _ = history_sequence.size()
        
        # Prepare Q/K/V
        query = self.query_proj(target_item)
        if len(target_item.shape) == 2:
            query = query.unsqueeze(0)
        elif len(target_item.shape) == 3:
            query = query.transpose(0, 1)
        key = self.key_proj(history_sequence.transpose(0, 1))
        value = self.value_proj(history_sequence.transpose(0, 1)) 
        
        # Create attention mask
        if mask is not None:
            attn_mask = (~mask.bool())
        else:
            attn_mask = None
        
        # Cross-attention
        attn_output, _ = self.multihead_attn(query=query,
                                             key=key,
                                             value=value,
                                             key_padding_mask=attn_mask)
        
        # Process output

        if attn_output.shape[0] ==1:
             attn_output = attn_output.squeeze(0)
        else:
            attn_output = attn_output.transpose(0, 1)
        attn_output = self.norm(attn_output)
        return attn_output

    
class CrossTransformerLayer(nn.Module):
    def __init__(self, 
                 embedding_dim=256,
                 num_heads=4,
                 ff_dim=256,
                 dropout=0.1,
                 use_softmax=True):
        super(CrossTransformerLayer, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Cross-Attention Block
        self.cross_attn = CrossAttention(embedding_dim, num_heads, dropout, use_softmax)
        self.attn_norm = nn.LayerNorm(embedding_dim)
        
        # Feed-Forward Block
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )
        #self.ffn_norm = nn.RMSNorm(embedding_dim)
        self.ffn_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_item, history_sequence, mask=None):
        # Cross-Attention with residual connection
        attn_out = self.cross_attn(target_item, history_sequence, mask)
        attn_out = self.attn_norm(target_item + self.dropout(attn_out))
        
        # Feed-Forward with residual connection
        ffn_out = self.ffn(attn_out)
        output = self.ffn_norm(attn_out + self.dropout(ffn_out))
        
        return output

class CrossTransformerEncoder(nn.Module):
    def __init__(self,
                 num_layers=2,
                 embedding_dim=256,
                 num_heads=4,
                 ff_dim=256,
                 dropout=0.1,
                 use_softmax=True):
        super(CrossTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            CrossTransformerLayer(embedding_dim, num_heads, ff_dim, dropout, use_softmax)
            for _ in range(num_layers)
        ])
        self.USE_self_attn = False

    def forward(self, target_item, history_sequence, mask=None):
        Query = target_item
        for i,layer in enumerate(self.layers):
            if(i==0 and self.USE_self_attn):
                output = layer(history_sequence, history_sequence, mask)
            elif(i==0):
                output = layer(Query, history_sequence,mask)
            else:
                output = layer(output, history_sequence, mask)
        return output



class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(
        self,
        n_head,
        d_model,
        d_k,
        d_v,
        dropout=0.0,
        activation="softmax",
        prenorm=False,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=True)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=True)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5, dropout=dropout, activation=activation
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.prenorm = prenorm

    def forward(self, x, mask=None):
        if x.ndim==3:
            x=x
        else:
            x = x.unsqueeze(0)

        residual = x
        if self.prenorm:
            x = self.layer_norm(x)

        q = x
        k = x
        v = x

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # if mask is not None:
        #    mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q = self.dropout(q)
        q = q + residual

        if not self.prenorm:  # post norm
            q = self.layer_norm(q)

        q = q.squeeze(0)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, dropout=0.0, activation="softmax"):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if activation == "softmax":
            self.activation = lambda x: F.softmax(x, dim=-1)
        elif activation == "entmax":
            self.activation = lambda x: entmax15(x, dim=-1)
        else:
            raise ValueError("Unknown activation!")

    def forward(self, q, k, v, neg_inf_mask=None):
        # b x n x lq x dv
        attn = torch.matmul(
            q / self.temperature, k.transpose(2, 3)
        )  # (b x n x lq x dv) (b x n x dv x lq)
        if neg_inf_mask is not None:
            attn = attn + neg_inf_mask
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.0, activation="relu", prenorm=False):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self.prenorm = prenorm

    def forward(self, x):
        residual = x
        if self.prenorm:
            x = self.layer_norm(x)
        x = self.w_2(self.activation(self.w_1(x)))
        x = self.dropout(x)
        x = x + residual
        if not self.prenorm:  # post norm
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        activation_attn="softmax",
        activation_ff="relu",
        prenorm=False,
    ):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout, activation_attn, prenorm
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout, activation_ff, prenorm
        )

    def forward(self, enc_input, mask=None):
        enc_output, attn = self.slf_attn(enc_input, mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, d_in):
        super().__init__()
        num_w = d_model // 2 // d_in
        self.num_pad = d_model - num_w * 2 * d_in
        w = 1e-3 ** torch.linspace(0, 1, num_w)
        self.w = nn.Parameter(w.view(1, -1, 1).repeat(1, 1, d_in))

    def forward(self, x):
        x=x.squeeze(0)
        x = x.unsqueeze(1)
        x = torch.cat((torch.sin(self.w * x), torch.cos(self.w * x)), 1)
        x = x.flatten(1)
        if self.num_pad:
            x = F.pad(x, (0, self.num_pad))
        return x


class TransformerEncoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        dropout,
        activation_attn="softmax",
        activation_ff="relu",
        prenorm=False,
    ):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_head,
                    d_k,
                    d_v,
                    dropout,
                    activation_attn,
                    activation_ff,
                    prenorm,
                )
                for _ in range(n_layers)
            ]
        )
        self.prenorm = prenorm
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, pos_enc, mode='add',mask=None):
        if mode=='add':
            enc_output = self.dropout(x + pos_enc)
        elif mode=='cat':
            enc_output=self.dropout(torch.cat([x,pos_enc],dim=-1))
        else:
            enc_output=self.dropout(x+ pos_enc)
        if not self.prenorm:  # post-norm
            enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, attn = enc_layer(enc_output, mask)

        if self.prenorm:
            enc_output = self.layer_norm(enc_output)  # pre-norm

        return enc_output, attn


class ResNet(nn.Module):
    def __init__(self, n_res, d_model, d_in=1, pretrained=False):
        super().__init__()
        resnet_fn = getattr(tvm, "resnet%d" % n_res)
        model = resnet_fn(
            pretrained=pretrained,
            norm_layer=lambda x: nn.BatchNorm2d(x, track_running_stats=False),
        )
        model.fc = nn.Linear(model.fc.in_features, d_model)
        if not pretrained:
            model.conv1 = nn.Conv2d(
                d_in, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.model = model
        self.pretrained = pretrained

    def forward(self, x):
        if self.pretrained:
            x = x.expand(-1, 3, -1, -1)
        return self.model(x)

class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Args:
        query_dim: number of channels in the query.
        cross_attention_dim: number of channels in the context.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each head.
        dropout: dropout probability to use.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim = None,
        num_attention_heads: int = 8,
        num_head_channels: int = 64,
        dropout: float = 0.0,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        inner_dim = num_head_channels * num_attention_heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = 1 / math.sqrt(num_head_channels)
        self.num_heads = num_attention_heads

        self.upcast_attention = upcast_attention

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype=dtype)

        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor  = None) -> torch.Tensor:
        query = self.to_q(x)
        context = context if context is not None else x
        key = self.to_k(context)
        value = self.to_v(context)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        return self.to_out(x)


class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block.

    Args:
        num_channels: number of channels in the input and output.
        num_attention_heads: number of heads to use for multi-head attention.
        num_head_channels: number of channels in each attention head.
        dropout: dropout probability to use.
        cross_attention_dim: size of the context vector for cross attention.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        num_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        dropout: float = 0.0,
        cross_attention_dim: int  = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=num_channels,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )  # is a self-attention
        self.ff = MLPBlock(hidden_size=num_channels, mlp_dim=num_channels * 4, act="GELU", dropout_rate=dropout)
        self.attn2 = CrossAttention(
            query_dim=num_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            num_head_channels=num_head_channels,
            dropout=dropout,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )  # is a self-attention if context is None
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)
        self.norm3 = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor  = None) -> torch.Tensor:
        # 1. Self-Attention
        x = self.attn1(self.norm1(x)) + x

        # 2. Cross-Attention
        x = self.attn2(self.norm2(x), context=context) + x

        # 3. Feed-forward
        x = self.ff(self.norm3(x)) + x
        return x
"""
An implementation of entmax (Peters et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.

This builds on previous work with sparsemax (Martins & Astudillo, 2016).
See https://arxiv.org/pdf/1602.02068.
"""

# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>
# License: MIT


def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for sparsemax: optimal threshold and support size.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.

    dim : int
        The dimension along which to apply sparsemax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


def _entmax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for 1.5-entmax: optimal threshold and support size.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.

    dim : int
        The dimension along which to apply 1.5-entmax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


class Entmax15Function(Function):
    @classmethod
    def forward(cls, ctx, X, dim=0, k=None):
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


def sparsemax(X, dim=-1, k=None):
    """sparsemax: normalizing sparse transform (a la softmax).

    Solves the projection:

        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    dim : int
        The dimension along which to apply sparsemax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """

    return SparsemaxFunction.apply(X, dim, k)


def entmax15(X, dim=-1, k=None):
    """1.5-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    dim : int
        The dimension along which to apply 1.5-entmax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """

    return Entmax15Function.apply(X, dim, k)


class Sparsemax(nn.Module):
    def __init__(self, dim=-1, k=None):
        """sparsemax: normalizing sparse transform (a la softmax).

        Solves the projection:

            min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

        Parameters
        ----------
        dim : int
            The dimension along which to apply sparsemax.

        k : int or None
            number of largest elements to partial-sort over. For optimal
            performance, should be slightly bigger than the expected number of
            nonzeros in the solution. If the solution is more than k-sparse,
            this function is recursively called with a 2*k schedule.
            If `None`, full sorting is performed from the beginning.
        """
        self.dim = dim
        self.k = k
        super(Sparsemax, self).__init__()

    def forward(self, X):
        return sparsemax(X, dim=self.dim, k=self.k)


class Entmax15(nn.Module):
    def __init__(self, dim=-1, k=None):
        """1.5-entmax: normalizing sparse transform (a la softmax).

        Solves the optimization problem:

            max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

        where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

        Parameters
        ----------
        dim : int
            The dimension along which to apply 1.5-entmax.

        k : int or None
            number of largest elements to partial-sort over. For optimal
            performance, should be slightly bigger than the expected number of
            nonzeros in the solution. If the solution is more than k-sparse,
            this function is recursively called with a 2*k schedule.
            If `None`, full sorting is performed from the beginning.
        """
        self.dim = dim
        self.k = k
        super(Entmax15, self).__init__()

    def forward(self, X):
        return entmax15(X, dim=self.dim, k=self.k)
