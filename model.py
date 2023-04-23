import math
import torch
from torch import nn
import torch.nn.functional as F


class LatentEmbeddings(nn.Module):
    def __init__(self, latent_seq_len, latent_channels):
        ''' stole this bad boy from HF transformers perceiver impl. '''
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_seq_len, latent_channels))

    def forward(self, batch_size):
        # we want to have the same latents across batch dimension
        return self.latents.expand(batch_size, -1, -1)

class CrossAttention(nn.Module):
    def __init__(
        self,
        latent_channels,
        in_channels,
        qk_channels=None,
        v_channels=None,
        out_channels=None,
        nxheads=1,
        dropout=0.0
    ):
        super().__init__()

        # we want to default to latent channels for q/k/v
        if qk_channels is None:
            qk_channels = latent_channels
        if v_channels is None:
            v_channels = qk_channels
        if out_channels is None:
            # not sure why deepmind code defaults to v_channels since we want
            # the final channel number to match latent channels regardless
            out_channels = latent_channels

        assert qk_channels % nxheads == 0
        assert v_channels % nxheads == 0

        self.W_Q = nn.Linear(latent_channels, qk_channels, bias=False)
        self.W_K = nn.Linear(in_channels, qk_channels, bias=False)
        
        self.W_V = nn.Linear(in_channels, v_channels, bias=False)
        self.W_O = nn.Linear(v_channels, out_channels, bias=False)

        self.v_channels = v_channels

        self.qk_head_dim = qk_channels // nxheads 
        self.v_head_dim = v_channels // nxheads
        
        self.nxheads = nxheads
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent_q, input_kv):
        batch_size, input_seq_len, in_channels = input_kv.size()
        _, latent_seq_len, latent_channels = latent_q.size()

        # (batch_size, (latent/input)_seq_len, latent_channels) -> (batch_size, nheads, (latent/input)_seq_len, (qk/v)_head_dim)
        Q = self.W_Q(latent_q).reshape(batch_size, latent_seq_len, self.nxheads, self.qk_head_dim).transpose(1, 2)
        K = self.W_K(input_kv).reshape(batch_size, input_seq_len, self.nxheads, self.qk_head_dim).transpose(1, 2)
        V = self.W_V(input_kv).reshape(batch_size, input_seq_len, self.nxheads, self.v_head_dim).transpose(1, 2)

        # (batch_size, nheads, latent_seq_len, input_seq_len)
        attn = (Q @ K.transpose(-2, -1)) / (1.0 * math.sqrt(self.qk_head_dim))
        attn = F.softmax(attn, dim=-1)
        
        attn = self.dropout(attn)

        # (batch_size, nheads, latent_seq_len, v_head_dim)
        out = attn @ V
        # (batch_size, latent_seq_len, v_channels)
        out = out.transpose(1, 2).reshape(batch_size, latent_seq_len, self.v_channels)

        # (batch_size, latent_seq_len, out_channels/latent_channels)
        out = self.W_O(out) # project v_channels to out_channels/latent_channels

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels, nheads, dropout=0.0):
        super().__init__()
        assert in_channels % nheads == 0

        self.W_Q = nn.Linear(in_channels, in_channels, bias=False)
        self.W_K = nn.Linear(in_channels, in_channels, bias=False)
        
        self.W_V = nn.Linear(in_channels, in_channels, bias=False)
        self.W_O = nn.Linear(in_channels, in_channels, bias=False)

        self.head_dim = in_channels // nheads
        self.nheads = nheads
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, in_channels = x.size()
        
        Q = self.W_Q(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).reshape(batch_size, seq_len, self.nheads, self.head_dim).transpose(1, 2)

        # (batch_size, nheads, seq_len, seq_len)
        attn = (Q @ K.transpose(-2, -1)) / (1.0 * math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        
        attn = self.dropout(attn)

        # (batch_size, nheads, seq_len, head_dim)
        out = attn @ V
        # (batch_size, seq_len, in_channels)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, in_channels)

        # (batch_size, seq_len, in_channels)
        out = self.W_O(out)

        return out


class MLP(nn.Module):
    def __init__(self, in_channels, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 4 * in_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, nheads, dropout=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_channels)
        self.attn = SelfAttention(in_channels, nheads, dropout)
        self.ln_2 = nn.LayerNorm(in_channels)
        self.mlp = MLP(in_channels, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, latent_channels, in_channels, nxheads, dropout=0.0):
        ''' each cross-attention is followed by an MLP in the Perciever '''
        super().__init__()
        self.xattn = CrossAttention(
            latent_channels,
            in_channels,
            nxheads=nxheads,
            dropout=dropout
        )
        self.mlp = MLP(latent_channels, dropout=dropout)
        
        self.ln_latent = nn.LayerNorm(latent_channels)
        self.ln_input = nn.LayerNorm(in_channels)
        self.ln_mlp = nn.LayerNorm(latent_channels)

    def forward(self, latents, x):
        latents = latents + self.xattn(self.ln_latent(latents), self.ln_input(x))
        latents = latents + self.mlp(self.ln_mlp(latents))

        return latents


class PerceiverBlock(nn.Module):
    def __init__(
        self,
        latent_channels,
        in_channels,
        nheads,
        nxheads,
        nlayers,
        dropout=0.0
    ):
        ''' PerceiverBlock is one CrossAttentionBlock followed by nlayer standard TransformerBlocks '''
        super().__init__()
        self.xattn_block = CrossAttentionBlock(latent_channels, in_channels, nxheads=nxheads, dropout=dropout)
        self.attn_blocks = nn.ModuleList([
            TransformerBlock(latent_channels, nheads=nheads, dropout=dropout)
            for _ in range(nlayers)
        ])

    def forward(self, latents, x):
        latents = self.xattn_block(latents, x)
        for block in self.attn_blocks:
            latents = block(latents)

        return latents


class PerceiverBase(nn.Module):
    def __init__(
        self,
        latent_channels,
        latent_seq_len,
        in_channels,
        input_seq_len,
        nheads,
        nxheads,
        nlayers,
        nblocks,
        pos_emb_channels,
        dropout=0.0
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(input_seq_len, pos_emb_channels)
        self.latents = LatentEmbeddings(latent_seq_len, latent_channels)
        self.perceiver_blocks = nn.ModuleList([
            PerceiverBlock(
                latent_channels,
                in_channels + pos_emb_channels,
                nheads,
                nxheads,
                nlayers,
                dropout
            )
            for _ in range(nblocks)
        ])

    def forward(self, input):
        batch_size, seq_len, in_channels = input.size()
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input.device)
        pos_emb = self.pos_emb(pos).expand(batch_size, -1, -1)
        
        input = torch.cat([input, pos_emb], dim=-1)

        latents = self.latents(batch_size)

        for block in self.perceiver_blocks:
            latents = block(latents, input)
            
        return latents


class PerceiverClassificationHead(nn.Module):
    def __init__(
        self,
        latent_channels,
        latent_seq_len,
        in_channels,
        input_seq_len,
        out_channels,
        nheads,
        nxheads,
        nlayers,
        nblocks,
        pos_emb_channels,
        dropout=0.0
    ):
        super().__init__()
        self.perceiver = PerceiverBase(
            latent_channels,
            latent_seq_len,
            in_channels,
            input_seq_len,
            nheads,
            nxheads,
            nlayers,
            nblocks,
            pos_emb_channels,
            dropout
        )
        
        self.head = nn.Linear(latent_channels, out_channels)

    def forward(self, x):
        x = self.perceiver(x)
        x = torch.mean(x, dim=-2)
        x = self.head(x) # logits for classification
        
        return x