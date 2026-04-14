"""
DiffGPT — autoregressive transformer for git diff explanation.

The model is trained to generate natural-language commit messages from raw
git diff text.  Each training sequence has the form:

    <|diff|>
    {raw unified diff}
    <|endofdiff|>
    <|msg|>
    {commit message}
    <|endofmsg|>

Loss is computed only on the commit-message tokens; the diff prompt is masked
out via IGNORE_INDEX so the model learns to condition on the diff and
produce a concise explanation rather than memorising patch syntax.

Two usage modes are supported:

* **From scratch** — ``DiffGPT``: a compact GPT-2-style decoder trained
  entirely on CommitBench data.
* **Fine-tuned** — ``load_pretrained_gpt2``: loads an HuggingFace GPT-2
  checkpoint and fine-tunes it on the same task, benefiting from the
  pretrained language model's world knowledge.

Context window: controlled by ``ModelConfig.context_length`` (default 1024),
which must match the ``allowed_max_length`` used in ``data.collate_diff_batch``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, context_length, drop_rate, qkv_bias):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.Wq = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.attn_drop = nn.Dropout(drop_rate)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        )

    def forward(self, x):
        B, T, D = x.shape
        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T], -torch.inf)
        weights = self.attn_drop(torch.softmax(scores, dim=-1))

        context = (weights @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, drop_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.emb_dim)
        self.ln2 = LayerNorm(cfg.emb_dim)
        self.attn = MultiHeadCausalSelfAttention(
            cfg.emb_dim, cfg.n_heads, cfg.context_length, cfg.drop_rate, cfg.qkv_bias
        )
        self.ff = FeedForward(cfg.emb_dim, cfg.drop_rate)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DiffGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop_emb(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        return self.out_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_length:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k, dim=-1)
                logits = torch.where(logits < v[:, -1:], torch.full_like(logits, -1e9), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if next_id[0].item() == 50260:  # <|endofmsg|>
                break

        return idx


def load_pretrained_gpt2(cfg: ModelConfig, model_name="gpt2"):
    from transformers import GPT2LMHeadModel
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    return hf_model
