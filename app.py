"""
CodeMix-T: Unified Streamlit GUI
Consolidates all 6 phases of the CodeMix-T pipeline into a single interactive app.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import re
import time
import unicodedata
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CodeMix-T",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & Language Keywords (from Phase 1 / Phase 2)
# ---------------------------------------------------------------------------
LANG_EN, LANG_HI, LANG_TA, LANG_UNK = 0, 1, 2, 3
LANG_NAMES = {0: "EN", 1: "HI", 2: "TA", 3: "UNK"}

HINDI_ROMANIZED_KEYWORDS = {
    "hai", "hain", "tha", "thi", "the", "main", "mein", "nahi", "nahin",
    "kya", "koi", "aur", "lekin", "par", "toh", "bhi", "abhi", "yahan",
    "wahan", "kab", "kyun", "kaisa", "kaisi", "bahut", "accha", "theek",
    "kal", "aaj", "subah", "raat", "gaya", "aya", "dekh", "baat", "karo",
    "hoga", "hogi", "karke", "leke", "jaake", "rehna", "matlab", "samajh",
    "pata", "paise", "kaam", "log", "dost", "yaar", "bhai",
}
TAMIL_ROMANIZED_KEYWORDS = {
    "naan", "nee", "avan", "aval", "avanga", "romba", "sollu", "solla",
    "paar", "paaru", "vaa", "po", "enna", "epdi", "eppova", "konjam",
    "koncham", "theriyum", "theriyala", "irukku", "irukken", "pannuven",
    "pannrom", "seri", "illa", "illai", "ama", "aama", "inge", "anga",
    "yaar", "yaaru", "enga", "venum", "vendam", "paavam", "sappa", "super",
}

DEMO_EXAMPLES = [
    ("kal main market gaya tha for vegetables", "Hinglish"),
    ("yaar bahut accha movie tha, we should go again", "Hinglish"),
    ("mujhe bahut neend aa rahi hai today", "Hinglish"),
    ("office mein aaj meeting cancel ho gayi", "Hinglish"),
    ("main thoda busy hoon right now call later", "Hinglish"),
    ("naan romba tired aa irukken today", "Tanglish"),
    ("konjam wait panna sollu, I am coming", "Tanglish"),
    ("avan super talented da, definitely win panuvan", "Tanglish"),
    ("enna pannre nee, let us go eat something", "Tanglish"),
    ("theriyuma, I got promoted today", "Tanglish"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Data Pipeline Functions
# ═══════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_pair(src: str, tgt: str, min_len: int = 3, max_len: int = 128, max_ratio: float = 3.0) -> bool:
    if not src or not tgt:
        return False
    src_words = src.split()
    tgt_words = tgt.split()
    if len(src_words) < min_len or len(tgt_words) < min_len:
        return False
    if len(src_words) > max_len or len(tgt_words) > max_len:
        return False
    ratio = max(len(src_words), len(tgt_words)) / max(min(len(src_words), len(tgt_words)), 1)
    if ratio > max_ratio:
        return False
    if src.strip().lower() == tgt.strip().lower():
        return False
    return True


def has_code_mixing(text: str) -> bool:
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    has_tamil = bool(re.search(r"[\u0B80-\u0BFF]", text))
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    if (has_devanagari or has_tamil) and has_latin:
        return True
    if has_latin:
        return True
    return False


def get_token_lang_id(token: str) -> int:
    token_lower = token.lower().strip()
    if any("\u0900" <= c <= "\u097F" for c in token):
        return LANG_HI
    if any("\u0B80" <= c <= "\u0BFF" for c in token):
        return LANG_TA
    if token_lower in HINDI_ROMANIZED_KEYWORDS:
        return LANG_HI
    if token_lower in TAMIL_ROMANIZED_KEYWORDS:
        return LANG_TA
    if any(c.isascii() and c.isalpha() for c in token):
        return LANG_EN
    return LANG_UNK


def tag_sentence(sentence: str) -> list:
    tokens = sentence.split()
    return [(tok, get_token_lang_id(tok)) for tok in tokens]


def apply_lang_tags(row: pd.Series) -> pd.Series:
    tagged = tag_sentence(row["source"])
    tokens = [t for t, _ in tagged]
    lang_ids = [lid for _, lid in tagged]
    return pd.Series({
        "source_tokens": " ".join(tokens),
        "source_lang_ids": " ".join(map(str, lang_ids)),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Model Architecture
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CodeMixTConfig:
    vocab_size: int = 16000
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    num_languages: int = 4
    lang_embed_dim: int = 64
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_enc_layers: int = 6
    num_dec_layers: int = 6
    max_seq_len: int = 128
    dropout: float = 0.1
    beam_size: int = 4
    max_gen_len: int = 128

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CodeMixEmbedding(nn.Module):
    def __init__(self, cfg: CodeMixTConfig):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.lang_embed = nn.Embedding(cfg.num_languages, cfg.lang_embed_dim)
        self.lang_proj = nn.Linear(cfg.lang_embed_dim, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = math.sqrt(cfg.d_model)

    def forward(self, token_ids: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        tok = self.token_embed(token_ids) * self.scale
        lang = self.lang_proj(self.lang_embed(lang_ids))
        x = tok + lang
        x = self.pos_encoding(x)
        return x


class CodeMixEmbeddingAblation(nn.Module):
    """Embedding variant that can disable Language-ID (for ablation experiments)."""

    def __init__(self, cfg: CodeMixTConfig, use_lang_id: bool = True):
        super().__init__()
        self.use_lang_id = use_lang_id
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = math.sqrt(cfg.d_model)
        if use_lang_id and cfg.lang_embed_dim > 0:
            self.lang_embed = nn.Embedding(cfg.num_languages, cfg.lang_embed_dim)
            self.lang_proj = nn.Linear(cfg.lang_embed_dim, cfg.d_model, bias=False)
        else:
            self.lang_embed = None
            self.lang_proj = None

    def forward(self, token_ids: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        tok = self.token_embed(token_ids) * self.scale
        if self.use_lang_id and self.lang_embed is not None:
            lang = self.lang_proj(self.lang_embed(lang_ids))
            tok = tok + lang
        return self.pos_encoding(tok)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: CodeMixTConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.d_k = cfg.d_k
        self.d_model = cfg.d_model
        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = math.sqrt(self.d_k)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, V)
        context = self.merge_heads(context)
        return self.W_o(context)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, cfg: CodeMixTConfig):
        super().__init__()
        self.linear1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.linear2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: CodeMixTConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg)
        self.ffn = PositionwiseFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, mask))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: CodeMixTConfig):
        super().__init__()
        self.embedding = CodeMixEmbedding(cfg)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_enc_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, token_ids, lang_ids, src_mask=None):
        x = self.embedding(token_ids, lang_ids)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: CodeMixTConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg)
        self.cross_attn = MultiHeadAttention(cfg)
        self.ffn = PositionwiseFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, tgt_mask))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask))
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ffn(x))
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: CodeMixTConfig):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_dec_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.scale = math.sqrt(cfg.d_model)

    def forward(self, tgt_ids, enc_output, tgt_mask=None, src_mask=None):
        x = self.token_embed(tgt_ids) * self.scale
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.norm(x)


def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    B, T = tgt.shape
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool().unsqueeze(0).unsqueeze(0)
    return pad_mask & causal_mask


class CodeMixT(nn.Module):
    def __init__(self, cfg: CodeMixTConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.output_projection = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.output_projection.weight = self.decoder.token_embed.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def encode(self, src_ids, lang_ids, src_mask=None):
        return self.encoder(src_ids, lang_ids, src_mask)

    def decode(self, tgt_ids, enc_output, tgt_mask=None, src_mask=None):
        dec_out = self.decoder(tgt_ids, enc_output, tgt_mask, src_mask)
        return self.output_projection(dec_out)

    def forward(self, src_ids, lang_ids, tgt_ids):
        src_mask = make_src_mask(src_ids, self.cfg.pad_id).to(src_ids.device)
        tgt_mask = make_tgt_mask(tgt_ids, self.cfg.pad_id).to(tgt_ids.device)
        enc_output = self.encode(src_ids, lang_ids, src_mask)
        return self.decode(tgt_ids, enc_output, tgt_mask, src_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BeamSearchDecoder:
    def __init__(self, model: CodeMixT, cfg: CodeMixTConfig):
        self.model = model
        self.cfg = cfg
        self.beam_size = cfg.beam_size

    @torch.no_grad()
    def decode(self, src_ids, lang_ids, max_len=None):
        if max_len is None:
            max_len = self.cfg.max_gen_len
        self.model.eval()
        device = src_ids.device
        src_mask = make_src_mask(src_ids, self.cfg.pad_id).to(device)
        enc_output = self.model.encode(src_ids, lang_ids, src_mask)
        enc_output = enc_output.repeat(self.beam_size, 1, 1)
        src_mask = src_mask.repeat(self.beam_size, 1, 1, 1)
        beams = [(0.0, [self.cfg.bos_id])]
        completed = []
        for _ in range(max_len):
            candidates = []
            beam_seqs = [seq for _, seq in beams]
            max_beam_len = max(len(s) for s in beam_seqs)
            padded = [s + [self.cfg.pad_id] * (max_beam_len - len(s)) for s in beam_seqs]
            tgt_tensor = torch.tensor(padded, device=device)
            tgt_mask = make_tgt_mask(tgt_tensor, self.cfg.pad_id).to(device)
            logits = self.model.decode(tgt_tensor, enc_output[: len(beams)], tgt_mask, src_mask[: len(beams)])
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            for i, (score, seq) in enumerate(beams):
                if seq[-1] == self.cfg.eos_id:
                    completed.append((score, seq))
                    continue
                topk_probs, topk_ids = log_probs[i].topk(self.beam_size)
                for prob, tok_id in zip(topk_probs.tolist(), topk_ids.tolist()):
                    candidates.append((score + prob, seq + [tok_id]))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[: self.beam_size]
            if all(seq[-1] == self.cfg.eos_id for _, seq in beams):
                completed.extend(beams)
                break
        all_seqs = completed if completed else beams
        all_seqs.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
        best_seq = all_seqs[0][1]
        if best_seq and best_seq[0] == self.cfg.bos_id:
            best_seq = best_seq[1:]
        if best_seq and best_seq[-1] == self.cfg.eos_id:
            best_seq = best_seq[:-1]
        return best_seq


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Training Utilities
# ═══════════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_val = self.smoothing / (self.vocab_size - 2)
        with torch.no_grad():
            dist = torch.full_like(log_probs, smooth_val)
            dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            dist[:, self.pad_id] = 0
            mask = targets == self.pad_id
            dist[mask] = 0
        loss = -(dist * log_probs).sum(dim=-1)
        non_pad = (~mask).sum()
        return loss.sum() / non_pad.clamp(min=1)


class WarmupInvSqrtScheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _get_lr(self):
        s = self.step_num
        w = self.warmup_steps
        return (self.d_model**-0.5) * min(s**-0.5, s * w**-1.5)

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, sd):
        self.step_num = sd["step_num"]


class CodeMixDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, max_src_len: int = 128, max_tgt_len: int = 128):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                src = rec["src_ids"]
                tgt = rec["tgt_ids"]
                lid = list(map(int, rec["source_lang_ids"].split()))
                if len(lid) < len(src):
                    lid = lid + [0] * (len(src) - len(lid))
                lid = lid[: len(src)]
                if 3 <= len(src) <= max_src_len and 3 <= len(tgt) <= max_tgt_len:
                    self.samples.append((src, tgt, lid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, lid = self.samples[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            torch.tensor(lid, dtype=torch.long),
        )


def collate_fn(batch, pad_id=0):
    srcs, tgts, lids = zip(*batch)
    max_src = max(s.size(0) for s in srcs)
    max_tgt = max(t.size(0) for t in tgts)
    src_pad = torch.full((len(srcs), max_src), pad_id, dtype=torch.long)
    tgt_pad = torch.full((len(tgts), max_tgt), pad_id, dtype=torch.long)
    lid_pad = torch.zeros(len(lids), max_src, dtype=torch.long)
    for i, (s, t, l_) in enumerate(zip(srcs, tgts, lids)):
        src_pad[i, : s.size(0)] = s
        tgt_pad[i, : t.size(0)] = t
        lid_pad[i, : l_.size(0)] = l_
    return src_pad, tgt_pad, lid_pad


def train_one_step(model, batch, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    src_ids, tgt_ids, lang_ids = [x.to(device) for x in batch]
    dec_input = tgt_ids[:, :-1]
    dec_target = tgt_ids[:, 1:]
    logits = model(src_ids, lang_ids, dec_input)
    B, T, V = logits.shape
    loss = criterion(logits.reshape(B * T, V), dec_target.reshape(B * T))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    lr = scheduler.step()
    return loss.item(), lr


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in val_loader:
        src_ids, tgt_ids, lang_ids = [x.to(device) for x in batch]
        dec_input = tgt_ids[:, :-1]
        dec_target = tgt_ids[:, 1:]
        logits = model(src_ids, lang_ids, dec_input)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), dec_target.reshape(B * T))
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Ablation Configurations
# ═══════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = {
    "full_base": {
        "d_model": 512, "d_ff": 2048, "num_heads": 8,
        "num_enc_layers": 6, "num_dec_layers": 6,
        "lang_embed_dim": 64, "use_lang_id": True,
        "train_languages": "both",
        "description": "Full CodeMix-T with LangID embeddings",
    },
    "no_lang_id": {
        "d_model": 512, "d_ff": 2048, "num_heads": 8,
        "num_enc_layers": 6, "num_dec_layers": 6,
        "lang_embed_dim": 0, "use_lang_id": False,
        "train_languages": "both",
        "description": "Ablation: no Language-ID embeddings",
    },
    "small_lang_id": {
        "d_model": 256, "d_ff": 1024, "num_heads": 4,
        "num_enc_layers": 4, "num_dec_layers": 4,
        "lang_embed_dim": 32, "use_lang_id": True,
        "train_languages": "both",
        "description": "Small model (256 dim) with LangID",
    },
    "hinglish_only": {
        "d_model": 512, "d_ff": 2048, "num_heads": 8,
        "num_enc_layers": 6, "num_dec_layers": 6,
        "lang_embed_dim": 64, "use_lang_id": True,
        "train_languages": "hinglish",
        "description": "Hinglish-only training",
    },
    "tanglish_only": {
        "d_model": 512, "d_ff": 2048, "num_heads": 8,
        "num_enc_layers": 6, "num_dec_layers": 6,
        "lang_embed_dim": 64, "use_lang_id": True,
        "train_languages": "tanglish",
        "description": "Tanglish-only training",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Chatbot wrapper (used in Phase 6 demo)
# ═══════════════════════════════════════════════════════════════════════════

class CodeMixTChatbot:
    def __init__(self, model: CodeMixT, tokenizer_path: str, cfg: CodeMixTConfig):
        import sentencepiece as spm

        self.model = model
        self.cfg = cfg
        self.beam_decoder = BeamSearchDecoder(model, cfg)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

    def _get_token_lang_id(self, token: str) -> int:
        return get_token_lang_id(token)

    def _prepare_input(self, text: str):
        pieces = self.sp.encode_as_pieces(text)
        token_ids = [self.cfg.bos_id] + self.sp.encode_as_ids(text) + [self.cfg.eos_id]
        lang_ids = [LANG_EN]
        for piece in pieces:
            clean = piece.replace("\u2581", "").strip()
            lang_ids.append(self._get_token_lang_id(clean))
        lang_ids.append(LANG_EN)
        token_ids = token_ids[: self.cfg.max_seq_len]
        lang_ids = lang_ids[: self.cfg.max_seq_len]
        src_tensor = torch.tensor([token_ids], device=DEVICE)
        lang_tensor = torch.tensor([lang_ids], device=DEVICE)
        return src_tensor, lang_tensor

    def translate(self, text: str) -> str:
        self.model.eval()
        src_tensor, lang_tensor = self._prepare_input(text)
        output_ids = self.beam_decoder.decode(src_tensor, lang_tensor)
        return self.sp.decode_ids(output_ids)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: model summary
# ═══════════════════════════════════════════════════════════════════════════

def get_model_summary(model: CodeMixT) -> pd.DataFrame:
    components = {
        "Encoder Embedding (Token+LangID+Pos)": model.encoder.embedding,
        "Encoder Layers": model.encoder.layers,
        "Decoder Embedding": model.decoder.token_embed,
        "Decoder Layers": model.decoder.layers,
        "Output Projection": model.output_projection,
    }
    rows = []
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append({"Component": name, "Parameters": f"{params:,}"})
    total = model.count_parameters()
    rows.append({"Component": "Total (trainable)", "Parameters": f"{total:,}"})
    rows.append({"Component": "Approx size (fp32)", "Parameters": f"{total * 4 / 1e6:.1f} MB"})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "cfg": None,
        "model": None,
        "chatbot": None,
        "training_log": [],
        "data_df": None,
        "train_df": None,
        "val_df": None,
        "test_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.title("CodeMix-T")
st.sidebar.caption("Code-Mixed Translator")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Phase 1 — Data Pipeline",
        "Phase 2 — Architecture",
        "Phase 3 — Training",
        "Phase 4 — Ablation Studies",
        "Phase 5 — Evaluation",
        "Phase 6 — Live Demo",
    ],
)

st.sidebar.divider()
st.sidebar.markdown(f"**Device:** `{DEVICE}`")
if torch.cuda.is_available():
    st.sidebar.markdown(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    st.sidebar.markdown(f"**VRAM:** `{vram:.1f} GB`")

st.sidebar.divider()

BASE_DIR = st.sidebar.text_input(
    "Project Data Directory",
    value=str(Path.cwd()),
    help="Root directory containing data/, tokenizer/, model/, checkpoints/ folders",
)


# ═══════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════

if page == "Home":
    st.title("CodeMix-T")
    st.subheader("Language-ID-Aware Transformer for Code-Mixed Translation")
    st.markdown("---")

    st.markdown("""
    **CodeMix-T** is a custom encoder-decoder Transformer trained from scratch
    for translating **Tanglish** (Tamil-English) and **Hinglish** (Hindi-English)
    code-mixed text into standard English.

    The key novelty is a **Language-ID-Aware Embedding** layer that augments
    standard token and positional embeddings with a learned language-identity
    vector per token, explicitly encoding whether each token is Tamil, Hindi,
    or English.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Languages", "3", help="English, Hindi, Tamil")
    with col2:
        st.metric("Architecture", "Transformer", help="Custom encoder-decoder")
    with col3:
        st.metric("Vocab Size", "16,000", help="Custom BPE tokenizer")

    st.markdown("---")
    st.subheader("Pipeline Phases")

    phases = [
        ("Phase 1", "Data Pipeline", "Download, clean, tag, split, and tokenize data"),
        ("Phase 2", "Architecture", "Define the CodeMix-T Transformer with Language-ID embeddings"),
        ("Phase 3", "Training", "Train with label smoothing, warmup schedule, and checkpointing"),
        ("Phase 4", "Ablation Studies", "Compare variants: no LangID, small model, single-language"),
        ("Phase 5", "Evaluation", "BLEU, chrF++, mBART baseline, error analysis"),
        ("Phase 6", "Live Demo", "Interactive translation chatbot"),
    ]

    for phase_id, title, desc in phases:
        with st.expander(f"**{phase_id}: {title}**"):
            st.write(desc)

    st.markdown("---")
    st.subheader("Architecture Diagram")
    st.code("""
    Input (code-mixed tokens)
            |
            v
    +----------------------------------+
    |  Token Embedding                 |
    |+ Positional Embedding            |
    |+ Language-ID Embedding  <-- NOVEL|
    +---------------+------------------+
                    |
           +--------v--------+
           |    Encoder      |  6 layers, 8 heads
           |    Stack        |  d_model=512
           +--------+--------+
                    | (encoder memory)
           +--------v--------+
           |    Decoder      |  6 layers
           |    Stack        |  cross-attention
           +--------+--------+
                    |
           +--------v--------+
           |    Linear       |  project to vocab
           |  + Softmax      |
           +-----------------+
                    |
            English translation
    """, language=None)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Phase 1 — Data Pipeline":
    st.title("Phase 1: Data Collection & Preprocessing")
    st.markdown("---")

    tab_upload, tab_clean, tab_langtag, tab_split, tab_tokenizer = st.tabs([
        "Upload Data", "Clean & Filter", "Language Tagging", "Train/Val/Test Split", "Tokenizer",
    ])

    # --- Upload Data ---
    with tab_upload:
        st.subheader("Upload Parallel Data")
        st.markdown("""
        Upload a CSV file with at least two columns:
        - **`source`**: code-mixed text (Hinglish / Tanglish)
        - **`target`**: English translation

        Optionally include a **`language`** column (`hinglish`, `tanglish`, `hindi`).
        """)

        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.session_state["data_df"] = df
            st.success(f"Loaded {len(df):,} rows")
            st.dataframe(df.head(20), use_container_width=True)

            if "source" in df.columns and "target" in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total rows", f"{len(df):,}")
                with col2:
                    if "language" in df.columns:
                        lang_counts = df["language"].value_counts()
                        st.write("**Language distribution:**")
                        st.dataframe(lang_counts, use_container_width=True)
            else:
                st.warning("CSV must contain `source` and `target` columns.")

        # Or load from existing JSONL
        st.markdown("---")
        st.subheader("Or Load Existing JSONL Files")
        data_dir = os.path.join(BASE_DIR, "data", "final")
        if os.path.isdir(data_dir):
            jsonl_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
            if jsonl_files:
                st.success(f"Found existing data in `{data_dir}`: {', '.join(jsonl_files)}")
                for fname in jsonl_files:
                    fpath = os.path.join(data_dir, fname)
                    df_peek = pd.read_json(fpath, lines=True, nrows=5)
                    with st.expander(f"Preview: {fname}"):
                        st.dataframe(df_peek, use_container_width=True)
        else:
            st.info(f"No existing data directory found at `{data_dir}`.")

    # --- Clean & Filter ---
    with tab_clean:
        st.subheader("Text Cleaning & Filtering")

        if st.session_state["data_df"] is not None:
            df = st.session_state["data_df"].copy()

            st.markdown("**Cleaning pipeline:**")
            st.markdown("""
            1. Unicode NFC normalization
            2. Remove URLs, HTML tags
            3. Remove @mentions, #hashtags (keep words)
            4. Collapse repeated punctuation
            5. Normalize whitespace
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                min_len = st.number_input("Min words per sentence", 1, 20, 3)
            with col2:
                max_len = st.number_input("Max words per sentence", 50, 500, 128)
            with col3:
                max_ratio = st.number_input("Max src/tgt length ratio", 1.0, 10.0, 3.0)

            if st.button("Run Cleaning Pipeline", type="primary"):
                with st.spinner("Cleaning..."):
                    df["source"] = df["source"].apply(clean_text)
                    df["target"] = df["target"].apply(clean_text)
                    before = len(df)
                    df = df[df.apply(lambda r: is_valid_pair(r["source"], r["target"], min_len, max_len, max_ratio), axis=1)]
                    after = len(df)
                    st.session_state["data_df"] = df

                st.success(f"Cleaning complete: {before:,} -> {after:,} pairs ({before - after:,} removed)")
                st.dataframe(df.head(10), use_container_width=True)
        else:
            st.info("Upload data first in the **Upload Data** tab.")

    # --- Language Tagging ---
    with tab_langtag:
        st.subheader("Per-Token Language ID Tagging")

        st.markdown("""
        Each token is tagged with a language label based on:
        - **Unicode script detection** (Devanagari -> Hindi, Tamil script -> Tamil)
        - **Romanized keyword lookup** for transliterated Hindi/Tamil words
        - Everything else defaults to English
        """)

        test_input = st.text_input(
            "Try language tagging on a sentence:",
            value="kal main market gaya tha for vegetables",
        )
        if test_input:
            tagged = tag_sentence(test_input)
            tagged_str = " ".join([f"`{tok}`[**{LANG_NAMES[lid]}**]" for tok, lid in tagged])
            st.markdown(f"**Tagged:** {tagged_str}")

            lang_dist = {}
            for _, lid in tagged:
                name = LANG_NAMES[lid]
                lang_dist[name] = lang_dist.get(name, 0) + 1
            st.bar_chart(pd.DataFrame({"Count": lang_dist}, index=lang_dist.keys()))

        if st.session_state["data_df"] is not None:
            df = st.session_state["data_df"]
            if st.button("Apply Language Tags to Full Dataset", type="primary"):
                with st.spinner("Tagging all sentences..."):
                    tag_cols = df.apply(apply_lang_tags, axis=1)
                    df = pd.concat([df, tag_cols], axis=1)
                    st.session_state["data_df"] = df
                st.success("Language tagging complete!")
                st.dataframe(df[["source", "source_lang_ids", "target"]].head(10), use_container_width=True)

    # --- Split ---
    with tab_split:
        st.subheader("Train / Validation / Test Split")

        if st.session_state["data_df"] is not None:
            df = st.session_state["data_df"]
            col1, col2, col3 = st.columns(3)
            with col1:
                train_pct = st.slider("Train %", 50, 95, 90)
            with col2:
                val_pct = st.slider("Val %", 1, 25, 5)
            with col3:
                test_pct = 100 - train_pct - val_pct
                st.metric("Test %", f"{test_pct}%")

            if test_pct <= 0:
                st.error("Train + Val must be less than 100%.")
            elif st.button("Create Splits", type="primary"):
                df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
                n = len(df_shuffled)
                n_train = int(n * train_pct / 100)
                n_val = int(n * val_pct / 100)
                train_df = df_shuffled[:n_train]
                val_df = df_shuffled[n_train : n_train + n_val]
                test_df = df_shuffled[n_train + n_val :]

                st.session_state["train_df"] = train_df
                st.session_state["val_df"] = val_df
                st.session_state["test_df"] = test_df

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Train", f"{len(train_df):,}")
                with col2:
                    st.metric("Validation", f"{len(val_df):,}")
                with col3:
                    st.metric("Test", f"{len(test_df):,}")

                save_dir = os.path.join(BASE_DIR, "data", "processed")
                os.makedirs(save_dir, exist_ok=True)
                train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
                val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
                test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
                st.success(f"Splits saved to `{save_dir}/`")
        else:
            st.info("Upload and clean data first.")

    # --- Tokenizer ---
    with tab_tokenizer:
        st.subheader("SentencePiece BPE Tokenizer")

        st.markdown("""
        Trains a custom BPE tokenizer on the combined code-mixed corpus.
        This handles mixed-script tokens (Latin + Devanagari + Tamil) properly.
        """)

        tok_vocab = st.number_input("Vocabulary size", 4000, 64000, 16000, step=1000)

        existing_tok = os.path.join(BASE_DIR, "tokenizer", "codemix_bpe.model")
        if os.path.exists(existing_tok):
            st.success(f"Existing tokenizer found at `{existing_tok}`")
            if st.button("Load & Test Existing Tokenizer"):
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                sp.load(existing_tok)
                st.write(f"Vocab size: **{sp.get_piece_size()}**")
                test_text = "kal main market gaya tha for vegetables"
                pieces = sp.encode_as_pieces(test_text)
                ids = sp.encode_as_ids(test_text)
                st.write(f"Input: `{test_text}`")
                st.write(f"Pieces: `{pieces}`")
                st.write(f"IDs: `{ids}`")
        else:
            st.info("No existing tokenizer found.")

        if st.session_state["train_df"] is not None and st.button("Train New Tokenizer", type="primary"):
            import sentencepiece as spm

            with st.spinner("Training SentencePiece tokenizer..."):
                train_df = st.session_state["train_df"]
                tok_dir = os.path.join(BASE_DIR, "tokenizer")
                os.makedirs(tok_dir, exist_ok=True)

                corpus_path = os.path.join(tok_dir, "corpus.txt")
                with open(corpus_path, "w", encoding="utf-8") as f:
                    for _, row in train_df.iterrows():
                        if "source" in row:
                            f.write(str(row["source"]) + "\n")
                        if "target" in row:
                            f.write(str(row["target"]) + "\n")

                model_prefix = os.path.join(tok_dir, "codemix_bpe")
                spm.SentencePieceTrainer.train(
                    input=corpus_path,
                    model_prefix=model_prefix,
                    vocab_size=tok_vocab,
                    model_type="bpe",
                    pad_id=0,
                    unk_id=1,
                    bos_id=2,
                    eos_id=3,
                    character_coverage=0.9995,
                    input_sentence_size=500000,
                    shuffle_input_sentence=True,
                    user_defined_symbols=["<translate>"],
                )

            st.success(f"Tokenizer trained and saved to `{model_prefix}.model`")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Phase 2 — Architecture":
    st.title("Phase 2: CodeMix-T Architecture")
    st.markdown("---")

    tab_config, tab_build, tab_sanity = st.tabs(["Configure", "Build Model", "Sanity Checks"])

    with tab_config:
        st.subheader("Model Configuration")

        col1, col2 = st.columns(2)
        with col1:
            preset = st.selectbox("Preset", ["Base (512d, 6L)", "Small (256d, 4L)", "Custom"])
        with col2:
            st.metric("Device", str(DEVICE))

        if preset == "Base (512d, 6L)":
            d_model, d_ff, n_heads = 512, 2048, 8
            enc_layers, dec_layers = 6, 6
            lang_dim = 64
        elif preset == "Small (256d, 4L)":
            d_model, d_ff, n_heads = 256, 1024, 4
            enc_layers, dec_layers = 4, 4
            lang_dim = 32
        else:
            d_model = st.number_input("d_model", 64, 1024, 512, step=64)
            d_ff = st.number_input("d_ff", 256, 4096, 2048, step=256)
            n_heads = st.number_input("num_heads", 1, 16, 8)
            enc_layers = st.number_input("Encoder layers", 1, 12, 6)
            dec_layers = st.number_input("Decoder layers", 1, 12, 6)
            lang_dim = st.number_input("Language embed dim", 0, 256, 64, step=16)

        col1, col2, col3 = st.columns(3)
        with col1:
            vocab_size = st.number_input("Vocab size", 4000, 64000, 16000, step=1000)
        with col2:
            max_seq = st.number_input("Max seq length", 32, 512, 128, step=32)
        with col3:
            dropout = st.number_input("Dropout", 0.0, 0.5, 0.1, step=0.05)

        cfg = CodeMixTConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=n_heads,
            num_enc_layers=enc_layers,
            num_dec_layers=dec_layers,
            lang_embed_dim=lang_dim,
            max_seq_len=max_seq,
            dropout=dropout,
        )
        st.session_state["cfg"] = cfg

        st.markdown("---")
        config_data = {
            "d_model": cfg.d_model, "d_ff": cfg.d_ff, "num_heads": cfg.num_heads,
            "d_k (per head)": cfg.d_k, "Encoder layers": cfg.num_enc_layers,
            "Decoder layers": cfg.num_dec_layers, "Vocab size": cfg.vocab_size,
            "LangID embed dim": cfg.lang_embed_dim, "Max seq len": cfg.max_seq_len,
            "Dropout": cfg.dropout,
        }
        st.json(config_data)

    with tab_build:
        st.subheader("Build & Inspect Model")
        cfg = st.session_state.get("cfg")
        if cfg is None:
            st.info("Configure the model first.")
        else:
            if st.button("Build Model", type="primary"):
                with st.spinner("Building CodeMix-T..."):
                    model = CodeMixT(cfg).to(DEVICE)
                    st.session_state["model"] = model

                n_params = model.count_parameters()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Parameters", f"{n_params:,}")
                with col2:
                    st.metric("Size (fp32)", f"{n_params * 4 / 1e6:.1f} MB")
                with col3:
                    st.metric("Size (millions)", f"{n_params / 1e6:.1f}M")

                st.markdown("---")
                st.subheader("Component Breakdown")
                summary_df = get_model_summary(model)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            if st.session_state.get("model") is not None:
                model = st.session_state["model"]

                save_dir = os.path.join(BASE_DIR, "model")
                os.makedirs(save_dir, exist_ok=True)

                if st.button("Save Config & Initial Weights"):
                    config_path = os.path.join(save_dir, "config.json")
                    config_dict = {
                        "vocab_size": cfg.vocab_size, "pad_id": cfg.pad_id,
                        "bos_id": cfg.bos_id, "eos_id": cfg.eos_id,
                        "num_languages": cfg.num_languages, "lang_embed_dim": cfg.lang_embed_dim,
                        "d_model": cfg.d_model, "d_ff": cfg.d_ff, "num_heads": cfg.num_heads,
                        "num_enc_layers": cfg.num_enc_layers, "num_dec_layers": cfg.num_dec_layers,
                        "max_seq_len": cfg.max_seq_len, "dropout": cfg.dropout,
                        "beam_size": cfg.beam_size, "max_gen_len": cfg.max_gen_len,
                    }
                    with open(config_path, "w") as f:
                        json.dump(config_dict, f, indent=2)
                    weights_path = os.path.join(save_dir, "codemix_t_init.pt")
                    torch.save(model.state_dict(), weights_path)
                    st.success(f"Saved config and weights to `{save_dir}/`")

    with tab_sanity:
        st.subheader("Architecture Sanity Checks")
        model = st.session_state.get("model")
        cfg = st.session_state.get("cfg")
        if model is None or cfg is None:
            st.info("Build the model first.")
        else:
            if st.button("Run All Checks", type="primary"):
                results = []
                model.eval()
                B, SRC, TGT = 4, 25, 18

                try:
                    src_ids = torch.randint(4, cfg.vocab_size, (B, SRC)).to(DEVICE)
                    lang_ids = torch.randint(0, cfg.num_languages, (B, SRC)).to(DEVICE)
                    tgt_ids = torch.randint(4, cfg.vocab_size, (B, TGT)).to(DEVICE)
                    with torch.no_grad():
                        logits = model(src_ids, lang_ids, tgt_ids)
                    assert logits.shape == (B, TGT, cfg.vocab_size)
                    results.append(("Forward pass shape", "PASS", f"{logits.shape}"))
                except Exception as e:
                    results.append(("Forward pass shape", "FAIL", str(e)))

                try:
                    assert not torch.isnan(logits).any()
                    results.append(("No NaN in output", "PASS", ""))
                except Exception as e:
                    results.append(("No NaN in output", "FAIL", str(e)))

                try:
                    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_id)
                    loss = criterion(logits.reshape(-1, cfg.vocab_size), tgt_ids.reshape(-1))
                    expected = math.log(cfg.vocab_size)
                    results.append(("Loss computation", "PASS", f"loss={loss.item():.4f} (expected ~{expected:.2f})"))
                except Exception as e:
                    results.append(("Loss computation", "FAIL", str(e)))

                try:
                    beam_decoder = BeamSearchDecoder(model, cfg)
                    out = beam_decoder.decode(src_ids[:1], lang_ids[:1], max_len=20)
                    results.append(("Beam search", "PASS", f"output_ids={out[:10]}..."))
                except Exception as e:
                    results.append(("Beam search", "FAIL", str(e)))

                try:
                    tgt_check = torch.tensor([[cfg.bos_id, 100, 200, 300]]).to(DEVICE)
                    mask = make_tgt_mask(tgt_check, cfg.pad_id)
                    assert mask[0, 0, 0, 1].item() is False
                    results.append(("Causal masking", "PASS", ""))
                except Exception as e:
                    results.append(("Causal masking", "FAIL", str(e)))

                try:
                    tok_only = torch.randint(4, cfg.vocab_size, (1, 8)).to(DEVICE)
                    lang_a = torch.zeros(1, 8, dtype=torch.long).to(DEVICE)
                    lang_b = torch.ones(1, 8, dtype=torch.long).to(DEVICE)
                    emb = model.encoder.embedding
                    with torch.no_grad():
                        out_a = emb(tok_only, lang_a)
                        out_b = emb(tok_only, lang_b)
                    assert not torch.allclose(out_a, out_b)
                    diff = (out_a - out_b).abs().mean().item()
                    results.append(("LangID embedding effect", "PASS", f"mean_diff={diff:.4f}"))
                except Exception as e:
                    results.append(("LangID embedding effect", "FAIL", str(e)))

                df_checks = pd.DataFrame(results, columns=["Check", "Status", "Details"])
                st.dataframe(df_checks, use_container_width=True, hide_index=True)

                n_pass = sum(1 for r in results if r[1] == "PASS")
                if n_pass == len(results):
                    st.success(f"All {n_pass} checks passed! Architecture is ready for training.")
                else:
                    st.warning(f"{n_pass}/{len(results)} checks passed.")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Phase 3 — Training":
    st.title("Phase 3: Training Loop")
    st.markdown("---")

    tab_setup, tab_train, tab_lr = st.tabs(["Setup", "Train", "LR Schedule"])

    with tab_setup:
        st.subheader("Training Configuration")

        col1, col2 = st.columns(2)
        with col1:
            num_epochs = st.number_input("Epochs", 1, 100, 30)
            batch_size = st.number_input("Batch size", 4, 256, 32, step=4)
            warmup_steps = st.number_input("Warmup steps", 500, 10000, 4000, step=500)
        with col2:
            label_smoothing = st.number_input("Label smoothing", 0.0, 0.5, 0.1, step=0.05)
            grad_clip = st.number_input("Gradient clip", 0.1, 10.0, 1.0, step=0.1)
            patience = st.number_input("Early stopping patience", 1, 20, 5)

        st.markdown("---")
        st.subheader("Load Existing Model / Config")

        config_path = os.path.join(BASE_DIR, "model", "config.json")
        init_path = os.path.join(BASE_DIR, "model", "codemix_t_init.pt")

        if os.path.exists(config_path):
            st.success(f"Config found: `{config_path}`")
        else:
            st.warning(f"No config at `{config_path}`. Build model in Phase 2 first.")

        if os.path.exists(init_path):
            st.success(f"Initial weights found: `{init_path}`")
        else:
            st.info("No initial weights found. Will use random initialization.")

        if st.button("Load Model from Config"):
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg_dict = json.load(f)
                cfg = CodeMixTConfig(**cfg_dict)
                st.session_state["cfg"] = cfg
                model = CodeMixT(cfg).to(DEVICE)
                if os.path.exists(init_path):
                    model.load_state_dict(torch.load(init_path, map_location=DEVICE, weights_only=True))
                    st.success("Model loaded with saved initial weights.")
                else:
                    st.info("Model created with random initialization.")
                st.session_state["model"] = model
                st.write(f"Parameters: **{model.count_parameters() / 1e6:.1f}M**")
            else:
                st.error("Config file not found.")

    with tab_train:
        st.subheader("Training")

        model = st.session_state.get("model")
        cfg = st.session_state.get("cfg")

        if model is None or cfg is None:
            st.info("Load or build a model first (Phase 2 or Setup tab).")
        else:
            data_dir = os.path.join(BASE_DIR, "data", "final")
            train_path = os.path.join(data_dir, "train.jsonl")
            val_path = os.path.join(data_dir, "val.jsonl")

            if not os.path.exists(train_path) or not os.path.exists(val_path):
                st.warning(f"Training data not found at `{data_dir}/`. Run Phase 1 first to create JSONL files.")
            else:
                st.success(f"Training data found at `{data_dir}/`")

                if st.button("Start Training", type="primary"):
                    try:
                        train_ds = CodeMixDataset(train_path)
                        val_ds = CodeMixDataset(val_path)
                    except Exception as e:
                        st.error(f"Error loading data: {e}")
                        st.stop()

                    train_loader = torch.utils.data.DataLoader(
                        train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, cfg.pad_id), num_workers=0,
                    )
                    val_loader = torch.utils.data.DataLoader(
                        val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, cfg.pad_id), num_workers=0,
                    )

                    st.write(f"Train: **{len(train_ds):,}** samples, **{len(train_loader)}** batches")
                    st.write(f"Val: **{len(val_ds):,}** samples, **{len(val_loader)}** batches")

                    criterion = LabelSmoothingLoss(cfg.vocab_size, cfg.pad_id, label_smoothing)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
                    scheduler = WarmupInvSqrtScheduler(optimizer, cfg.d_model, warmup_steps)

                    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
                    os.makedirs(ckpt_dir, exist_ok=True)

                    best_val_loss = float("inf")
                    no_improve = 0
                    training_log = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metrics_container = st.empty()
                    chart_container = st.empty()

                    for epoch in range(num_epochs):
                        model.train()
                        epoch_loss = 0.0
                        t0 = time.time()

                        for step_i, batch in enumerate(train_loader):
                            loss_val, lr = train_one_step(model, batch, optimizer, scheduler, criterion, DEVICE, grad_clip)
                            epoch_loss += loss_val
                            progress = (epoch * len(train_loader) + step_i + 1) / (num_epochs * len(train_loader))
                            progress_bar.progress(min(progress, 1.0))
                            if (step_i + 1) % 10 == 0:
                                status_text.text(
                                    f"Epoch {epoch + 1}/{num_epochs} | "
                                    f"Step {step_i + 1}/{len(train_loader)} | "
                                    f"Loss: {loss_val:.4f} | LR: {lr:.2e}"
                                )

                        avg_train_loss = epoch_loss / max(len(train_loader), 1)
                        val_loss, val_ppl = validate(model, val_loader, criterion, DEVICE)
                        epoch_time = time.time() - t0

                        training_log.append({
                            "Epoch": epoch + 1,
                            "Train Loss": round(avg_train_loss, 4),
                            "Val Loss": round(val_loss, 4),
                            "Val PPL": round(val_ppl, 2),
                            "LR": round(lr, 6),
                            "Time (s)": round(epoch_time, 1),
                        })

                        with metrics_container.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Epoch", f"{epoch + 1}/{num_epochs}")
                            with col2:
                                st.metric("Train Loss", f"{avg_train_loss:.4f}")
                            with col3:
                                st.metric("Val Loss", f"{val_loss:.4f}")
                            with col4:
                                st.metric("Val PPL", f"{val_ppl:.2f}")

                        log_df = pd.DataFrame(training_log)
                        with chart_container.container():
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=log_df["Epoch"], y=log_df["Train Loss"], name="Train Loss", mode="lines+markers"))
                            fig.add_trace(go.Scatter(x=log_df["Epoch"], y=log_df["Val Loss"], name="Val Loss", mode="lines+markers"))
                            fig.update_layout(title="Training Progress", xaxis_title="Epoch", yaxis_title="Loss", height=400)
                            st.plotly_chart(fig, use_container_width=True)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save({
                                "epoch": epoch + 1, "val_loss": val_loss,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                            }, os.path.join(ckpt_dir, "codemix_t_best.pt"))
                            no_improve = 0
                        else:
                            no_improve += 1
                            if no_improve >= patience:
                                st.warning(f"Early stopping at epoch {epoch + 1}")
                                break

                    progress_bar.progress(1.0)
                    st.session_state["training_log"] = training_log
                    st.success(f"Training complete! Best val_loss = {best_val_loss:.4f}")
                    st.dataframe(pd.DataFrame(training_log), use_container_width=True, hide_index=True)

    with tab_lr:
        st.subheader("Learning Rate Schedule Visualization")

        col1, col2 = st.columns(2)
        with col1:
            viz_d_model = st.number_input("d_model (for viz)", 64, 1024, 512, step=64, key="lr_dmodel")
        with col2:
            viz_warmup = st.number_input("Warmup steps (for viz)", 500, 10000, 4000, step=500, key="lr_warmup")

        total_steps = st.slider("Total steps to plot", 1000, 50000, 20000, step=1000)

        steps = list(range(1, total_steps + 1))
        dummy_sched = WarmupInvSqrtScheduler(torch.optim.Adam([torch.zeros(1)]), viz_d_model, viz_warmup)
        lrs = []
        for s in steps:
            dummy_sched.step_num = s
            lrs.append(dummy_sched._get_lr())

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=lrs, mode="lines", name="LR"))
        fig.update_layout(title="Warmup + Inverse Sqrt Schedule", xaxis_title="Step", yaxis_title="Learning Rate", height=400)
        st.plotly_chart(fig, use_container_width=True)

        peak_lr = max(lrs)
        peak_step = lrs.index(peak_lr) + 1
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak LR", f"{peak_lr:.6f}")
        with col2:
            st.metric("Peak at Step", f"{peak_step}")
        with col3:
            st.metric("LR at step 10K", f"{lrs[min(9999, len(lrs) - 1)]:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — ABLATION STUDIES
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Phase 4 — Ablation Studies":
    st.title("Phase 4: Ablation Studies")
    st.markdown("---")

    tab_configs, tab_run, tab_results = st.tabs(["Configurations", "Run Ablations", "Results"])

    with tab_configs:
        st.subheader("Ablation Experiment Configurations")
        st.markdown("""
        Each ablation modifies **one component** to measure its contribution.
        This provides clean evidence for the paper about what each part contributes.
        """)

        for name, cfg_dict in ABLATION_CONFIGS.items():
            with st.expander(f"**{name}** — {cfg_dict['description']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**d_model:** {cfg_dict['d_model']}")
                    st.write(f"**d_ff:** {cfg_dict['d_ff']}")
                    st.write(f"**num_heads:** {cfg_dict['num_heads']}")
                with col2:
                    st.write(f"**enc/dec layers:** {cfg_dict['num_enc_layers']}/{cfg_dict['num_dec_layers']}")
                    st.write(f"**lang_embed_dim:** {cfg_dict['lang_embed_dim']}")
                    st.write(f"**use_lang_id:** {cfg_dict['use_lang_id']}")
                    st.write(f"**languages:** {cfg_dict['train_languages']}")

        st.markdown("---")
        comparison = []
        for name, c in ABLATION_CONFIGS.items():
            comparison.append({
                "Experiment": name,
                "d_model": c["d_model"],
                "Layers": f"{c['num_enc_layers']}E/{c['num_dec_layers']}D",
                "LangID": c["use_lang_id"],
                "LangID dim": c["lang_embed_dim"],
                "Languages": c["train_languages"],
            })
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    with tab_run:
        st.subheader("Run Ablation Experiments")
        st.markdown("""
        Select which ablation experiments to run. Each trains a variant model
        for a reduced number of epochs (for relative comparison).
        """)

        selected = st.multiselect(
            "Select experiments",
            list(ABLATION_CONFIGS.keys()),
            default=list(ABLATION_CONFIGS.keys()),
        )
        abl_epochs = st.number_input("Epochs per ablation", 1, 30, 10)

        data_dir = os.path.join(BASE_DIR, "data", "final")
        train_path = os.path.join(data_dir, "train.jsonl")
        val_path = os.path.join(data_dir, "val.jsonl")

        if not os.path.exists(train_path):
            st.warning("Training data not found. Run Phase 1 first.")
        elif st.button("Run Selected Ablations", type="primary"):
            main_cfg = st.session_state.get("cfg")
            if main_cfg is None:
                main_cfg = CodeMixTConfig()
                st.session_state["cfg"] = main_cfg

            abl_results = {}
            progress = st.progress(0)

            for i, exp_name in enumerate(selected):
                exp_cfg = ABLATION_CONFIGS[exp_name]
                st.write(f"Running **{exp_name}**: {exp_cfg['description']}...")

                abl_cfg = CodeMixTConfig(
                    vocab_size=main_cfg.vocab_size,
                    d_model=exp_cfg["d_model"], d_ff=exp_cfg["d_ff"],
                    num_heads=exp_cfg["num_heads"],
                    num_enc_layers=exp_cfg["num_enc_layers"],
                    num_dec_layers=exp_cfg["num_dec_layers"],
                    lang_embed_dim=exp_cfg["lang_embed_dim"],
                    max_seq_len=main_cfg.max_seq_len, dropout=main_cfg.dropout,
                )

                abl_model = CodeMixT(abl_cfg).to(DEVICE)
                if not exp_cfg["use_lang_id"]:
                    abl_model.encoder.embedding = CodeMixEmbeddingAblation(
                        abl_cfg, use_lang_id=False
                    ).to(DEVICE)

                try:
                    train_ds = CodeMixDataset(train_path)
                    val_ds = CodeMixDataset(val_path)
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    continue

                batch_size = 32
                train_loader = torch.utils.data.DataLoader(
                    train_ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b, p=abl_cfg.pad_id: collate_fn(b, p), num_workers=0,
                )
                val_loader = torch.utils.data.DataLoader(
                    val_ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b, p=abl_cfg.pad_id: collate_fn(b, p), num_workers=0,
                )

                abl_opt = torch.optim.Adam(abl_model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
                abl_sched = WarmupInvSqrtScheduler(abl_opt, abl_cfg.d_model)
                abl_crit = LabelSmoothingLoss(abl_cfg.vocab_size, abl_cfg.pad_id)

                best_val = float("inf")
                for epoch in range(abl_epochs):
                    abl_model.train()
                    for batch in train_loader:
                        train_one_step(abl_model, batch, abl_opt, abl_sched, abl_crit, DEVICE)
                    val_loss, val_ppl = validate(abl_model, val_loader, abl_crit, DEVICE)
                    if val_loss < best_val:
                        best_val = val_loss
                        ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        torch.save(abl_model.state_dict(), os.path.join(ckpt_dir, f"ablation_{exp_name}_best.pt"))

                abl_results[exp_name] = {
                    "description": exp_cfg["description"],
                    "best_val_loss": best_val,
                    "params": abl_model.count_parameters(),
                }
                del abl_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                progress.progress((i + 1) / len(selected))

            if abl_results:
                st.session_state["ablation_results"] = abl_results
                st.success("All ablations complete!")

    with tab_results:
        st.subheader("Ablation Results Comparison")

        abl_results = st.session_state.get("ablation_results")

        existing_csv = os.path.join(BASE_DIR, "ablation_results.csv")
        if abl_results:
            rows = []
            for k, v in abl_results.items():
                rows.append({
                    "Experiment": k,
                    "Description": v["description"],
                    "Val Loss": round(v["best_val_loss"], 4),
                    "Perplexity": round(math.exp(min(v["best_val_loss"], 20)), 2),
                    "Params (M)": round(v["params"] / 1e6, 1),
                })
            results_df = pd.DataFrame(rows).sort_values("Val Loss")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            import plotly.express as px
            fig = px.bar(results_df, x="Experiment", y="Val Loss", color="Experiment",
                         title="Ablation Comparison — Validation Loss")
            st.plotly_chart(fig, use_container_width=True)

            results_df.to_csv(existing_csv, index=False)
            st.success(f"Results saved to `{existing_csv}`")

        elif os.path.exists(existing_csv):
            results_df = pd.read_csv(existing_csv)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.info("No ablation results yet. Run experiments in the **Run Ablations** tab.")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Phase 5 — Evaluation":
    st.title("Phase 5: Evaluation")
    st.markdown("---")

    tab_load, tab_scores, tab_baseline, tab_errors = st.tabs([
        "Load Model", "BLEU & chrF++", "mBART Baseline", "Error Analysis",
    ])

    with tab_load:
        st.subheader("Load Best Trained Model")

        config_path = os.path.join(BASE_DIR, "model", "config.json")
        ckpt_path = os.path.join(BASE_DIR, "checkpoints", "codemix_t_best.pt")
        tok_path = os.path.join(BASE_DIR, "tokenizer", "codemix_bpe.model")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Config: {'Found' if os.path.exists(config_path) else 'Missing'}")
        with col2:
            st.write(f"Checkpoint: {'Found' if os.path.exists(ckpt_path) else 'Missing'}")
        with col3:
            st.write(f"Tokenizer: {'Found' if os.path.exists(tok_path) else 'Missing'}")

        if st.button("Load Model for Evaluation", type="primary"):
            if not all(os.path.exists(p) for p in [config_path, ckpt_path, tok_path]):
                st.error("Missing files. Train the model first (Phase 3).")
            else:
                with st.spinner("Loading model..."):
                    with open(config_path) as f:
                        cfg = CodeMixTConfig(**json.load(f))
                    st.session_state["cfg"] = cfg
                    model = CodeMixT(cfg).to(DEVICE)
                    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                    if "model" in ckpt:
                        model.load_state_dict(ckpt["model"])
                        val_loss = ckpt.get("val_loss", "N/A")
                    else:
                        model.load_state_dict(ckpt)
                        val_loss = "N/A"
                    model.eval()
                    st.session_state["model"] = model

                    chatbot = CodeMixTChatbot(model, tok_path, cfg)
                    st.session_state["chatbot"] = chatbot

                st.success(f"Model loaded! Val loss: {val_loss}")

    with tab_scores:
        st.subheader("Automatic Evaluation Metrics")

        chatbot = st.session_state.get("chatbot")
        if chatbot is None:
            st.info("Load the model first in the **Load Model** tab.")
        else:
            test_path = os.path.join(BASE_DIR, "data", "final", "test.jsonl")
            if not os.path.exists(test_path):
                st.warning("Test data not found. Run Phase 1 to create test.jsonl.")
            elif st.button("Run Evaluation", type="primary"):
                df_test = pd.read_json(test_path, lines=True)
                st.write(f"Test samples: **{len(df_test):,}**")

                predictions, references, sources = [], [], []
                progress = st.progress(0)
                for i, (_, row) in enumerate(df_test.iterrows()):
                    pred = chatbot.translate(row["source"])
                    predictions.append(pred)
                    references.append(row["target"])
                    sources.append(row["source"])
                    if (i + 1) % 10 == 0:
                        progress.progress((i + 1) / len(df_test))
                progress.progress(1.0)

                from sacrebleu.metrics import BLEU, CHRF
                bleu_metric = BLEU()
                chrf_metric = CHRF(word_order=2)

                b = bleu_metric.corpus_score(predictions, [references])
                c = chrf_metric.corpus_score(predictions, [references])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("BLEU", f"{b.score:.2f}")
                with col2:
                    st.metric("chrF++", f"{c.score:.2f}")

                if "language" in df_test.columns:
                    st.markdown("---")
                    st.subheader("Per-Language Scores")
                    for lang in df_test["language"].unique():
                        mask = df_test["language"] == lang
                        lang_preds = [predictions[i] for i in range(len(predictions)) if mask.iloc[i]]
                        lang_refs = [references[i] for i in range(len(references)) if mask.iloc[i]]
                        if lang_preds:
                            lb = bleu_metric.corpus_score(lang_preds, [lang_refs])
                            lc = chrf_metric.corpus_score(lang_preds, [lang_refs])
                            st.write(f"**{lang}** ({len(lang_preds)} samples): BLEU={lb.score:.2f}, chrF++={lc.score:.2f}")

                st.session_state["eval_preds"] = predictions
                st.session_state["eval_refs"] = references
                st.session_state["eval_srcs"] = sources

                st.markdown("---")
                st.subheader("Sample Translations")
                sample_df = pd.DataFrame({
                    "Source": sources[:20],
                    "Prediction": predictions[:20],
                    "Reference": references[:20],
                })
                st.dataframe(sample_df, use_container_width=True, hide_index=True)

    with tab_baseline:
        st.subheader("mBART-50 Baseline (Zero-Shot)")
        st.markdown("""
        Compare CodeMix-T against mBART-50 (`facebook/mbart-large-50-many-to-many-mmt`),
        a large pretrained multilingual model, used in zero-shot mode.
        """)

        if st.button("Run mBART-50 Baseline", type="primary"):
            test_path = os.path.join(BASE_DIR, "data", "final", "test.jsonl")
            if not os.path.exists(test_path):
                st.warning("Test data not found.")
            else:
                with st.spinner("Loading mBART-50 (this may take several minutes)..."):
                    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
                    mbart_model = MBartForConditionalGeneration.from_pretrained(
                        "facebook/mbart-large-50-many-to-many-mmt"
                    )
                    mbart_tok = MBart50TokenizerFast.from_pretrained(
                        "facebook/mbart-large-50-many-to-many-mmt"
                    )
                    mbart_model.to(DEVICE)
                    mbart_model.eval()

                df_test = pd.read_json(test_path, lines=True)
                sources = df_test["source"].tolist()
                references = df_test["target"].tolist()

                mbart_tok.src_lang = "hi_IN"
                mbart_preds = []
                progress = st.progress(0)
                batch_size = 16
                for i in range(0, len(sources), batch_size):
                    batch = sources[i : i + batch_size]
                    encoded = mbart_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
                    with torch.no_grad():
                        generated = mbart_model.generate(
                            **encoded,
                            forced_bos_token_id=mbart_tok.lang_code_to_id["en_XX"],
                            num_beams=4, max_length=128,
                        )
                    decoded = mbart_tok.batch_decode(generated, skip_special_tokens=True)
                    mbart_preds.extend(decoded)
                    progress.progress(min((i + batch_size) / len(sources), 1.0))

                from sacrebleu.metrics import BLEU, CHRF
                bleu_metric = BLEU()
                chrf_metric = CHRF(word_order=2)
                b = bleu_metric.corpus_score(mbart_preds, [references])
                c = chrf_metric.corpus_score(mbart_preds, [references])

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("mBART-50 BLEU", f"{b.score:.2f}")
                with col2:
                    st.metric("mBART-50 chrF++", f"{c.score:.2f}")

                del mbart_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    with tab_errors:
        st.subheader("Error Analysis")

        preds = st.session_state.get("eval_preds")
        refs = st.session_state.get("eval_refs")
        srcs = st.session_state.get("eval_srcs")

        if preds is None:
            st.info("Run evaluation first in the **BLEU & chrF++** tab.")
        else:
            from sacrebleu.metrics import BLEU
            bleu_metric = BLEU(effective_order=True)

            sentence_bleus = []
            for pred, ref in zip(preds, refs):
                s = bleu_metric.sentence_score(pred, [ref])
                sentence_bleus.append(s.score)

            df_err = pd.DataFrame({
                "Source": srcs,
                "Prediction": preds,
                "Reference": refs,
                "BLEU": sentence_bleus,
            })

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean BLEU", f"{df_err['BLEU'].mean():.2f}")
            with col2:
                st.metric("Median BLEU", f"{df_err['BLEU'].median():.2f}")
            with col3:
                st.metric("BLEU > 20", f"{(df_err['BLEU'] > 20).sum()}")
            with col4:
                st.metric("BLEU = 0", f"{(df_err['BLEU'] == 0).sum()}")

            st.markdown("---")
            import plotly.express as px
            fig = px.histogram(df_err, x="BLEU", nbins=50, title="Sentence-Level BLEU Distribution")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Worst Translations")
            st.dataframe(df_err.nsmallest(20, "BLEU"), use_container_width=True, hide_index=True)

            st.subheader("Best Translations")
            st.dataframe(df_err.nlargest(10, "BLEU"), use_container_width=True, hide_index=True)

            save_path = os.path.join(BASE_DIR, "error_analysis.csv")
            df_err.to_csv(save_path, index=False)
            st.success(f"Error analysis saved to `{save_path}`")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6 — LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════════

elif page == "Phase 6 — Live Demo":
    st.title("Phase 6: Live Translation Demo")
    st.markdown("---")

    tab_chat, tab_batch, tab_details = st.tabs(["Chat", "Batch Translate", "Model Details"])

    with tab_chat:
        chatbot = st.session_state.get("chatbot")

        if chatbot is None:
            st.info("Load a trained model first (Phase 5 > Load Model).")

            config_path = os.path.join(BASE_DIR, "model", "config.json")
            ckpt_path = os.path.join(BASE_DIR, "checkpoints", "codemix_t_best.pt")
            tok_path = os.path.join(BASE_DIR, "tokenizer", "codemix_bpe.model")

            if all(os.path.exists(p) for p in [config_path, ckpt_path, tok_path]):
                if st.button("Quick Load Model"):
                    with st.spinner("Loading..."):
                        with open(config_path) as f:
                            cfg = CodeMixTConfig(**json.load(f))
                        st.session_state["cfg"] = cfg
                        model = CodeMixT(cfg).to(DEVICE)
                        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
                        if "model" in ckpt:
                            model.load_state_dict(ckpt["model"])
                        else:
                            model.load_state_dict(ckpt)
                        model.eval()
                        st.session_state["model"] = model
                        chatbot = CodeMixTChatbot(model, tok_path, cfg)
                        st.session_state["chatbot"] = chatbot
                    st.success("Model loaded!")
                    st.rerun()
        else:
            st.subheader("Translate Code-Mixed Text to English")

            st.markdown("**Try these examples:**")
            example_cols = st.columns(5)
            for i, (text, lang) in enumerate(DEMO_EXAMPLES[:5]):
                with example_cols[i]:
                    if st.button(f"{lang}: {text[:25]}...", key=f"ex_{i}"):
                        st.session_state["demo_input"] = text

            example_cols2 = st.columns(5)
            for i, (text, lang) in enumerate(DEMO_EXAMPLES[5:10]):
                with example_cols2[i]:
                    if st.button(f"{lang}: {text[:25]}...", key=f"ex2_{i}"):
                        st.session_state["demo_input"] = text

            st.markdown("---")

            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            default_val = st.session_state.pop("demo_input", None)
            user_input = st.chat_input("Type Tanglish or Hinglish here...")

            if default_val and not user_input:
                user_input = default_val

            if user_input:
                st.session_state["messages"].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Translating..."):
                        translation = chatbot.translate(user_input)

                    tagged = tag_sentence(user_input)
                    lang_info = " ".join([f"`{tok}`[{LANG_NAMES[lid]}]" for tok, lid in tagged])

                    response = f"**Translation:** {translation}\n\n**Language tags:** {lang_info}"
                    st.markdown(response)
                    st.session_state["messages"].append({"role": "assistant", "content": response})

            if st.session_state["messages"] and st.button("Clear Chat"):
                st.session_state["messages"] = []
                st.rerun()

    with tab_batch:
        st.subheader("Batch Translation")
        chatbot = st.session_state.get("chatbot")
        if chatbot is None:
            st.info("Load a model first.")
        else:
            batch_input = st.text_area(
                "Enter sentences (one per line):",
                height=200,
                placeholder="kal main market gaya tha\nnaan romba tired aa irukken today",
            )
            if st.button("Translate All", type="primary") and batch_input.strip():
                lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
                results = []
                progress = st.progress(0)
                for i, line in enumerate(lines):
                    translation = chatbot.translate(line)
                    tagged = tag_sentence(line)
                    lang_dist = {}
                    for _, lid in tagged:
                        name = LANG_NAMES[lid]
                        lang_dist[name] = lang_dist.get(name, 0) + 1
                    dominant = max(lang_dist, key=lang_dist.get) if lang_dist else "UNK"
                    results.append({
                        "Source": line,
                        "Translation": translation,
                        "Dominant Language": dominant,
                    })
                    progress.progress((i + 1) / len(lines))

                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    with tab_details:
        st.subheader("Model Details")

        model = st.session_state.get("model")
        cfg = st.session_state.get("cfg")

        if model is not None and cfg is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parameters", f"{model.count_parameters() / 1e6:.1f}M")
                st.metric("d_model", cfg.d_model)
                st.metric("d_ff", cfg.d_ff)
            with col2:
                st.metric("Attention Heads", cfg.num_heads)
                st.metric("Encoder Layers", cfg.num_enc_layers)
                st.metric("Decoder Layers", cfg.num_dec_layers)
            with col3:
                st.metric("Vocab Size", f"{cfg.vocab_size:,}")
                st.metric("LangID Embed Dim", cfg.lang_embed_dim)
                st.metric("Max Seq Length", cfg.max_seq_len)

            st.markdown("---")
            st.subheader("Component Breakdown")
            summary_df = get_model_summary(model)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Model Config (JSON)")
            config_dict = {
                "vocab_size": cfg.vocab_size, "pad_id": cfg.pad_id,
                "bos_id": cfg.bos_id, "eos_id": cfg.eos_id,
                "num_languages": cfg.num_languages, "lang_embed_dim": cfg.lang_embed_dim,
                "d_model": cfg.d_model, "d_ff": cfg.d_ff, "num_heads": cfg.num_heads,
                "num_enc_layers": cfg.num_enc_layers, "num_dec_layers": cfg.num_dec_layers,
                "max_seq_len": cfg.max_seq_len, "dropout": cfg.dropout,
                "beam_size": cfg.beam_size, "max_gen_len": cfg.max_gen_len,
            }
            st.json(config_dict)
        else:
            st.info("Load a model to see its details.")
