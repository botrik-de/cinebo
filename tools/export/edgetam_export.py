#!/usr/bin/env python3
"""Exportiert die Teile von EdgeTAM, die Qualcomm nicht fertig anbietet, und prüft die ganze Kette.

Qualcomm AI Hub liefert Encoder, Decoder und Memory-Encoder von EdgeTAM als
TFLite-Modelle. Die Memory-Attention, also der Teil, der ein Objekt über viele
Bilder hinweg wiedererkennt, fehlt dort (Qualcomm rechnet sie auf dem PC). Dieses
Skript baut sie mit festen Formen nach, damit sie auf dem KI-Chip des Handys laufen
kann, und prüft das Ergebnis Zahl für Zahl gegen das Original-Modell von Meta. Dazu
kommen ein kleines Vorbereitungsmodell vor Qualcomms Memory-Encoder, ein Decoder, der
wie in Metas Videoverfolgung rechnet (Qualcomms weicht davon ab), die Konstanten, mit
denen die App die Eingabe-Einbettung selbst rechnet, und ein Prüfvideo mit Metas
Masken. Mit Qualcomms Modellen (--qualcomm) rechnet es zusätzlich jede Kette der App
auf der CPU nach; das ist die Messlatte für das Ergebnis auf dem KI-Chip.

Erzeugt im Ausgabeordner:
  edgetam_memory_attention.tflite      Memory-Attention wie in Stufe 2 (Vergleichskette)
  edgetam_memory_attention_v2.tflite   gleiche Rechnung, für den KI-Chip umgebaut (Stufe 2b)
  edgetam_memory_encoder.tflite        eigener Memory-Encoder, NHWC-Eingang, Maske 256 x 256
  edgetam_memory_prep.tflite           Vorbereitung für Qualcomms Memory-Encoder
  edgetam_decoder_meta.tflite          Decoder wie in Metas Videoverfolgung (drei Masken)
  edgetam_pruefvideo.zip               Prüfvideo: Bilder (1024 x 1024) und Metas Masken (256 x 256)
  edgetam_konstanten.json              Konstanten für die App und alle Prüfergebnisse

Aufruf:
  python edgetam_export.py --repo EdgeTAM --checkpoint EdgeTAM/checkpoints/edgetam.pt \\
      --qualcomm edgetam-tflite-float --qualcomm-w8a8 edgetam-tflite-w8a8 --out out

Schnittstellen (Reihenfolge der Eingänge zählt, Namen vergibt der Konverter):
  Memory-Attention (beide Fassungen gleich)
    0 curr      [1,64,64,256]  image_embeddings des Encoders (ohne no_mem_embed)
    1 maskmem   [1,3584,64]    7 Speicherplätze zu je 512 Tokens. Platz 0 ist das
                               Startbild, Platz 1 bis 6 die letzten Bilder, Platz 6
                               ist das Bild direkt davor.
    2 obj_ptrs  [1,16,256]     Objekt-Zeiger, Reihenfolge egal
    3 bias      [1,1,3648]     0 für gültige Tokens, -10000 für leere Plätze
    Ausgang     [1,64,64,256]  geht als image_embeddings in den Decoder
  Memory-Encoder und Vorbereitung (gleiche Eingänge)
    0 pix_feat  [1,64,64,256]  image_embeddings des Encoders (ohne no_mem_embed)
    1 mask      [1,256,256,1]  Masken-Logits des Decoders
    2 cond      [1,1,1,1]      1 beim angetippten Startbild, sonst 0
    Ausgang     Memory-Encoder: [1,512,64] neuer Speicher-Eintrag
                Vorbereitung: [1,256,64,64] pix_feat und [1,1,1024,1024] mask_for_mem
                für Qualcomms Memory-Encoder (die App erkennt sie an der Form)
  Decoder wie im Original
    0 image_embeddings   [1,64,64,256]   beim Startbild pix_feat (mit no_mem_embed),
                                         sonst der Ausgang der Memory-Attention
    1 high_res_features1 [1,256,256,32]  vom Encoder
    2 high_res_features2 [1,128,128,64]  vom Encoder
    3 sparse             [1,2,256]       Eingabe-Einbettung: Punkt (oder Leerpunkt) und Leerpunkt
    Ausgänge (an der Form erkannt): Masken [1,3,256,256], IoU-Schätzungen [1,3],
                Objekt-Zeiger [1,3,256], Objekt-Wert [1,1]. Es zählt die Maske mit der
                besten IoU-Schätzung samt ihrem Zeiger; ist der Objekt-Wert <= 0, gilt
                „kein Objekt“ (Maske -1024, Zeiger noObjPtr).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MEM = 7  # Speicherplätze: Startbild + 6 letzte Bilder (num_maskmem)
TOK_MEM = 512  # Tokens je Platz: 256 globale + 256 räumliche Latents
TOK_ROPE = 256  # davon die räumlichen, die RoPE bekommen (16 x 16)
N_PTR = 16  # Objekt-Zeiger (max_obj_ptrs_in_encoder)
TOK_PTR = 4  # Tokens je Zeiger (256 / 64)
MEM_DIM = 64
D = 256
HW = 64
N_Q = HW * HW  # 4096
N_K = N_MEM * TOK_MEM + N_PTR * TOK_PTR  # 3648
MASK_VALUE = -10000.0
HALF = D // 2
# Kanalreihenfolge gerade, dann ungerade. Wird auf q und k gleich angewendet, das
# Skalarprodukt bleibt gleich, und RoPE wird zu einfachem Multiplizieren der Hälften.
PERM = torch.cat([torch.arange(0, D, 2), torch.arange(1, D, 2)])


def log(*args):
    print(*args, flush=True)


# --------------------------------------------------------------------------
# Modell laden


def load_predictor(repo: str, checkpoint: str):
    sys.path.insert(0, os.path.abspath(repo))
    import sam2  # noqa: F401  (richtet die Hydra-Konfiguration ein)
    import sam2.modeling.backbones.timm as timm_backbone
    from sam2.build_sam import build_sam2_video_predictor

    # Das Backbone würde sonst ImageNet-Gewichte aus dem Netz laden. Die braucht es nicht,
    # alle Gewichte kommen gleich danach aus dem EdgeTAM-Checkpoint.
    create = timm_backbone.create_model
    timm_backbone.create_model = lambda *args, **kw: create(*args, **{**kw, "pretrained": False})
    model = build_sam2_video_predictor("configs/edgetam.yaml", checkpoint, device="cpu")
    model.eval()
    return model


# --------------------------------------------------------------------------
# Bausteine in Formen, die TFLite und der QNN-Delegate gut verarbeiten


def rope_tables(end_x: int, end_y: int, dim: int = D, theta: float = 10000.0):
    """cos und sin wie compute_axial_cis von SAM2, als reelle Tabellen [N, dim/2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = t % end_x
    t_y = torch.div(t, end_x, rounding_mode="floor")
    ang = torch.cat([torch.outer(t_x, freqs), torch.outer(t_y, freqs)], dim=-1)
    return torch.cos(ang), torch.sin(ang)


def rope(x, cos, sin):
    a, b = x[..., :HALF], x[..., HALF:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class LayerNormExport(nn.Module):
    """LayerNorm als Einzelschritte (MEAN, SUB, MUL, RSQRT), wie in Qualcomms Modellen."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight = nn.Parameter(ln.weight.detach().clone())
        self.bias = nn.Parameter(ln.bias.detach().clone())
        self.eps = ln.eps

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=-1, keepdim=True)
        return xc * torch.rsqrt(var + self.eps) * self.weight + self.bias


def linear_copy(lin: nn.Linear, perm=None, scale: float = 1.0) -> nn.Linear:
    out = nn.Linear(lin.in_features, lin.out_features)
    w = lin.weight.detach().clone()
    b = lin.bias.detach().clone()
    if perm is not None:
        w, b = w[perm], b[perm]
    out.weight = nn.Parameter(w * scale)
    out.bias = nn.Parameter(b * scale)
    return out


def attend(q, k, v, bias, chunks: int):
    kt = k.transpose(1, 2)
    if chunks == 1:
        s = torch.matmul(q, kt)
        if bias is not None:
            s = s + bias
        return torch.matmul(torch.softmax(s, dim=-1), v)
    outs = []
    for qc in torch.chunk(q, chunks, dim=1):
        s = torch.matmul(qc, kt)
        if bias is not None:
            s = s + bias
        outs.append(torch.matmul(torch.softmax(s, dim=-1), v))
    return torch.cat(outs, dim=1)


class MemAttnLayerExport(nn.Module):
    def __init__(self, layer):
        super().__init__()
        sa, ca = layer.self_attn, layer.cross_attn_image
        assert sa.num_heads == 1 and ca.num_heads == 1
        assert sa.internal_dim == D and ca.internal_dim == D and ca.kv_in_dim == MEM_DIM
        assert not layer.pos_enc_at_attn
        assert layer.pos_enc_at_cross_attn_keys and not layer.pos_enc_at_cross_attn_queries
        assert layer.activation_str == "relu"
        scale = 1.0 / math.sqrt(D)  # Skalierung von scaled_dot_product_attention, in q gefaltet
        self.norm1 = LayerNormExport(layer.norm1)
        self.norm2 = LayerNormExport(layer.norm2)
        self.norm3 = LayerNormExport(layer.norm3)
        self.sa_q = linear_copy(sa.q_proj, PERM, scale)
        self.sa_k = linear_copy(sa.k_proj, PERM)
        self.sa_v = linear_copy(sa.v_proj)
        self.sa_o = linear_copy(sa.out_proj)
        self.ca_q = linear_copy(ca.q_proj, PERM, scale)
        self.ca_k = linear_copy(ca.k_proj, PERM)
        self.ca_v = linear_copy(ca.v_proj)
        self.ca_o = linear_copy(ca.out_proj)
        self.linear1 = linear_copy(layer.linear1)
        self.linear2 = linear_copy(layer.linear2)

    def forward(self, x, mem_k, mem_v, bias, cos_q, sin_q, cos_k, sin_k, chunks: int):
        t = self.norm1(x)
        q = rope(self.sa_q(t), cos_q, sin_q)
        k = rope(self.sa_k(t), cos_q, sin_q)
        x = x + self.sa_o(attend(q, k, self.sa_v(t), None, chunks))
        t = self.norm2(x)
        q = rope(self.ca_q(t), cos_q, sin_q)
        k = rope(self.ca_k(mem_k), cos_k, sin_k)
        x = x + self.ca_o(attend(q, k, self.ca_v(mem_v), bias, chunks))
        return x + self.linear2(F.relu(self.linear1(self.norm3(x))))


def vision_pos_64(sam) -> torch.Tensor:
    """Positions-Kodierung der 64x64-Ebene, wie sie der Bild-Encoder mitliefert."""
    with torch.no_grad():
        pe = sam.image_encoder.neck.position_encoding(torch.zeros(1, D, HW, HW))
    return pe.flatten(2).permute(0, 2, 1).contiguous()  # [1, 4096, 256]


def maskmem_pos_512(sam) -> torch.Tensor:
    """Konstante Positions-Kodierung eines Speicher-Eintrags nach dem Perceiver."""
    with torch.no_grad():
        feat = torch.zeros(1, D, HW, HW)
        mask = torch.zeros(1, 1, sam.image_size, sam.image_size)
        out = sam.memory_encoder(feat, mask, skip_mask_sigmoid=True)
        _, pos = sam.spatial_perceiver(out["vision_features"], out["vision_pos_enc"][0])
    return pos[0].contiguous()  # [512, 64]


class MemAttnExport(nn.Module):
    def __init__(self, sam, chunks: int = 1):
        super().__init__()
        ma = sam.memory_attention
        assert ma.pos_enc_at_input and ma.batch_first and len(ma.layers) == 2
        assert sam.num_maskmem == N_MEM and sam.max_obj_ptrs_in_encoder == N_PTR
        assert sam.mem_dim == MEM_DIM and not sam.add_tpos_enc_to_obj_ptrs
        self.chunks = chunks
        self.register_buffer("curr_pos01", 0.1 * vision_pos_64(sam))
        cos_q, sin_q = rope_tables(HW, HW)
        self.register_buffer("cos_q", cos_q)
        self.register_buffer("sin_q", sin_q)
        cos16, sin16 = rope_tables(16, 16)
        cos_k = torch.ones(N_K, HALF)
        sin_k = torch.zeros(N_K, HALF)
        for s in range(N_MEM):
            a = s * TOK_MEM + (TOK_MEM - TOK_ROPE)
            cos_k[a : a + TOK_ROPE] = cos16
            sin_k[a : a + TOK_ROPE] = sin16
        self.register_buffer("cos_k", cos_k)
        self.register_buffer("sin_k", sin_k)
        pos512 = maskmem_pos_512(sam)
        mem_pos = torch.zeros(N_K, MEM_DIM)
        for s in range(N_MEM):
            tpos = sam.maskmem_tpos_enc[N_MEM - s - 1].reshape(MEM_DIM).detach()
            mem_pos[s * TOK_MEM : (s + 1) * TOK_MEM] = pos512 + tpos
        self.register_buffer("mem_pos", mem_pos[None])
        self.layers = nn.ModuleList([MemAttnLayerExport(l) for l in ma.layers])
        self.norm = LayerNormExport(ma.norm)

    def forward(self, curr, maskmem, obj_ptrs, bias):
        x = curr.reshape(1, N_Q, D) + self.curr_pos01
        memory = torch.cat([maskmem, obj_ptrs.reshape(1, N_PTR * TOK_PTR, MEM_DIM)], dim=1)
        mem_k = memory + self.mem_pos
        for layer in self.layers:
            x = layer(x, mem_k, memory, bias, self.cos_q, self.sin_q, self.cos_k, self.sin_k, self.chunks)
        return self.norm(x).reshape(1, HW, HW, D)


def rope_t(xt, cos_t, sin_t):
    """RoPE auf einem gedrehten Tensor [1, D, N]: die Kanäle liegen in Dimension 1."""
    a, b = xt[:, :HALF, :], xt[:, HALF:, :]
    return torch.cat([a * cos_t - b * sin_t, a * sin_t + b * cos_t], dim=1)


class MemAttnLayerExportV2(nn.Module):
    """Wie MemAttnLayerExport, nur so geschrieben, wie der KI-Chip es mag (Stufe 2b).

    Qualcomm schreibt Attention für den Chip als BATCH_MATMUL in kleinen Blöcken. Der erste
    Export rechnete Q·Kᵀ als FULLY_CONNECTED mit wechselnden Gewichten über das ganze
    4096 x 4096 große Zwischenergebnis; das kostete auf dem Chip 190 ms. Hier wird K gleich nach
    der Projektion gedreht ([1, D, N]) und RoPE in dieser Lage gerechnet. So steht zwischen
    TRANSPOSE und Matmul eine Rechnung, der Konverter kann beides nicht zu FULLY_CONNECTED
    verschmelzen, und beide Produkte bleiben BATCH_MATMUL. Die Abfragen laufen in Blöcken,
    deren Zwischenergebnis (256 x 4096) in den schnellen Speicher des Chips passt.
    """

    def __init__(self, layer, chunks: int):
        super().__init__()
        self.inner = MemAttnLayerExport(layer)
        self.chunks = chunks

    @staticmethod
    def norm(ln: LayerNormExport, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        xc = x - mu
        var = torch.mean(xc * xc, dim=-1, keepdim=True)
        return xc * torch.rsqrt(var + ln.eps) * ln.weight + ln.bias

    def attend(self, q, kt, v, bias):
        outs = []
        for qc in torch.chunk(q, self.chunks, dim=1):
            s = torch.matmul(qc, kt)
            if bias is not None:
                s = s + bias
            outs.append(torch.matmul(torch.softmax(s, dim=-1), v))
        return torch.cat(outs, dim=1)

    def forward(self, x, mem_k, mem_v, bias, cos_q, sin_q, cos_qt, sin_qt, cos_kt, sin_kt):
        L = self.inner
        t = self.norm(L.norm1, x)
        q = rope(L.sa_q(t), cos_q, sin_q)
        kt = rope_t(L.sa_k(t).transpose(1, 2), cos_qt, sin_qt)
        x = x + L.sa_o(self.attend(q, kt, L.sa_v(t), None))
        t = self.norm(L.norm2, x)
        q = rope(L.ca_q(t), cos_q, sin_q)
        kt = rope_t(L.ca_k(mem_k).transpose(1, 2), cos_kt, sin_kt)
        x = x + L.ca_o(self.attend(q, kt, L.ca_v(mem_v), bias))
        return x + L.linear2(F.relu(L.linear1(self.norm(L.norm3, x))))


class MemAttnExportV2(nn.Module):
    """Memory-Attention mit denselben Ein- und Ausgängen wie MemAttnExport, chipfreundlich gebaut."""

    def __init__(self, sam, chunks: int = 16):
        super().__init__()
        base = MemAttnExport(sam, chunks=1)
        for name in ("curr_pos01", "cos_q", "sin_q", "mem_pos"):
            self.register_buffer(name, getattr(base, name).clone())
        # RoPE-Tabellen für das gedrehte K: [1, D/2, N]
        for name in ("cos_q", "sin_q", "cos_k", "sin_k"):
            self.register_buffer(name + "t", getattr(base, name).t().contiguous()[None])
        self.layers = nn.ModuleList([MemAttnLayerExportV2(l, chunks) for l in sam.memory_attention.layers])
        self.norm = base.norm

    def forward(self, curr, maskmem, obj_ptrs, bias):
        x = curr.reshape(1, N_Q, D) + self.curr_pos01
        memory = torch.cat([maskmem, obj_ptrs.reshape(1, N_PTR * TOK_PTR, MEM_DIM)], dim=1)
        mem_k = memory + self.mem_pos
        for layer in self.layers:
            x = layer(x, mem_k, memory, bias, self.cos_q, self.sin_q, self.cos_qt, self.sin_qt, self.cos_kt, self.sin_kt)
        return MemAttnLayerExportV2.norm(self.norm, x).reshape(1, HW, HW, D)


class MemPrepExport(nn.Module):
    """Vorbereitung für Qualcomms Memory-Encoder (Stufe 2b).

    Qualcomms Memory-Encoder rechnet ganz auf dem Chip und braucht 20 statt 276 ms, will aber
    pix_feat als NCHW und die Maske schon fertig in 1024 x 1024: beim angetippten Startbild binär,
    sonst als Sigmoid (wie in Qualcomms und Metas Code). Das macht dieses kleine Modell, ebenfalls
    auf dem Chip. Statt (m > 0) rechnet es min(max(m · 10000, 0), 1): genau 0 oder 1 außer für
    Logits zwischen 0 und 0,0001, aber ohne Vergleich und Typwechsel, die der Chip nicht mag.
    Auch in halber Genauigkeit bleibt es exakt (zu große Werte werden unendlich und dann 1).
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, pix_feat, mask, cond):
        pix = pix_feat.permute(0, 3, 1, 2).contiguous()
        m = F.interpolate(mask.permute(0, 3, 1, 2), size=(self.size, self.size), mode="bilinear", align_corners=False)
        step = torch.clamp(m * 1e4, 0.0, 1.0)
        m = cond * step + (1.0 - cond) * torch.sigmoid(m)
        return pix, m


class MemEncExport(nn.Module):
    """Memory-Encoder samt Perceiver. Nimmt die kleine Maske und rechnet sie selbst hoch.

    Rechnet wie MemoryEncoder.forward, nur kommt die Positions-Kodierung als Konstante
    hinein, statt in jedem Durchlauf mit Sinus und Kosinus neu berechnet zu werden.
    """

    def __init__(self, sam):
        super().__init__()
        assert sam.no_obj_embed_spatial is None and sam.spatial_perceiver is not None
        self.me = sam.memory_encoder
        self.sp = sam.spatial_perceiver
        self.scale = float(sam.sigmoid_scale_for_mem_enc)
        self.bias = float(sam.sigmoid_bias_for_mem_enc)
        self.size = int(sam.image_size)
        with torch.no_grad():
            pos = self.me.position_encoding(torch.zeros(1, MEM_DIM, HW, HW))
        self.register_buffer("pos", pos.detach().clone())

    def forward(self, pix_feat, mask, cond):
        pix = pix_feat.permute(0, 3, 1, 2)
        m = F.interpolate(mask.permute(0, 3, 1, 2), size=(self.size, self.size), mode="bilinear", align_corners=False)
        m = cond * (m > 0).float() + (1.0 - cond) * torch.sigmoid(m)
        m = m * self.scale + self.bias
        x = self.me.pix_feat_proj(pix) + self.me.mask_downsampler(m)
        x = self.me.out_proj(self.me.fuser(x))
        feats, _ = self.sp(x, self.pos)
        return feats


def attention_math(self, q, k, v):
    """Wie Attention.forward in SAM2s Decoder, nur ohne scaled_dot_product_attention.

    Gleiche Rechnung (Skalierung 1/sqrt(Kanäle je Kopf)), aber als einfache Matrix-Produkte,
    die der Konverter sicher umsetzt. Wird für den Export an die Attention-Module gehängt.
    """
    q = self._separate_heads(self.q_proj(q), self.num_heads)
    k = self._separate_heads(self.k_proj(k), self.num_heads)
    v = self._separate_heads(self.v_proj(v), self.num_heads)
    s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    return self.out_proj(self._recombine_heads(torch.matmul(torch.softmax(s, dim=-1), v)))


def use_math_attention(sam):
    import types

    n = 0
    for m in sam.sam_mask_decoder.transformer.modules():
        if type(m).__name__ == "Attention":
            m.forward = types.MethodType(attention_math, m)
            n += 1
    assert n > 0
    return n


class DecoderExport(nn.Module):
    """Der Decoder so, wie Metas Videoverfolgung ihn benutzt (Stufe 2b).

    Qualcomms Decoder weicht vom Original ab: Er liefert nur eine Maske (bei unsicherer Maske
    die beste von dreien) mit dem Zeiger des ersten Tokens, und er lässt in der Hochskalierung
    eine LayerNorm weg. Metas Videoverfolgung nimmt beim Klick und in jedem Folgebild die beste
    von drei Masken samt dem Zeiger ihres Tokens. Dieser Export liefert alle drei Masken, ihre
    IoU-Schätzungen, die drei Zeiger und den Objekt-Wert; die App nimmt die Maske mit der besten
    Schätzung. Eingabe-Einbettung wie bei Meta: ein Punkt (oder Leerpunkt) und der angehängte
    Leerpunkt, also zwei Einbettungen.
    """

    def __init__(self, sam):
        super().__init__()
        md = sam.sam_mask_decoder
        assert md.pred_obj_scores and md.use_multimask_token_for_obj_ptr and md.use_high_res_features
        assert sam.multimask_output_in_sam and sam.multimask_output_for_tracking
        assert sam.multimask_min_pt_num == 0 and sam.multimask_max_pt_num == 1
        self.md = md
        self.obj_ptr_proj = sam.obj_ptr_proj
        pe = sam.sam_prompt_encoder
        with torch.no_grad():
            self.register_buffer("dense", pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, HW, HW).clone())
            self.register_buffer("image_pe", pe.get_dense_pe().clone())

    def forward(self, image_embeddings, hr1, hr2, sparse):
        masks, ious, tokens, score = self.md(
            image_embeddings=image_embeddings.permute(0, 3, 1, 2), image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=self.dense, multimask_output=True,
            repeat_image=False, high_res_features=[hr1.permute(0, 3, 1, 2), hr2.permute(0, 3, 1, 2)])
        return masks, ious, self.obj_ptr_proj(tokens), score


def meta_features(sam, img_u8: np.ndarray) -> dict:
    """Bildmerkmale aus Metas Encoder (wie die Videoverfolgung), in der Lage von Qualcomms Encoder (NHWC)."""
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = ((torch.from_numpy(img_u8 / 255.0).permute(2, 0, 1).float() - mean) / std)[None]
    with torch.inference_mode():
        _, feats, _, _ = sam._prepare_backbone_features(sam.forward_image(x))
    emb = feats[-1].permute(1, 2, 0).reshape(1, D, HW, HW).permute(0, 2, 3, 1).contiguous()
    return {
        "emb": emb,
        "pix": emb + sam.no_mem_embed.detach().reshape(1, 1, 1, D),
        "hr1": feats[0].permute(1, 2, 0).reshape(1, 32, 256, 256).permute(0, 2, 3, 1).contiguous(),
        "hr2": feats[1].permute(1, 2, 0).reshape(1, 64, 128, 128).permute(0, 2, 3, 1).contiguous(),
    }


# --------------------------------------------------------------------------
# Eingabe-Einbettung: Die App rechnet sie selbst (PromptEncoder in EdgeTamParts.kt)

# Proben für die Prüfung in der App: Punkte (u, v von 0 bis 1) und Labels wie in SAM2
# (1 positiv, 0 negativ, 2 und 3 Rahmenecken, -1 kein Punkt), auch an den Bildrändern.
PROMPT_PROBEN = [
    ([0.3], [0.6], [1]),
    ([0.5, 0.25], [0.5, 0.75], [1, 0]),
    ([0.1, 0.8], [0.2, 0.95], [2, 3]),
    ([0.0, 1.0], [1.0, 0.0], [1, -1]),
    ([0.999, 0.0], [0.0, 0.999], [0, 1]),
    ([0.0, 0.0], [0.0, 0.0], [-1, -1]),
]


def prompt_constants(sam) -> dict:
    """Alles, was die App für die Eingabe-Einbettung braucht, dazu Proben aus Metas Code."""
    pe = sam.sam_prompt_encoder
    size = int(sam.image_size)
    assert tuple(pe.input_image_size) == (size, size)
    gauss = pe.pe_layer.positional_encoding_gaussian_matrix.detach().float()
    assert tuple(gauss.shape) == (2, HALF), gauss.shape
    points = torch.stack([e.weight.detach().reshape(-1) for e in pe.point_embeddings]).float()
    assert tuple(points.shape) == (4, D)
    proben = []
    with torch.no_grad():
        for u, v, labels in PROMPT_PROBEN:
            # Wie add_new_points_or_box: Punkte in Pixeln des quadratischen Eingangsbilds.
            pts = torch.tensor([[[a * size, b * size] for a, b in zip(u, v)]], dtype=torch.float32)
            lab = torch.tensor([labels], dtype=torch.int32)
            emb = pe._embed_points(pts, lab, pad=True)[0]
            proben.append({"u": u, "v": v, "labels": labels, "einbettung": [[float(x) for x in row] for row in emb]})
    return {
        "bildgroesse": size,
        "gauss": [[float(x) for x in row] for row in gauss],
        "punkte": [[float(x) for x in row] for row in points],
        "keinPunkt": [float(x) for x in pe.not_a_point_embed.weight.detach().reshape(-1)],
        "proben": proben,
    }


# --------------------------------------------------------------------------
# Prüfung gegen das Original: ein kleines Video, in dem EdgeTAM einen Ball verfolgt


def make_video(folder: str, frames: int = 12):
    from PIL import Image, ImageDraw, ImageFilter

    rng = np.random.default_rng(7)
    w, h = 1280, 720
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 80 * xx / w, 70 + 60 * yy / h, 110 - 40 * xx / w], -1)
    noise = rng.normal(0, 1, (h // 16, w // 16, 3))
    noise = np.array(Image.fromarray(((noise * 20) + 128).clip(0, 255).astype(np.uint8)).resize((w, h), Image.BICUBIC))
    base = (base + (noise.astype(np.float32) - 128) * 0.6).clip(0, 255).astype(np.uint8)
    boxes = [(rng.integers(0, w - 200), rng.integers(0, h - 150), rng.integers(80, 220), rng.integers(60, 160)) for _ in range(6)]
    colors = [tuple(int(c) for c in rng.integers(30, 230, 3)) for _ in boxes]
    track = []
    for i in range(frames):
        img = Image.fromarray(base.copy())
        d = ImageDraw.Draw(img)
        for (x, y, bw, bh), c in zip(boxes, colors):
            d.rectangle([x, y, x + bw, y + bh], fill=c)
        cx, cy, r = 260 + i * 60, 380 + int(30 * math.sin(i / 2)), 95
        for k in range(r, 0, -3):  # Ball mit Licht oben links
            t = k / r
            col = (int(200 + 50 * (1 - t)), int(40 + 120 * (1 - t) ** 2), int(30 + 60 * (1 - t) ** 2))
            off = int((1 - t) * r * 0.35)
            d.ellipse([cx - k - off, cy - k - off, cx + k - off, cy + k - off], fill=col)
        img = img.filter(ImageFilter.GaussianBlur(0.6))
        img.save(os.path.join(folder, f"{i:05d}.jpg"), quality=95)
        track.append((cx, cy))
    return track, (w, h)


def load_frame(path: str, size: int) -> np.ndarray:
    """Ein Bild so, wie SAM2s Video-Lader es sieht (vor dem Normieren): RGB, uint8, quadratisch."""
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert("RGB").resize((size, size)))


def run_reference(model, folder: str, point):
    """Lässt das Original verfolgen; schreibt jeden Aufruf der Memory-Attention und jede Maske mit.

    Die Masken sind die kleinen Logits (256 x 256), wie sie auch der Decoder in der App liefert.
    """
    calls = []
    orig = model.memory_attention.forward

    def hook(curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0, num_spatial_mem=-1):
        out = orig(curr=curr, memory=memory, curr_pos=curr_pos, memory_pos=memory_pos,
                   num_obj_ptr_tokens=num_obj_ptr_tokens, num_spatial_mem=num_spatial_mem)
        calls.append({
            "curr": curr[0].detach().clone(), "curr_pos": curr_pos[0].detach().clone(),
            "memory": memory.detach().clone(), "memory_pos": memory_pos.detach().clone(),
            "ptr_tokens": int(num_obj_ptr_tokens), "spatial": int(num_spatial_mem),
            "out": out.detach().clone(),
        })
        return out

    model.memory_attention.forward = hook
    try:
        with torch.inference_mode():
            state = model.init_state(video_path=folder)
            model.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=np.array([point], dtype=np.float32),
                                        labels=np.array([1], dtype=np.int32))
            for _ in model.propagate_in_video(state):
                pass
    finally:
        model.memory_attention.forward = orig
    per_obj = state["output_dict_per_obj"][0]
    masks, scores = [], []
    for f in range(state["num_frames"]):
        out = per_obj["cond_frame_outputs"].get(f) or per_obj["non_cond_frame_outputs"][f]
        masks.append(out["pred_masks"][0, 0].float().numpy().copy())
        scores.append(float(out["object_score_logits"].reshape(-1)[0]))
    return calls, masks, scores, state["images"]


def to_fixed_inputs(call, pos512, tpos, frame: int):
    """Baut aus einem mitgeschriebenen Aufruf die festen Eingänge des Exports."""
    mem, mpos = call["memory"][:, 0], call["memory_pos"][:, 0]
    n_sp, n_pt = call["spatial"], call["ptr_tokens"]
    assert mem.shape[0] == n_sp * TOK_MEM + n_pt and n_pt % TOK_PTR == 0
    maskmem = torch.zeros(N_MEM, TOK_MEM, MEM_DIM)
    bias = torch.full((N_K,), MASK_VALUE)
    slots = []
    for i in range(n_sp):
        p = mpos[i * TOK_MEM : (i + 1) * TOK_MEM]
        dists = [float((p - (pos512 + tpos[N_MEM - s - 1])).abs().max()) for s in range(N_MEM)]
        s = int(np.argmin(dists))
        assert dists[s] < 1e-5, f"Platz nicht erkannt: {dists}"
        assert s not in slots
        slots.append(s)
        maskmem[s] = mem[i * TOK_MEM : (i + 1) * TOK_MEM]
        bias[s * TOK_MEM : (s + 1) * TOK_MEM] = 0.0
    n_ptr = n_pt // TOK_PTR
    assert n_ptr <= N_PTR
    ptrs = torch.zeros(N_PTR, D)
    if n_ptr:
        ptrs[:n_ptr] = mem[n_sp * TOK_MEM :].reshape(n_ptr, D)
        bias[N_MEM * TOK_MEM : N_MEM * TOK_MEM + n_pt] = 0.0
        assert float(mpos[n_sp * TOK_MEM :].abs().max()) == 0.0
    # Erwartete Belegung, so wie die App den Speicher führt:
    # Platz 0 = Startbild, Platz 7 - n = Bild vor n Schritten (n = 1 bis 6).
    expect = sorted([0] + [N_MEM - n for n in range(1, min(6, frame - 1) + 1)])
    assert sorted(slots) == expect, f"Bild {frame}: Plätze {sorted(slots)}, erwartet {expect}"
    assert n_ptr == 1 + min(N_PTR - 1, frame - 1), f"Bild {frame}: {n_ptr} Zeiger"
    curr = call["curr"][:, 0].reshape(1, HW, HW, D)
    return (curr, maskmem.reshape(1, N_MEM * TOK_MEM, MEM_DIM), ptrs[None], bias.reshape(1, 1, N_K))


def compare(a, b) -> dict:
    a = torch.as_tensor(a).double()
    b = torch.as_tensor(b).double()
    diff = (a - b).abs()
    return {"maxAbw": float(diff.max()), "relL2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
            "bereich": [float(b.min()), float(b.max())]}


def iou(mask, ref) -> float:
    """Wie die App vergleicht: Logit größer 0 heißt Objekt."""
    a, b = np.asarray(mask) > 0, np.asarray(ref) > 0
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def write_test_video(path: str, frames, masks):
    """Prüfvideo für die App: bild_NN.png (RGB, 1024 x 1024) und maske_NN.png (256 x 256, 255 = Objekt)."""
    import io
    import zipfile

    from PIL import Image

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for i, (img, m) in enumerate(zip(frames, masks)):
            for name, arr, mode in ((f"bild_{i:02d}.png", img, "RGB"), (f"maske_{i:02d}.png", np.where(m > 0, 255, 0).astype(np.uint8), "L")):
                buf = io.BytesIO()
                Image.fromarray(arr, mode).save(buf, format="PNG", optimize=True)
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))  # feste Zeit: gleicher Inhalt, gleiche Datei
                z.writestr(info, buf.getvalue())


# --------------------------------------------------------------------------
# TFLite


def convert(module: nn.Module, args: tuple, path: str) -> float:
    import litert_torch

    t = time.time()
    edge = litert_torch.convert(module.eval(), args)
    edge.export(path)
    return time.time() - t


class TfliteModel:
    """Ein TFLite-Modell auf der CPU. Eingänge in Modell-Reihenfolge oder nach Namen, Ausgänge nach Namen."""

    def __init__(self, path: str):
        from ai_edge_litert.interpreter import Interpreter

        self.path = path
        self.name = os.path.basename(path)
        self.it = Interpreter(model_path=path, num_threads=os.cpu_count() or 2)
        self.it.allocate_tensors()
        self.inputs = self.it.get_input_details()
        self.outputs = self.it.get_output_details()
        self.last_ms = 0.0

    def run(self, *args, **named) -> dict:
        if args:
            # Die App übergibt die Eingänge in genau dieser Reihenfolge.
            assert len(args) == len(self.inputs) and not named, (self.name, len(args), [d["name"] for d in self.inputs])
            pairs = list(zip(self.inputs, args))
        else:
            by_name = {d["name"]: d for d in self.inputs}
            assert set(named) == set(by_name), (self.name, sorted(named), sorted(by_name))
            pairs = [(by_name[k], v) for k, v in named.items()]
        for d, a in pairs:
            arr = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
            arr = np.ascontiguousarray(arr, dtype=d["dtype"])
            assert list(d["shape"]) == list(arr.shape), (self.name, d["name"], list(d["shape"]), list(arr.shape))
            self.it.set_tensor(d["index"], arr)
        t = time.time()
        self.it.invoke()
        self.last_ms = (time.time() - t) * 1000
        return {d["name"]: self.it.get_tensor(d["index"]).copy() for d in self.outputs}

    def by_shape(self, outs: dict, *shape) -> np.ndarray:
        """Ausgang mit genau dieser Form (so findet auch die App die Ausgänge unserer Modelle)."""
        hits = [v for v in outs.values() if list(v.shape) == list(shape)]
        assert len(hits) == 1, (self.name, shape, [list(v.shape) for v in outs.values()])
        return hits[0]

    def quant(self, name: str, inputs: bool = True):
        d = {x["name"]: x for x in (self.inputs if inputs else self.outputs)}[name]
        return float(d["quantization"][0]), int(d["quantization"][1])

    def interface(self) -> dict:
        return {"ein": [{"name": d["name"], "form": [int(x) for x in d["shape"]], "typ": d["dtype"].__name__} for d in self.inputs],
                "aus": [{"name": d["name"], "form": [int(x) for x in d["shape"]], "typ": d["dtype"].__name__} for d in self.outputs]}

    def ops(self) -> dict:
        """Operationen im Modell selbst (ohne die CPU-Beschleunigung, die sie sonst zusammenfasst)."""
        import collections

        from ai_edge_litert.interpreter import Interpreter, OpResolverType

        it = Interpreter(model_path=self.path, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
        c = collections.Counter(o["op_name"] for o in it._get_ops_details())
        return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


# --------------------------------------------------------------------------
# Die Ketten der App, auf der CPU nachgerechnet (EdgeTamTracker und FastTracker in der App).
# Gleiche Modelle, gleiche Reihenfolge, gleiche Regeln; nur rechnet hier die CPU in voller
# Genauigkeit. Was die App auf dem KI-Chip im Prüfvideo erreicht, lässt sich daran messen.


def app_prompt(pc: dict, u, v, labels) -> np.ndarray:
    """Wie PromptEncoder.embed in der App (Kotlin): (Punkte + 1) x 256 Werte, der letzte ist leer."""
    g = np.asarray(pc["gauss"], np.float32)
    pts = np.asarray(pc["punkte"], np.float32)
    nap = np.asarray(pc["keinPunkt"], np.float32)
    one = np.float32(1.0 / pc["bildgroesse"])
    out = np.zeros((len(labels) + 1, D), np.float32)
    for i in range(len(labels) + 1):
        lab = labels[i] if i < len(labels) else -1
        if lab == -1:
            out[i] = nap
            continue
        x = np.float32(2.0) * np.float32(u[i]) - np.float32(1.0) + one
        y = np.float32(2.0) * np.float32(v[i]) - np.float32(1.0) + one
        ang = 2.0 * np.pi * (x * g[0] + y * g[1]).astype(np.float64)
        out[i, :HALF] = np.sin(ang).astype(np.float32) + pts[lab, :HALF]
        out[i, HALF:] = np.cos(ang).astype(np.float32) + pts[lab, HALF:]
    return out[None]


class QcEncoder:
    """Qualcomms Encoder in float, wie FloatEncoder in der App."""

    def __init__(self, model: TfliteModel):
        self.m = model
        self.name = "fp16"

    def __call__(self, img: np.ndarray, coords=None, labels=None) -> dict:
        coords = np.zeros((1, 2, 2), np.float32) if coords is None else coords
        labels = np.full((1, 2), -1.0, np.float32) if labels is None else labels
        o = self.m.run(image=(img.astype(np.float32) / 255.0)[None], unnorm_coords=coords, labels=labels)
        return {"emb": o["image_embeddings"], "pix": o["pix_feat"], "hr1": o["high_res_features1"],
                "hr2": o["high_res_features2"], "sparse": o["sparse_embedding"]}


class QcEncoderW8a8:
    """Qualcomms quantisierter Encoder, wie QuantEncoder in der App: Bild nach UINT8, Ausgänge zurück."""

    def __init__(self, model: TfliteModel):
        self.m = model
        self.name = "w8a8"
        cs, cz = model.quant("unnorm_coords")
        ls, lz = model.quant("labels")
        self.coords = np.full((1, 2, 2), np.clip(np.round(0.0 / cs + cz), 0, 255), np.uint8)
        self.labels = np.full((1, 2), np.clip(np.round(-1.0 / ls + lz), 0, 255), np.uint8)

    def __call__(self, img: np.ndarray, coords=None, labels=None) -> dict:
        s, z = self.m.quant("image")
        x = (img.astype(np.float32) / np.float32(255.0))[None]
        x8 = np.clip(np.round(x * np.float32(1.0 / s) + z), 0, 255).astype(np.uint8)
        o = self.m.run(image=x8, unnorm_coords=self.coords, labels=self.labels)

        def deq(name):
            s_, z_ = self.m.quant(name, inputs=False)
            return ((o[name].astype(np.int32) - z_) * np.float32(s_)).astype(np.float32)

        return {"emb": deq("image_embeddings"), "pix": deq("pix_feat"), "hr1": deq("high_res_features1"),
                "hr2": deq("high_res_features2"), "sparse": None}


def prompt_for(pc: dict, points: int, u: float, v: float, label: int) -> np.ndarray:
    """Einbettung für einen Decoder, der [points] Punkte erwartet: der Punkt, dann Leerpunkte (wie in der App)."""
    pad = points - 1
    return app_prompt(pc, [u] + [0.0] * pad, [v] + [0.0] * pad, [label] + [-1] * pad)


class QcDecoder:
    """Qualcomms Decoder (QualcommDecoder in der App): eine Maske, zwei Punkte plus Leerpunkt."""

    name = "qualcomm"
    points = 2

    def __init__(self, model: TfliteModel):
        self.m = model

    def __call__(self, emb, hr1, hr2, sparse):
        o = self.m.run(image_embeddings=emb, high_res_features1=hr1, high_res_features2=hr2, sparse_embedding=sparse)
        return o["masks"].reshape(256, 256), o["obj_ptr"].reshape(D), float(o["object_score_logits"].reshape(-1)[0])


class MetaDecoder:
    """Unser Decoder wie im Original (MetaDecoder in der App): drei Masken, die mit der besten IoU-Schätzung zählt."""

    name = "wie Meta"
    points = 1

    def __init__(self, model: TfliteModel):
        self.m = model

    def __call__(self, emb, hr1, hr2, sparse):
        o = self.m.run(emb, hr1, hr2, sparse)
        best = int(np.argmax(self.m.by_shape(o, 1, 3)[0]))
        masks = self.m.by_shape(o, 1, 3, 256, 256)
        ptrs = self.m.by_shape(o, 1, 3, D)
        return masks[0, best], ptrs[0, best], float(self.m.by_shape(o, 1, 1).reshape(-1)[0])


class OwnMemory:
    """Unser Memory-Encoder (OwnMemoryWriter in der App)."""

    name = "eigener"

    def __init__(self, model: TfliteModel):
        self.m = model

    def __call__(self, emb, mask, start: bool) -> np.ndarray:
        o = self.m.run(emb, mask.reshape(1, 256, 256, 1), np.full((1, 1, 1, 1), 1.0 if start else 0.0, np.float32))
        return self.m.by_shape(o, 1, TOK_MEM, MEM_DIM)[0]


class QcMemory:
    """Unsere Vorbereitung und Qualcomms Memory-Encoder (QualcommMemoryWriter in der App)."""

    name = "qualcomm"

    def __init__(self, prep: TfliteModel, model: TfliteModel):
        self.prep, self.m = prep, model

    def __call__(self, emb, mask, start: bool) -> np.ndarray:
        o = self.prep.run(emb, mask.reshape(1, 256, 256, 1), np.full((1, 1, 1, 1), 1.0 if start else 0.0, np.float32))
        pix = self.prep.by_shape(o, 1, D, HW, HW)
        m = self.prep.by_shape(o, 1, 1, 1024, 1024)
        return self.m.run(pix_feat=pix, mask_for_mem=m)["maskmem_features"].reshape(TOK_MEM, MEM_DIM)


class AppChain:
    """EdgeTamTracker der App: Startbild, sechs letzte Bilder, bis zu 16 Objekt-Zeiger."""

    def __init__(self, name, encoder, decoder, attention: TfliteModel, memory, no_obj_ptr, pc=None):
        self.name, self.enc, self.dec, self.attn, self.mem = name, encoder, decoder, attention, memory
        self.no_obj_ptr = np.asarray(no_obj_ptr, np.float32)
        self.pc = pc  # None: Einbettung aus dem Encoder (Stufe 2)
        assert pc is not None or decoder.points == 2, "Die Einbettung des Encoders passt nur zu Qualcomms Decoder"

    def _gate(self, mask, ptr, score):
        if score > 0:
            return mask, ptr
        return np.full_like(mask, -1024.0), self.no_obj_ptr.copy()

    def start(self, img, u, v):
        if self.pc is None:
            e = self.enc(img, np.array([[[u, v], [0.0, 0.0]]], np.float32), np.array([[1.0, -1.0]], np.float32))
            sparse = e["sparse"]
        else:
            e = self.enc(img)
            sparse = prompt_for(self.pc, self.dec.points, u, v, 1)
        mask, ptr, score = self.dec(e["pix"], e["hr1"], e["hr2"], sparse)
        mask, ptr = self._gate(mask, ptr, score)
        self.cond_mem = self.mem(e["emb"], mask, True)
        self.cond_ptr = ptr
        self.recent_mem, self.recent_ptr = [], []
        return mask, score

    def step(self, img):
        e = self.enc(img)
        sparse = e["sparse"] if self.pc is None else prompt_for(self.pc, self.dec.points, 0.0, 0.0, -1)
        bank = np.zeros((N_MEM, TOK_MEM, MEM_DIM), np.float32)
        bias = np.full((N_K,), MASK_VALUE, np.float32)
        bank[0] = self.cond_mem
        bias[:TOK_MEM] = 0.0
        for slot in range(1, N_MEM):
            i = len(self.recent_mem) - (N_MEM - slot)  # Platz 7 - n: Bild vor n Schritten
            if i >= 0:
                bank[slot] = self.recent_mem[i]
                bias[slot * TOK_MEM : (slot + 1) * TOK_MEM] = 0.0
        ptrs = np.zeros((N_PTR, D), np.float32)
        ptrs[0] = self.cond_ptr
        base = N_MEM * TOK_MEM
        bias[base : base + TOK_PTR] = 0.0
        for j in range(1, N_PTR):
            i = len(self.recent_ptr) - j  # neuester zuerst
            if i >= 0:
                ptrs[j] = self.recent_ptr[i]
                bias[base + j * TOK_PTR : base + (j + 1) * TOK_PTR] = 0.0
        o = self.attn.run(e["emb"], bank.reshape(1, N_MEM * TOK_MEM, MEM_DIM), ptrs[None], bias.reshape(1, 1, N_K))
        cur = self.attn.by_shape(o, 1, HW, HW, D)
        mask, ptr, score = self.dec(cur, e["hr1"], e["hr2"], sparse)
        mask, ptr = self._gate(mask, ptr, score)
        self.recent_mem = (self.recent_mem + [self.mem(e["emb"], mask, False)])[-(N_MEM - 1):]
        self.recent_ptr = (self.recent_ptr + [ptr])[-(N_PTR - 1):]
        return mask, score


def mask_center(mask):
    """Wie FastTracker.centerOf in der App: Maskenpunkt nächst dem Schwerpunkt, als (u, v), oder None."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    k = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
    n = mask.shape[0]
    return (float((xs[k] + 0.5) / n), float((ys[k] + 0.5) / n))


class FastChain:
    """FastTracker der App: nur Encoder und Decoder; der Punkt kommt aus der Mitte der Maske davor."""

    def __init__(self, name, encoder, decoder, pc):
        self.name, self.enc, self.dec, self.pc = name, encoder, decoder, pc

    def _track(self, img, u, v):
        e = self.enc(img)
        mask, _, score = self.dec(e["pix"], e["hr1"], e["hr2"], prompt_for(self.pc, self.dec.points, u, v, 1))
        if score <= 0:
            return np.full_like(mask, -1024.0), score
        self.point = mask_center(mask) or self.point  # ohne Objekt bleibt der alte Punkt
        return mask, score

    def start(self, img, u, v):
        self.point = (min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0))
        return self._track(img, u, v)

    def step(self, img):
        return self._track(img, *self.point)


class CachedEncoder:
    """Merkt sich die Merkmale je Bild; Einbettungen aus Punkten rechnet er immer neu."""

    def __init__(self, enc, frames):
        self.enc, self.name = enc, enc.name
        self.ids = {id(f): i for i, f in enumerate(frames)}
        self.cache = {}

    def __call__(self, img, coords=None, labels=None):
        if coords is not None:
            return self.enc(img, coords, labels)
        k = self.ids[id(img)]
        if k not in self.cache:
            self.cache[k] = self.enc(img)
        return self.cache[k]


def track_video(chain, frames, start, ref_masks) -> dict:
    ious, found = [], []
    t = time.time()
    for f, img in enumerate(frames):
        mask, score = chain.start(img, *start) if f == 0 else chain.step(img)
        ious.append(round(iou(mask, ref_masks[f]), 4))
        found.append(bool(score > 0))
    return {"name": chain.name, "jeBild": ious, "maskeGleichMittel": round(float(np.mean(ious)), 4),
            "maskeGleichSchlechtestes": min(ious), "objektGefunden": found, "cpuS": round(time.time() - t, 1)}


# --------------------------------------------------------------------------


ATTN_CHUNKS = 16  # Abfrage-Blöcke der Memory-Attention v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qualcomm", help="Ordner mit Qualcomms EdgeTAM-Modellen in float (encoder, video_decoder, memory_encoder)")
    ap.add_argument("--qualcomm-w8a8", help="Ordner mit Qualcomms quantisiertem EdgeTAM-Encoder")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--no-convert", action="store_true", help="nur prüfen, nichts umwandeln")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    report: dict = {"werkzeuge": {"torch": torch.__version__}}
    try:
        import importlib.metadata as md

        for p in ("litert-torch", "ai-edge-litert", "litert-converter", "timm"):
            report["werkzeuge"][p] = md.version(p)
    except Exception:
        pass

    log("== Modell laden")
    model = load_predictor(a.repo, a.checkpoint)
    sam = model
    size = int(sam.image_size)

    log("== Prüfvideo: das Original verfolgen lassen")
    with tempfile.TemporaryDirectory() as tmp:
        track, (vw, vh) = make_video(tmp, a.frames)
        t = time.time()
        calls, ref_masks, ref_scores, images = run_reference(model, tmp, track[0])
        log(f"   {len(calls)} Aufrufe der Memory-Attention in {time.time() - t:.0f} s")
        frames = [load_frame(os.path.join(tmp, f"{i:05d}.jpg"), size) for i in range(a.frames)]
    # Die Bilder des Prüfvideos müssen genau die sein, die das Original gesehen hat.
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    frame_dev = max(float((images[i].cpu() - (torch.from_numpy(f / 255.0).permute(2, 0, 1).float() - mean) / std).abs().max())
                    for i, f in enumerate(frames))
    assert frame_dev < 1e-5, frame_dev
    # Der Ball muss im letzten Bild noch gefunden werden, sonst taugt die Prüfung nichts.
    cx, cy = track[-1]
    hit = bool(ref_masks[-1][min(int(cy / vh * 256), 255), min(int(cx / vw * 256), 255)] > 0)
    share = [round(float((m > 0).mean()), 4) for m in ref_masks]
    report["verfolgung"] = {"bilder": a.frames, "ballImLetztenBild": hit, "maskenAnteil": share,
                            "objektWerte": [round(s, 3) for s in ref_scores], "bilderWieOriginal": frame_dev}
    log(f"   Ball im letzten Bild getroffen: {hit}, Maskenanteile {share}")
    assert hit, "Das Original verliert den Ball, Prüfvideo taugt nicht"
    start_uv = [track[0][0] / vw, track[0][1] / vh]

    pos512 = maskmem_pos_512(sam)
    tpos = sam.maskmem_tpos_enc.detach().reshape(N_MEM, MEM_DIM)
    attn = {"v1": MemAttnExport(sam, chunks=1).eval(), "v2": MemAttnExportV2(sam, chunks=ATTN_CHUNKS).eval()}
    ref_pos = vision_pos_64(sam)
    pos_dev = float((calls[0]["curr_pos"][:, 0] - ref_pos[0]).abs().max())
    report["positionsKodierung"] = pos_dev
    assert pos_dev < 1e-5, pos_dev

    log("== Memory-Attention: Nachbau gegen Original (PyTorch, fp32)")
    checks = []
    fixed_inputs = []
    with torch.inference_mode():
        for i, call in enumerate(calls):
            frame = i + 1  # Bild 0 ist das Startbild ohne Memory-Attention
            args = to_fixed_inputs(call, pos512, tpos, frame)
            fixed_inputs.append(args)
            ref = call["out"][:, 0].reshape(1, HW, HW, D)
            row = {"bild": frame, "plaetze": int(call["spatial"]), "zeiger": call["ptr_tokens"] // TOK_PTR}
            for k, m in attn.items():
                row[k] = compare(m(*args), ref)
            checks.append(row)
            log(f"   Bild {frame:2d}: {row['plaetze']} Plätze, {row['zeiger']:2d} Zeiger, Abweichung rel "
                f"v1 {row['v1']['relL2']:.2e}, v2 {row['v2']['relL2']:.2e}")
    report["memoryAttention"] = checks
    worst = {k: max(r[k]["relL2"] for r in checks) for k in attn}
    assert max(worst.values()) < 1e-4, worst
    attn_last = (fixed_inputs[-1], calls[-1]["out"][:, 0].reshape(1, HW, HW, D))

    # Halbe Genauigkeit wie auf dem KI-Chip (grobe Abschätzung auf der CPU).
    try:
        with torch.inference_mode():
            m16 = MemAttnExportV2(sam, chunks=ATTN_CHUNKS).eval().half()
            out16 = m16(*(x.half() for x in attn_last[0])).float()
            report["memoryAttentionFp16"] = compare(out16, attn_last[1])
            log(f"   fp16-Abschätzung: rel {report['memoryAttentionFp16']['relL2']:.2e}")
    except Exception as e:  # nicht jede CPU kann fp16
        report["memoryAttentionFp16"] = f"nicht möglich: {e}"

    log("== Memory-Encoder und Vorbereitung für Qualcomms Memory-Encoder: gegen das Original")
    enc = MemEncExport(sam).eval()
    prep = MemPrepExport(size).eval()
    mem_cases = []  # (Eingänge, Metas Ergebnis) je Startbild/Folgebild
    enc_checks = []
    with torch.inference_mode():
        g = torch.Generator().manual_seed(3)
        feat = calls[-1]["curr"][:, 0].reshape(1, HW, HW, D)  # echte Merkmale aus dem Video
        for cond in (1.0, 0.0):
            logits = torch.randn(1, 1, 256, 256, generator=g) * 6
            logits[:, :, 80:170, 60:190] += 12  # ein Objekt in der Mitte
            hi = F.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)
            ref, _ = sam._encode_new_memory(
                current_vision_feats=[feat.reshape(1, N_Q, D).permute(1, 0, 2)], feat_sizes=[(HW, HW)],
                pred_masks_high_res=hi, object_score_logits=torch.ones(1, 1), is_mask_from_pts=bool(cond))
            args = (feat, logits.permute(0, 2, 3, 1).contiguous(), torch.full((1, 1, 1, 1), cond))
            r = compare(enc(*args), ref)
            r["start"] = bool(cond)
            # Vorbereitung: Merkmale nur umsortiert, Maske wie bei Meta vor Skalierung und Verschiebung.
            pix_p, mask_p = prep(*args)
            meta_mask = (hi > 0).float() if cond else torch.sigmoid(hi)
            off = (mask_p - meta_mask).abs()
            near_zero = int(((hi > 0) & (hi < 1e-4)).sum())
            r["vorbereitung"] = {"merkmaleGleich": bool(torch.equal(pix_p, feat.permute(0, 3, 1, 2))),
                                 "maskeMaxAbw": float(off.max()), "abweichendePixel": int((off > 1e-6).sum()),
                                 "logitsZwischen0und1e-4": near_zero}
            enc_checks.append(r)
            mem_cases.append((args, ref))
            log(f"   {'Startbild' if cond else 'Folgebild'}: eigener Memory-Encoder rel {r['relL2']:.2e}; "
                f"Vorbereitung: {r['vorbereitung']['abweichendePixel']} Pixel anders als bei Meta "
                f"({near_zero} Logits zwischen 0 und 0,0001)")
            assert r["vorbereitung"]["merkmaleGleich"]
            assert r["vorbereitung"]["abweichendePixel"] <= (near_zero if cond else 0), r["vorbereitung"]
    report["memoryEncoder"] = enc_checks
    assert max(r["relL2"] for r in enc_checks) < 1e-4

    log("== Eingabe-Einbettung für die App")
    pc = prompt_constants(sam)
    dev = max(float(np.abs(app_prompt(pc, p["u"], p["v"], p["labels"])[0] - np.asarray(p["einbettung"], np.float32)).max())
              for p in pc["proben"])
    report["einbettung"] = {"proben": len(pc["proben"]), "groessteAbweichungAppRechnung": dev}
    log(f"   Rechnung der App gegen Metas Code: größte Abweichung {dev:.2e}")
    assert dev < 1e-4, dev

    log("== Decoder wie im Original: Nachbau gegen Metas Videoverfolgung")
    # Zwei Fälle wie in der App: Startbild mit Klick (pix_feat mit no_mem_embed) und das letzte
    # Folgebild (Ausgang der Memory-Attention, kein Punkt). Erst Metas Rechnung, dann unsere.
    f0 = meta_features(sam, frames[0])
    fl = meta_features(sam, frames[-1])
    curr_last = attn_last[1]  # Ausgang der Memory-Attention im letzten Bild [1, 64, 64, 256]
    dec_cases = []
    with torch.inference_mode():
        for name, emb, feats, pts, labels in (
            ("start", f0["pix"], f0, [[start_uv[0] * size, start_uv[1] * size]], [1]),
            ("folge", curr_last, fl, None, None),
        ):
            hr = [feats["hr1"].permute(0, 3, 1, 2), feats["hr2"].permute(0, 3, 1, 2)]
            point_inputs = None if pts is None else {"point_coords": torch.tensor([pts], dtype=torch.float32),
                                                     "point_labels": torch.tensor([labels], dtype=torch.int32)}
            meta = sam._forward_sam_heads(emb.permute(0, 3, 1, 2), point_inputs, None, hr, multimask_output=True)
            sparse = torch.from_numpy(prompt_for(pc, 1, *(start_uv if pts else (0.0, 0.0)), 1 if pts else -1))
            dec_cases.append((name, (emb, feats["hr1"], feats["hr2"], sparse), meta))
    use_math_attention(sam)  # ab hier ohne scaled_dot_product_attention: gleiche Rechnung, sicher umzuwandeln
    dec = DecoderExport(sam).eval()
    dec_checks = []
    with torch.inference_mode():
        for name, args, meta in dec_cases:
            masks, ious, ptrs, score = dec(*args)
            best = int(torch.argmax(ious[0]))
            present = float(score.reshape(-1)[0]) > 0
            r = {"bild": name, "gewaehlt": best, "gewaehltMeta": int(torch.argmax(meta[2][0])), "objekt": present,
                 "ious": [round(float(x), 4) for x in ious[0]],
                 "maske": compare(masks[0, best] if present else torch.full_like(masks[0, best], -1024.0), meta[3][0, 0]),
                 "zeiger": compare(ptrs[0, best] if present else sam.no_obj_ptr.reshape(-1), meta[5][0]),
                 "objektWert": compare(score, meta[6])}
            dec_checks.append(r)
            log(f"   {name}: Maske {best} von 3 (Meta {r['gewaehltMeta']}), Maske rel {r['maske']['relL2']:.2e}, "
                f"Zeiger rel {r['zeiger']['relL2']:.2e}")
            assert r["gewaehlt"] == r["gewaehltMeta"] and r["maske"]["relL2"] < 1e-4 and r["zeiger"]["relL2"] < 1e-4, r
    report["decoder"] = dec_checks
    # Startbild wie in Metas Verfolgung des Prüfvideos?
    with torch.inference_mode():
        m0, i0, _, _ = dec(*dec_cases[0][1])
    report["decoderStartbildWieVerfolgung"] = round(iou(m0[0, int(torch.argmax(i0[0]))].numpy(), ref_masks[0]), 4)

    konst = {
        "version": 2,
        "noObjPtr": [float(x) for x in sam.no_obj_ptr.detach().reshape(-1)],
        "maskWert": MASK_VALUE,
        "plaetze": N_MEM, "tokensJePlatz": TOK_MEM, "zeiger": N_PTR, "tokensJeZeiger": TOK_PTR,
        "bildgroesse": size,
        "prompt": pc,
        "hinweise": {
            "platzOrdnung": "Platz 0 Startbild, Platz 7-n Bild vor n Schritten (n=1..6)",
            "zeigerGating": "object_score_logits <= 0: Maske -1024 und Zeiger = noObjPtr",
            "startMaske": "Beim Startbild cond=1 (Maske wird binär), sonst cond=0 (Sigmoid)",
            "einbettung": "Punkte u, v von 0 bis 1; x = 2u - 1 + 1/bildgroesse; [sin, cos](2π · (x, y) · gauss) + punkte[label]; "
                          "label -1: keinPunkt; am Ende immer ein keinPunkt",
        },
        "pruefvideo": {
            "datei": "edgetam_pruefvideo.zip", "bilder": a.frames, "start": start_uv,
            "quelle": "Synthetisches Video (Ball vor Rechtecken), verfolgt von Metas EdgeTAM auf der CPU; Masken = Logits > 0",
            "maskenAnteil": share, "objektWerte": [round(s, 3) for s in ref_scores],
        },
    }

    if not a.no_convert:
        log("== Umwandeln nach TFLite")
        # Zwischenspeicher der Positions-Kodierung stammen aus dem Prüflauf (inference_mode)
        # und dürfen nicht in die Umwandlung geraten.
        for m in sam.modules():
            if isinstance(getattr(m, "cache", None), dict):
                m.cache.clear()
        with torch.inference_mode():
            prep_refs = [prep(*args) for args, _ in mem_cases]
            enc_refs = [enc(*args) for args, _ in mem_cases]
            dec_refs = [(args, list(dec(*args))) for _, args, _ in dec_cases[::-1]]  # Folgebild zuerst
        targets = [
            ("edgetam_memory_attention.tflite", attn["v1"], [(attn_last[0], [attn_last[1]])]),
            ("edgetam_memory_attention_v2.tflite", attn["v2"], [(attn_last[0], [attn_last[1]])]),
            ("edgetam_memory_encoder.tflite", enc, [(args, [r]) for (args, _), r in zip(mem_cases[::-1], enc_refs[::-1])]),
            ("edgetam_memory_prep.tflite", prep, [(args, list(r)) for (args, _), r in zip(mem_cases[::-1], prep_refs[::-1])]),
            ("edgetam_decoder_meta.tflite", dec, dec_refs),
        ]
        report["tflite"] = {}
        models = {}
        for name, mod, cases in targets:
            path = os.path.join(a.out, name)
            # normale Tensoren statt inference_mode; umgewandelt wird mit dem ersten Fall (Folgebild)
            secs = convert(mod, tuple(x.clone() for x in cases[0][0]), path)
            m = TfliteModel(path)
            models[name] = m
            rows = []
            for args, refs in cases:
                outs = m.run(*args)
                for ref in refs:
                    hit = m.by_shape(outs, *ref.shape)
                    rows.append(compare(hit, ref))
            worst_rel = max(r["relL2"] for r in rows)
            report["tflite"][name] = {"relL2": worst_rel, "maxAbw": max(r["maxAbw"] for r in rows), "faelle": len(rows),
                                      "umwandlungS": round(secs, 1), "cpuMs": round(m.last_ms), "bytes": os.path.getsize(path),
                                      "schnittstelle": m.interface(), "operationen": m.ops()}
            log(f"   {name}: {os.path.getsize(path) / 1e6:.1f} MB, CPU {m.last_ms:.0f} ms, "
                f"Abweichung zum Nachbau rel {worst_rel:.2e}, {sum(report['tflite'][name]['operationen'].values())} Operationen")
            assert worst_rel < 1e-4, (name, rows)
            if name == "edgetam_memory_attention_v2.tflite":
                # Beide Produkte jeder Attention als BATCH_MATMUL; FULLY_CONNECTED nur für die 20 festen Schichten.
                ops = report["tflite"][name]["operationen"]
                assert ops.get("BATCH_MATMUL") == 2 * 2 * 2 * ATTN_CHUNKS and ops.get("FULLY_CONNECTED") == 20, ops

        if a.qualcomm:
            log("== Qualcomms Memory-Encoder hinter unserer Vorbereitung: gegen das Original")
            qc_mem = TfliteModel(os.path.join(a.qualcomm, "memory_encoder.tflite"))
            qc_rows = []
            for args, ref in mem_cases:
                got = QcMemory(models["edgetam_memory_prep.tflite"], qc_mem)(args[0], args[1].numpy(), bool(args[2].item()))
                r = compare(got.reshape(ref.shape), ref)
                r["start"] = bool(args[2].item())
                qc_rows.append(r)
                log(f"   {'Startbild' if r['start'] else 'Folgebild'}: rel {r['relL2']:.2e}, max {r['maxAbw']:.2e}")
            report["memoryEncoderQualcomm"] = qc_rows
            assert max(r["relL2"] for r in qc_rows) < 1e-3, qc_rows

            log("== Ketten der App auf der CPU nachrechnen (Prüfvideo)")
            qenc = CachedEncoder(QcEncoder(TfliteModel(os.path.join(a.qualcomm, "encoder.tflite"))), frames)
            qdec = QcDecoder(TfliteModel(os.path.join(a.qualcomm, "video_decoder.tflite")))
            mdec = MetaDecoder(models["edgetam_decoder_meta.tflite"])
            no_ptr = konst["noObjPtr"]
            own = OwnMemory(models["edgetam_memory_encoder.tflite"])
            qcm = QcMemory(models["edgetam_memory_prep.tflite"], qc_mem)
            v1 = models["edgetam_memory_attention.tflite"]
            v2 = models["edgetam_memory_attention_v2.tflite"]
            # Schlüssel wie in der App (AiEngine.videoCheck); „/eigener“ und „/qualcomm“: welcher
            # Memory-Encoder (die App nimmt live den schnelleren, rechnerisch sind beide gleich).
            chains = {
                "stufe2": AppChain("Stufe 2: Qualcomms Decoder, Einbettung vom Encoder", qenc, qdec, v1, own, no_ptr),
                "voll/eigener": AppChain("voll · Decoder wie Meta · Memory-Encoder eigener", qenc, mdec, v2, own, no_ptr, pc),
                "voll/qualcomm": AppChain("voll · Decoder wie Meta · Memory-Encoder qualcomm", qenc, mdec, v2, qcm, no_ptr, pc),
                "vollQualcomm/eigener": AppChain("voll · Qualcomms Decoder · Memory-Encoder eigener", qenc, qdec, v2, own, no_ptr, pc),
                "vollQualcomm/qualcomm": AppChain("voll · Qualcomms Decoder · Memory-Encoder qualcomm", qenc, qdec, v2, qcm, no_ptr, pc),
                "schnell": FastChain("schnell · Decoder wie Meta", qenc, mdec, pc),
            }
            if a.qualcomm_w8a8:
                q8 = CachedEncoder(QcEncoderW8a8(TfliteModel(os.path.join(a.qualcomm_w8a8, "encoder.tflite"))), frames)
                chains["schnellQuantisiert"] = FastChain("schnell · Encoder w8a8 · Decoder wie Meta", q8, mdec, pc)
            cpu = {}
            for key, chain in chains.items():
                r = track_video(chain, frames, start_uv, ref_masks)
                top, _, sub = key.partition("/")
                if sub:
                    cpu.setdefault(top, {})[sub] = r
                else:
                    cpu[top] = r
                log(f"   {key:22s} gleich mittel {r['maskeGleichMittel']:.3f}, schlechtestes "
                    f"{r['maskeGleichSchlechtestes']:.3f}  {r['jeBild']}")
            konst["pruefvideo"]["cpuKetten"] = cpu

    log("== Prüfvideo schreiben")
    video_path = os.path.join(a.out, konst["pruefvideo"]["datei"])
    write_test_video(video_path, frames, ref_masks)
    log(f"   {os.path.getsize(video_path) / 1e6:.1f} MB")

    konst["pruefung"] = report
    with open(os.path.join(a.out, "edgetam_konstanten.json"), "w") as f:
        json.dump(konst, f, indent=1, ensure_ascii=False)
    log("== Fertig")


if __name__ == "__main__":
    main()
