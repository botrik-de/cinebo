#!/usr/bin/env python3
"""Exportiert die Teile von EdgeTAM, die Qualcomm nicht fertig anbietet.

Qualcomm AI Hub liefert Encoder, Decoder und Memory-Encoder von EdgeTAM als
TFLite-Modelle. Die Memory-Attention, also der Teil, der ein Objekt über viele
Bilder hinweg wiedererkennt, fehlt dort (Qualcomm rechnet sie auf dem PC). Dieses
Skript baut sie mit festen Formen nach, damit sie auf dem KI-Chip des Handys laufen
kann, und prüft das Ergebnis Zahl für Zahl gegen das Original-Modell von Meta.

Erzeugt im Ausgabeordner:
  edgetam_memory_attention.tflite      Memory-Attention, 7 Speicherplätze, 16 Zeiger
  edgetam_memory_attention_q4.tflite   gleiche Rechnung, Abfragen in 4 Blöcken
  edgetam_memory_encoder.tflite        Memory-Encoder mit NHWC-Eingang und Maske 256x256
  edgetam_konstanten.json              Konstanten für die App und Prüfergebnisse

Aufruf:
  python edgetam_export.py --repo EdgeTAM --checkpoint EdgeTAM/checkpoints/edgetam.pt --out out

Schnittstellen (Reihenfolge der Eingänge zählt, Namen vergibt der Konverter):
  Memory-Attention
    0 curr      [1,64,64,256]  image_embeddings des Encoders (ohne no_mem_embed)
    1 maskmem   [1,3584,64]    7 Speicherplätze zu je 512 Tokens. Platz 0 ist das
                               Startbild, Platz 1 bis 6 die letzten Bilder, Platz 6
                               ist das Bild direkt davor.
    2 obj_ptrs  [1,16,256]     Objekt-Zeiger, Reihenfolge egal
    3 bias      [1,1,3648]     0 für gültige Tokens, -10000 für leere Plätze
    Ausgang     [1,64,64,256]  geht als image_embeddings in den Decoder
  Memory-Encoder
    0 pix_feat  [1,64,64,256]  image_embeddings des Encoders (ohne no_mem_embed)
    1 mask      [1,256,256,1]  Masken-Logits des Decoders
    2 cond      [1,1,1,1]      1 beim angetippten Startbild, sonst 0
    Ausgang     [1,512,64]     neuer Speicher-Eintrag
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


def run_reference(model, folder: str, point):
    """Lässt das Original verfolgen und schreibt jeden Aufruf der Memory-Attention mit."""
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
    masks = {}
    with torch.inference_mode():
        state = model.init_state(video_path=folder)
        model.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=np.array([point], dtype=np.float32),
                                    labels=np.array([1], dtype=np.int32))
        for f, _, logits in model.propagate_in_video(state):
            masks[f] = (logits[0, 0] > 0).cpu().numpy()
    model.memory_attention.forward = orig
    return calls, masks


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


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double(), b.double()
    diff = (a - b).abs()
    return {"maxAbw": float(diff.max()), "relL2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
            "bereich": [float(b.min()), float(b.max())]}


# --------------------------------------------------------------------------


def convert(module: nn.Module, args: tuple, path: str) -> float:
    import litert_torch

    t = time.time()
    edge = litert_torch.convert(module.eval(), args)
    edge.export(path)
    return time.time() - t


def run_tflite(path: str, args: tuple):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path, num_threads=os.cpu_count() or 2)
    it.allocate_tensors()
    ins = it.get_input_details()
    assert len(ins) == len(args), [d["name"] for d in ins]
    info = []
    for d, a in zip(ins, args):  # Reihenfolge der Eingänge wie beim Umwandeln
        arr = a.detach().numpy().astype(np.float32)
        assert list(d["shape"]) == list(arr.shape), (d["name"], d["shape"], arr.shape)
        it.set_tensor(d["index"], arr)
        info.append({"name": d["name"], "shape": [int(x) for x in d["shape"]], "dtype": str(d["dtype"].__name__)})
    t = time.time()
    it.invoke()
    dt = time.time() - t
    outs = it.get_output_details()
    assert len(outs) == 1, [d["name"] for d in outs]
    out = torch.from_numpy(it.get_tensor(outs[0]["index"]).copy())
    info.append({"ausgang": outs[0]["name"], "shape": [int(x) for x in outs[0]["shape"]]})
    return out, dt, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
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

    log("== Original verfolgen lassen")
    with tempfile.TemporaryDirectory() as tmp:
        track, _ = make_video(tmp, a.frames)
        t = time.time()
        calls, masks = run_reference(model, tmp, track[0])
        log(f"   {len(calls)} Aufrufe der Memory-Attention in {time.time() - t:.0f} s")
    # Der Ball muss im letzten Bild noch gefunden werden, sonst taugt die Prüfung nichts.
    last = masks[a.frames - 1]
    cx, cy = track[-1]
    hit = bool(last[int(cy), int(cx)])
    report["verfolgung"] = {"bilder": a.frames, "ballImLetztenBild": hit, "maskenAnteil": float(last.mean())}
    log(f"   Ball im letzten Bild getroffen: {hit}, Maskenanteil {last.mean():.3f}")
    assert hit, "Das Original verliert den Ball, Prüfvideo taugt nicht"

    pos512 = maskmem_pos_512(sam)
    tpos = sam.maskmem_tpos_enc.detach().reshape(N_MEM, MEM_DIM)
    attn = {c: MemAttnExport(sam, chunks=c).eval() for c in (1, 4)}
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
            for c, m in attn.items():
                row[f"block{c}"] = compare(m(*args), ref)
            checks.append(row)
            log(f"   Bild {frame:2d}: {row['plaetze']} Plätze, {row['zeiger']:2d} Zeiger, "
                f"Abweichung max {row['block1']['maxAbw']:.2e}, rel {row['block1']['relL2']:.2e}")
    report["memoryAttention"] = checks
    worst = max(r["block1"]["relL2"] for r in checks)
    worst4 = max(r["block4"]["relL2"] for r in checks)
    assert worst < 1e-4 and worst4 < 1e-4, (worst, worst4)

    # Halbe Genauigkeit wie auf dem KI-Chip (grobe Abschätzung auf der CPU).
    try:
        with torch.inference_mode():
            m16 = MemAttnExport(sam, chunks=1).eval().half()
            args = fixed_inputs[-1]
            out16 = m16(*(x.half() for x in args)).float()
            ref = calls[-1]["out"][:, 0].reshape(1, HW, HW, D)
            report["memoryAttentionFp16"] = compare(out16, ref)
            log(f"   fp16-Abschätzung: rel {report['memoryAttentionFp16']['relL2']:.2e}")
    except Exception as e:  # nicht jede CPU kann fp16
        report["memoryAttentionFp16"] = f"nicht möglich: {e}"

    log("== Memory-Encoder: Nachbau gegen Original")
    enc = MemEncExport(sam).eval()
    enc_checks = []
    with torch.inference_mode():
        g = torch.Generator().manual_seed(3)
        feat = calls[-1]["curr"][:, 0].reshape(1, HW, HW, D)  # echte Merkmale aus dem Video
        for cond in (1.0, 0.0):
            logits = torch.randn(1, 1, 256, 256, generator=g) * 6
            logits[:, :, 80:170, 60:190] += 12  # ein Objekt in der Mitte
            hi = F.interpolate(logits, size=(sam.image_size,) * 2, mode="bilinear", align_corners=False)
            ref, _ = sam._encode_new_memory(
                current_vision_feats=[feat.reshape(1, N_Q, D).permute(1, 0, 2)], feat_sizes=[(HW, HW)],
                pred_masks_high_res=hi, object_score_logits=torch.ones(1, 1), is_mask_from_pts=bool(cond))
            mine = enc(feat, logits.permute(0, 2, 3, 1), torch.full((1, 1, 1, 1), cond))
            r = compare(mine, ref)
            r["start"] = bool(cond)
            enc_checks.append(r)
            log(f"   {'Startbild' if cond else 'Folgebild'}: Abweichung max {r['maxAbw']:.2e}, rel {r['relL2']:.2e}")
        enc_args = (feat, logits.permute(0, 2, 3, 1).contiguous(), torch.zeros(1, 1, 1, 1))
        enc_ref = mine
    report["memoryEncoder"] = enc_checks
    assert max(r["relL2"] for r in enc_checks) < 1e-4

    konst = {
        "version": 1,
        "noObjPtr": [float(x) for x in sam.no_obj_ptr.detach().reshape(-1)],
        "maskWert": MASK_VALUE,
        "plaetze": N_MEM, "tokensJePlatz": TOK_MEM, "zeiger": N_PTR, "tokensJeZeiger": TOK_PTR,
        "bildgroesse": int(sam.image_size),
        "hinweise": {
            "platzOrdnung": "Platz 0 Startbild, Platz 7-n Bild vor n Schritten (n=1..6)",
            "zeigerGating": "object_score_logits <= 0: Maske -1024 und Zeiger = noObjPtr",
            "startMaske": "Beim Startbild cond=1 (Maske wird binär), sonst cond=0 (Sigmoid)",
        },
    }

    if not a.no_convert:
        log("== Umwandeln nach TFLite")
        targets = [
            ("edgetam_memory_attention.tflite", attn[1], fixed_inputs[-1], calls[-1]["out"][:, 0].reshape(1, HW, HW, D)),
            ("edgetam_memory_attention_q4.tflite", attn[4], fixed_inputs[-1], calls[-1]["out"][:, 0].reshape(1, HW, HW, D)),
            ("edgetam_memory_encoder.tflite", enc, enc_args, enc_ref),
        ]
        report["tflite"] = {}
        # Zwischenspeicher der Positions-Kodierung stammen aus dem Prüflauf (inference_mode)
        # und dürfen nicht in die Umwandlung geraten.
        for m in sam.modules():
            if isinstance(getattr(m, "cache", None), dict):
                m.cache.clear()
        for name, mod, args, ref in targets:
            args = tuple(x.clone() for x in args)  # normale Tensoren statt inference_mode
            path = os.path.join(a.out, name)
            secs = convert(mod, args, path)
            out, dt, info = run_tflite(path, args)
            r = compare(out.reshape(ref.shape), ref)
            r.update({"umwandlungS": round(secs, 1), "cpuMs": round(dt * 1000), "bytes": os.path.getsize(path), "schnittstelle": info})
            report["tflite"][name] = r
            log(f"   {name}: {os.path.getsize(path) / 1e6:.1f} MB, CPU {dt * 1000:.0f} ms, "
                f"Abweichung zum Nachbau rel {r['relL2']:.2e}")
            assert r["relL2"] < 1e-4, r

    konst["pruefung"] = report
    with open(os.path.join(a.out, "edgetam_konstanten.json"), "w") as f:
        json.dump(konst, f, indent=1, ensure_ascii=False)
    log("== Fertig")


if __name__ == "__main__":
    main()
