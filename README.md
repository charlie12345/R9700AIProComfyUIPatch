# R9700 AI Pro — ComfyUI / MiniMax-H3 speed patches

Make **MiniMax-H3** video generation fast on an **AMD Radeon AI PRO R9700** (gfx1201 / RDNA 4) under Windows.

---

> ## 🙏 Thank you to AMD
>
> This work was made possible by the **AMD Ryzen Threadripper 9980X** and the
> **AMD Radeon AI PRO R9700**. Every measurement, patch and workflow in this
> repository was developed and benchmarked on that hardware.
>
> Thank you to AMD for building silicon that makes local, open-source video
> generation genuinely practical.

---

> ### ⚠️ Updating ComfyUI removes these patches
>
> A ComfyUI update overwrites `comfy/ldm/minimax/model.py` and **silently reverts
> both patches** — no error, no warning, you just quietly lose the speed.
> **Re-run the patch script after every ComfyUI update:**
>
> ```bash
> python apply_h3_rdna4_patches.py --comfy-path C:\path\to\ComfyUI
> ```
>
> Not sure whether they're applied? `python apply_h3_rdna4_patches.py --check`
> See [After a ComfyUI update](#6-after-a-comfyui-update).

---

Two small patches to ComfyUI's H3 model file, plus the launcher settings that matter. Everything here was measured on real renders, not estimated.

| 864×480, 124 frames (5.2 s), 20 steps | time |
|---|---|
| stock ComfyUI | **386 s** |
| + these patches + launch flags | **272 s** (1.42x) |
| + the 4-step Turbo LoRA (third-party, optional) | **~85 s** (~4.5x) |

The single biggest win is not a patch at all — it's one launcher flag. See [Launch settings](#3-launch-settings).

---

## Contents

1. [Requirements](#1-requirements)
2. [Install](#2-install)
3. [Launch settings](#3-launch-settings)
4. [What the patches actually do](#4-what-the-patches-actually-do)
5. [Check it's working](#5-check-its-working)
6. [After a ComfyUI update](#6-after-a-comfyui-update)
7. [Speed tuning: the token budget](#7-speed-tuning-the-token-budget)
8. [Optional: the 4-step Turbo LoRA](#8-optional-the-4-step-turbo-lora)
9. [Caveats — read before trusting the numbers](#9-caveats--read-before-trusting-the-numbers)
10. [Uninstall](#10-uninstall)

---

## 1. Requirements

**Hardware.** Built and measured on:

| | |
|---|---|
| CPU | **AMD Ryzen Threadripper 9980X** |
| GPU | **AMD Radeon AI PRO R9700** (gfx1201, 32 GB) |
| RAM | 128 GB |

Should also help other RDNA 4 (gfx1200/1201) and probably RDNA 3 (gfx1100–1103, gfx1150–1153) cards — comfy-kitchen's HIP backend covers those too, but they are untested here.

The 32 GB of VRAM is load-bearing for the tuning numbers: several findings below (the token budget, the `--disable-smart-memory` cliff) are consequences of a 20 GB model and a 14.6 GB text encoder not fitting together in 32 GB.

**Software.** The versions this was measured against:

| | version |
|---|---|
| OS | Windows 11 |
| PyTorch | 2.9.1+rocm7.2.1 |
| HIP | 7.2.53211 |
| ComfyUI | **0.30.0 or newer** (H3 support landed in 0.30.0) |
| comfy-kitchen | **0.2.26 or newer** — this is the important one |
| Triton | 3.7.1 or newer |

> **comfy-kitchen must be ≥ 0.2.26.** Version 0.2.26 introduced a native **HIP/WMMA backend** that auto-enables on gfx1201. Patch 1 depends on it. On 0.2.22 the patch safely does nothing.

Check yours:

```bash
python -c "import importlib.metadata as m; print('comfy-kitchen', m.version('comfy-kitchen'))"
python -c "import torch; print(torch.__version__, torch.version.hip)"
```

**Models.** These patches only affect MiniMax-H3. You need the H3 weights — the `pruned_fp8_scaled` DiT is the right pick on this card (fp8 measured **235 TFLOPS** vs 166 for int8 and 118 for bf16).

---

## 2. Install

```bash
git clone https://github.com/charlie12345/R9700AIProComfyUIPatch.git
cd R9700AIProComfyUIPatch
python apply_h3_rdna4_patches.py --comfy-path C:\path\to\ComfyUI
```

`--comfy-path` should point at the folder containing `comfy/ldm/minimax/model.py`. In a portable install that's `ComfyUI_windows_portable\ComfyUI`. If you drop the script into the portable root it auto-detects and you can omit the flag.

Then **fully restart ComfyUI** — a browser refresh does not reload Python.

Useful flags:

```bash
python apply_h3_rdna4_patches.py --check    # report status, change nothing
python apply_h3_rdna4_patches.py --revert   # restore stock
```

The script is idempotent, writes a `.orig-backup` before its first edit, and **refuses to apply if the upstream file has changed** (every anchor must match exactly once) — so it will not silently mangle a newer ComfyUI.

---

## 3. Launch settings

**This section matters more than the patches.** Copy these into your launcher `.bat`:

```bat
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
set MIOPEN_FIND_MODE=FAST
set TORCHINDUCTOR_CACHE_DIR=%~dp0.inductor_cache

.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build ^
    --disable-dynamic-vram --bf16-vae --reserve-vram 2 ^
    --disable-smart-memory --auto-launch
```

A ready-made `Launch ComfyUI AMD RDNA4.bat` is included in this repo.

### Why each one

**`--disable-smart-memory`** — the biggest single win, and it is not a patch.

Without it, ComfyUI keeps the 14.6 GB text encoder partially resident while the 20 GB DiT samples, leaving under ~3 GB free of 32 GB. At that pressure Windows/WDDM demand paging makes large GEMMs collapse:

| | in situ | same shape, standalone |
|---|---|---|
| `mlp.fc2` bf16 GEMM (M=3771, K=14336, N=5376) | **116 ms** | **5.3 ms** |

That is ~22x. Proven environmental rather than a bad kernel: freshly allocated tensors of identical shape at the same point in the loop were equally slow, and an M-sweep showed no alignment cliff.

Measured effect on a 608×352 / 56-frame render: **8.97 s/it → 1.71 s/it (5.2x)**.
This flag changes *when* models are evicted — no weight, dtype or kernel changes, so it cannot alter your output.

**`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** — without it, SDPA silently falls back to the math backend on RDNA 4, which allocates a full attention matrix and is dramatically slower.

**`TORCHINDUCTOR_CACHE_DIR`** — Patch 2 autotunes a Triton attention kernel. That search takes ~40 s the first time a given sequence length is seen. This makes the result persist across restarts instead of being repaid every launch.

**`--bf16-vae`** — fixes fp16 overflow in the video VAE's temporal convolutions (the classic flat-grey-frames bug). bf16 has fp32's exponent range so it cannot overflow, at roughly half the cost of fp32.

**`--disable-dynamic-vram`** — `comfy-aimdo` is an NVIDIA-only path; forcing it on AMD produces runs where weights are staged but never materialise.

### Do **not** use SageAttention

ComfyUI's docs suggest SageAttention for H3. On this card it is **slower**: 39.3 TFLOPS vs 50.2 for plain SDPA at H3's attention shape. Measured, not assumed.

---

## 4. What the patches actually do

Both are defensively gated — if the required backend is missing they return to stock behaviour rather than failing. The patched file is safe on any GPU.

### Patch 1 — partial-rope HIP fast path (**bit-exact**)

MiniMax-H3 uses *partial* rotary embedding: `rot_dim` 96 inside a 128-wide head. comfy-kitchen's fused RMSNorm+rope WMMA kernel only accepts `rot_dim == head_dim`, and hands anything else to a pure-PyTorch fallback — so **all 52 attention blocks silently lose their kernel**.

The fix splits the operation: full-width RMSNorm, then the HIP rope on just the rotated prefix. Same maths, kernel restored.

| | |
|---|---|
| rope op alone | **1.97–2.04x** (9.34 ms → 4.74 ms at L=8192) |
| full attention block | 1.07x @ L=8192, 1.18x @ L=2048 |
| output difference | **`max_abs_diff = 0.0`** — bit-identical |

Verified by running patched and stock `Attention` modules side by side with identical weights and inputs.

### Patch 2 — autotuned Triton attention (**not** bit-exact)

Routes H3's long-sequence attention through `flex_attention` compiled with `max-autotune`, instead of AOTriton SDPA. Inductor searches Triton configs for *your* GPU and picks, on gfx1201, `BLOCK_M=128 / BLOCK_N=16 / num_warps=4` — `BLOCK_N=16` matches RDNA 4's WMMA width, and AOTriton ships no equivalent config.

Measured at H3's real attention shape (B=1, H=56, L=15488, D=128, bf16):

| backend | time | throughput | |
|---|---|---|---|
| SDPA (AOTriton flash) | 141.7 ms | 48.6 TFLOPS | baseline |
| flex_attention, default configs | 120.7 ms | 57.0 TFLOPS | 1.17x |
| **flex_attention, max-autotune** | **94.1 ms** | **73.1 TFLOPS** | **1.51x** |

The autotune is most of the win — default flex configs only give 1.17x.

> **This one changes your output.** Per call the difference is 2.4e-4 (bf16 rounding from a different accumulation order — *not* reduced precision). But diffusion compounds it: a full 20-step 864×480 render lands at **~23.5 dB PSNR** against the SDPA render. That is a **different sample from the same distribution** — equivalent quality, not the same frames. Set `H3_FLEX_ATTENTION=off` to reproduce stock output exactly.

---

## 5. Check it's working

Start ComfyUI and run any H3 workflow. In the console you should see:

```
MiniMax-H3: using autotuned flex_attention (H3_FLEX_ATTENTION=off to disable)
```

And when the DiT loads, confirm it is **not** offloading:

```
loaded completely; 25519.09 MB usable, 19984.52 MB loaded, full load: True
```

**`loaded partially` or `offloaded` is the warning sign** — it means weights are streaming over PCIe every layer, every step, and you will be 5–20x slower. Fix by reducing tokens (see below) or confirming `--disable-smart-memory` is on.

Status check without running anything:

```bash
python apply_h3_rdna4_patches.py --check
```

---

## 6. After a ComfyUI update

A ComfyUI update overwrites `comfy/ldm/minimax/model.py` and **silently reverts both patches**. Re-run:

```bash
python apply_h3_rdna4_patches.py --comfy-path C:\path\to\ComfyUI
```

If the upstream file has changed, the script **refuses rather than guessing** — every anchor must match exactly once. If that happens the patches need re-deriving; the docstring at the top of the script describes each edit precisely.

---

## 7. Speed tuning: the token budget

**This is what governs speed — not resolution alone.** More useful than any patch once you're set up.

```
tokens   = latent_t × (width / 32) × (height / 32)
latent_t = ((frames − 5) // 17) × 5 + 2
```

Measured knee on a 32 GB card. It is a **cliff, not a slope** — 20 GB of weights leaves only ~5.5 GB for activations:

| tokens | example | s/it |
|---|---|---|
| 14,985 | 864×480, 124f (5.2 s) | **13.5** |
| 21,060 | 864×480, 175f (7.3 s) | **~76** |
| 52,020 | 960×544, 345f (14.4 s) | model only partially loads — unusable |

**Target ≤ ~17,000 tokens.**

- good: `864×480 @ 124f` (5.2 s) = 14,985
- good: `640×352 @ 243f` (10.1 s) = 15,840
- trap: `864×480 @ 243f` (10.1 s) = 29,160

**Duration and resolution trade against each other.** Render small and upscale afterwards.

Also: width/height must be multiples of 32; frame count snaps to H3's `17k+5` grid; max useful length is ~362 frames (15.1 s); and H3 is a **768p model** — larger canvases are *not* downscaled by the node and will not fit.

### First render always feels slow — this is normal

| one-time cost | when |
|---|---|
| ~65 s text encode | new prompt text (changing only seed/steps reuses it) |
| 25 s – 3 min attention autotune | new canvas or frame count (then cached **permanently**) |
| ~50 s checkpoint load | new model file (cached in RAM until restart) |

Real example: a render took 190 s the first time and **50 s** on re-run with only the seed changed. So: lock resolution early, write the prompt, then iterate seeds freely.

---

## 8. Optional: the 4-step Turbo LoRA

Not part of this repo, but it stacks cleanly and is where the ~4.5x comes from.

- LoRA: [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
- Nodes: [`Larryvrh/ComfyUI-MiniMax-H3-Turbo`](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)

**No AMD porting needed** — it's device-agnostic PyTorch with zero dependencies. It applies via *bypass* (runs the base module's own forward, then adds the low-rank delta in activation space), so **the native fp8 GEMM path survives** and it coexists with Patch 2.

Two things worth knowing:

- **Keep steps at 4.** It is step-*distilled* for the schedule `[1.0, 0.973, 0.9231, 0.8, 0.0]`; that single large `0.8 → 0.0` jump is the whole trick. At 8 steps ComfyUI inserts a `0.6316` waypoint the LoRA has never seen — confirmed both visually worse and objectively noisier. This is the opposite of normal sampling intuition.
- **Artifacting is fixed by *lowering* strength**, not sharpening — it is over-sharp by design. 1.00 → 0.85 → 0.70, cleanest at 0.70.

It also works on the `ref2va` checkpoint (multi-character reference images) — all 259 LoRA modules exist there and both checkpoints have identical module sets — so multi-character work doesn't cost you 4-step speed.

The author describes it as preview-stage; treat it accordingly.

---

## 9. Caveats — read before trusting the numbers

**Measured on one machine.** Everything above comes from a single R9700 on the stack in [Requirements](#1-requirements). The *method* generalises further than the numbers do.

**`--disable-smart-memory` is Windows-specific.** The 5.2x comes from WDDM demand paging. On Linux ROCm the memory manager behaves differently and the cliff may be smaller or absent. It is also **canvas-dependent**: at larger canvases ComfyUI evicts the encoder on its own, so the flag is closer to break-even there. The 5.2x figure is from a *small* canvas where ComfyUI over-retains.

**The token budget is 32 GB-specific.** A card with more VRAM moves the knee.

**Patch 2 changes your output** (see above). Patch 1 does not.

**Version sensitivity.** comfy-kitchen's HIP backend did not exist before 0.2.26, and a torch/ROCm upgrade could change the picture again — including possibly fixing `torch._int_mm`, which is currently broken on gfx1201 (it segfaults, exit 139, no traceback) but is no longer on ComfyUI's path because the HIP backend serves int8 itself.

**If you're on different hardware:** apply the patches (they're safely gated) but **re-measure before trusting any number here.** The transferable part is the method — find the VRAM cliff, autotune attention, and check whether "fused" kernels are silently falling back to eager.

---

## 10. Uninstall

```bash
python apply_h3_rdna4_patches.py --comfy-path C:\path\to\ComfyUI --revert
```

Restores `model.py` from the `.orig-backup` written on first apply. Then restart ComfyUI.

---

## Credits

Patches and measurements by the repo owner, developed against ComfyUI 0.30.0 on an
**AMD Ryzen Threadripper 9980X** and **AMD Radeon AI PRO R9700** — with thanks to AMD
for the hardware that made it possible.

MiniMax-H3 by MiniMax; ComfyUI by Comfy Org; Turbo LoRA by larryvrh.

## License

The contents of this repository — the patch script, the helper script, the
launcher and the example workflows — are released under the
**[MIT License](LICENSE)**. Use them freely.

**What this repo does *not* contain:** any ComfyUI source. `apply_h3_rdna4_patches.py`
edits your local copy of `comfy/ldm/minimax/model.py` in place; no modified
ComfyUI file is redistributed here. ComfyUI itself is **GPL-3.0** and stays
under its own licence, as do MiniMax-H3 (MiniMax H3 Community Licence) and the
third-party Turbo LoRA and its node pack (Apache-2.0). Nothing in this repo
changes the terms of any of those.
