@echo off
title ComfyUI - AMD RDNA4 (R9700 / gfx1201)

REM ===========================================================================
REM  Place this file in your ComfyUI portable ROOT -- the folder that contains
REM  both python_embeded\ and ComfyUI\ -- then double-click it.
REM  %~dp0 resolves to that folder, so there are no paths to edit.
REM ===========================================================================
cd /d "%~dp0"

if not exist ".\python_embeded\python.exe" (
    echo.
    echo   ERROR: python_embeded\python.exe not found.
    echo   Put this .bat in the ComfyUI portable root ^(next to python_embeded^).
    echo.
    pause
    exit /b 1
)

REM --- Environment -----------------------------------------------------------

REM Gives SDPA a real flash-attention kernel on RDNA4. Without it SDPA silently
REM falls back to the math backend, which allocates a full attention matrix.
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

set MIOPEN_FIND_MODE=FAST

REM Persists Inductor's autotuned Triton kernels. The MiniMax-H3 patch compiles
REM an autotuned flex_attention kernel (1.51x over AOTriton SDPA on gfx1201);
REM without a stable cache dir the ~40 s autotune is repaid every launch.
REM Set H3_FLEX_ATTENTION=off to fall back to stock SDPA.
set TORCHINDUCTOR_CACHE_DIR=%~dp0.inductor_cache

REM --- Launch ----------------------------------------------------------------
REM
REM  --disable-smart-memory   THE BIG ONE. Without it ComfyUI keeps the 14.6 GB
REM      text encoder partially resident while the 20 GB DiT samples, leaving
REM      under ~3 GB free of 32 GB. At that pressure Windows/WDDM demand paging
REM      makes large GEMMs collapse -- mlp.fc2 measured 116 ms per call in situ
REM      versus 5.3 ms for the identical shape standalone. Measured effect on a
REM      608x352/56f render: 8.97 s/it -> 1.71 s/it (5.2x). This flag changes
REM      only WHEN models are evicted; it alters no weight, dtype or kernel, so
REM      it cannot change your output.
REM
REM  --bf16-vae               fixes fp16 overflow in the video VAE's temporal
REM      convolutions (the flat-grey-frames bug). bf16 has fp32's exponent range
REM      so it cannot overflow, at roughly half the cost of fp32.
REM
REM  --disable-dynamic-vram   comfy-aimdo is an NVIDIA-only path; forcing it on
REM      AMD produces runs where weights are "Staged" but never materialise.
REM
REM  Do NOT add --use-quad-cross-attention: ComfyUI already enables SDPA for
REM  gfx1200/gfx1201 on ROCm >= 7.0, and that flag overrides it with the slow
REM  sub-quadratic path.
REM
REM  Do NOT add --use-sage-attention: measured SLOWER on this card
REM  (39.3 vs 50.2 TFLOPS at H3's attention shape).

.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build ^
    --disable-dynamic-vram --bf16-vae --reserve-vram 2 ^
    --disable-smart-memory --auto-launch

pause
